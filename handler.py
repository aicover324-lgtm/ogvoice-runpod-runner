import os
import json
import shutil
import subprocess
import zipfile
import time
import logging
import inspect
from pathlib import Path
from urllib.request import Request, urlopen

import boto3
import runpod
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


APPLIO_DIR = Path("/content/Applio")
WORK_DIR = Path("/workspace")
PREREQ_MARKER = APPLIO_DIR / ".prerequisites_ready"

# Always use these advanced pretrained weights (32k).
# Default: baked into the image at /content/Applio/pretrained_custom/*.pth
CUSTOM_PRETRAIN_DIR = APPLIO_DIR / "pretrained_custom"
CUSTOM_PRETRAIN_G_PATH = CUSTOM_PRETRAIN_DIR / "G_15.pth"
CUSTOM_PRETRAIN_D_PATH = CUSTOM_PRETRAIN_DIR / "D_15.pth"

# URLs are overridable via env vars for painless upgrades.
CUSTOM_PRETRAIN_G_URL = os.environ.get(
    "CUSTOM_PRETRAIN_G_URL",
    "https://huggingface.co/OrcunAICovers/legacy_core_pretrain_v1.5/resolve/main/G_15.pth?download=true",
)
CUSTOM_PRETRAIN_D_URL = os.environ.get(
    "CUSTOM_PRETRAIN_D_URL",
    "https://huggingface.co/OrcunAICovers/legacy_core_pretrain_v1.5/resolve/main/D_15.pth?download=true",
)

# Advanced pretrains are 32k; we force sample rate to match for consistency.
FORCED_SR_TAG = os.environ.get("FORCED_SR_TAG", "32k")
FORCED_SR = int(os.environ.get("FORCED_SR", "32000"))

# Opinionated defaults for training initialization.
# These align with the desired baseline settings and reduce variability.
FORCE_VOCODER = "HiFi-GAN"
FORCE_CUT_PREPROCESS = "Automatic"
FORCE_NORMALIZATION_MODE = "post"
FORCE_F0_METHOD = "rmvpe"
FORCE_EMBEDDER_MODEL = "contentvec"
FORCE_INCLUDE_MUTES = 2
FORCE_BATCH_SIZE = 4
FORCE_INDEX_ALGORITHM = "Auto"

# "Noise filter" in Applio UI maps to the preprocess flag `--process_effects`.
# This applies a simple filter during preprocessing.
FORCE_PROCESS_EFFECTS = True

# "Noise filter" in Applio UI maps to the preprocess flag `--process_effects`.
# This applies a simple filter during preprocessing.
FORCE_PROCESS_EFFECTS = True

# Noise settings:
# - "noise filter": on (if Applio supports a flag for it)
# - noise reduction: off
FORCE_NOISE_FILTER = True
FORCE_NOISE_REDUCTION = False

VOCAL_SEP_MODEL = os.environ.get(
    "VOCAL_SEP_MODEL",
    "vocals_mel_band_roformer.ckpt",
)
KARAOKE_SEP_MODEL = os.environ.get(
    "KARAOKE_SEP_MODEL",
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
)


def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def as_bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def as_int(v, default: int) -> int:
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        return int(float(s))
    return default


def as_float(v, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        return float(s)
    return default


def s3():
    endpoint = require_env("R2_ENDPOINT").rstrip("/")
    access_key = require_env("R2_ACCESS_KEY_ID")
    secret_key = require_env("R2_SECRET_ACCESS_KEY")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def run(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-8000:]
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{tail}")
    return p.stdout or ""


def get_gpu_diagnostics():
    info = {
        "torchAvailable": _TORCH_AVAILABLE,
        "cudaAvailable": False,
        "cudaDeviceCount": 0,
        "cudaVersion": None,
        "deviceNames": [],
        "nvidiaSmi": None,
    }

    # Best signal: torch sees CUDA
    if _TORCH_AVAILABLE and torch is not None:
        try:
            info["cudaAvailable"] = bool(torch.cuda.is_available())
            info["cudaDeviceCount"] = int(torch.cuda.device_count()) if info["cudaAvailable"] else 0
            info["cudaVersion"] = getattr(getattr(torch, "version", None), "cuda", None)
            if info["cudaAvailable"] and info["cudaDeviceCount"] > 0:
                names = []
                for i in range(info["cudaDeviceCount"]):
                    try:
                        names.append(str(torch.cuda.get_device_name(i)))
                    except Exception:
                        names.append(f"cuda:{i}")
                info["deviceNames"] = names
        except Exception:
            pass

    # Secondary signal: nvidia-smi output (if present)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        info["nvidiaSmi"] = (smi.stdout or "").strip()[:2000]
    except Exception:
        info["nvidiaSmi"] = None

    return info


def ensure_applio():
    if not APPLIO_DIR.exists():
        raise RuntimeError("Applio directory not found in image (expected /content/Applio).")
    core = APPLIO_DIR / "core.py"
    if not core.exists():
        raise RuntimeError(f"Applio core.py not found: {core}")


def download_file_http(url: str, dest: Path, retries: int = 3, timeout_sec: int = 120):
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    last_err = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "ogvoice-runpod-runner/1.0"})
            with urlopen(req, timeout=timeout_sec) as r:
                if getattr(r, "status", 200) >= 400:
                    raise RuntimeError(f"HTTP {getattr(r, 'status', '???')} while downloading {url}")

                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            if not tmp.exists() or tmp.stat().st_size < 5_000_000:
                raise RuntimeError(f"Downloaded file too small: {tmp.stat().st_size if tmp.exists() else 0} bytes")

            if dest.exists():
                dest.unlink()
            tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(2 * (i + 1))

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}\n{last_err}") from last_err


def ensure_custom_pretrained():
    # Always ensure advanced pretrained exists; fail job if can't fetch.
    missing = []
    if not CUSTOM_PRETRAIN_G_PATH.exists() or CUSTOM_PRETRAIN_G_PATH.stat().st_size < 5_000_000:
        missing.append("G")
    if not CUSTOM_PRETRAIN_D_PATH.exists() or CUSTOM_PRETRAIN_D_PATH.stat().st_size < 5_000_000:
        missing.append("D")

    if not missing:
        return

    print(json.dumps({"event": "custom_pretrained_download_start", "missing": missing}))
    if "G" in missing:
        download_file_http(CUSTOM_PRETRAIN_G_URL, CUSTOM_PRETRAIN_G_PATH)
    if "D" in missing:
        download_file_http(CUSTOM_PRETRAIN_D_URL, CUSTOM_PRETRAIN_D_PATH)
    print(
        json.dumps(
            {
                "event": "custom_pretrained_download_done",
                "gBytes": CUSTOM_PRETRAIN_G_PATH.stat().st_size if CUSTOM_PRETRAIN_G_PATH.exists() else None,
                "dBytes": CUSTOM_PRETRAIN_D_PATH.stat().st_size if CUSTOM_PRETRAIN_D_PATH.exists() else None,
            }
        )
    )

    if not CUSTOM_PRETRAIN_G_PATH.exists() or not CUSTOM_PRETRAIN_D_PATH.exists():
        raise RuntimeError("Custom pretrained download failed; required files are missing.")


def validate_forced_sample_rate():
    if FORCED_SR_TAG not in ("32k", "40k", "48k"):
        raise RuntimeError(f"Invalid FORCED_SR_TAG: {FORCED_SR_TAG}. Use 32k/40k/48k")
    if FORCED_SR not in (32000, 40000, 48000):
        raise RuntimeError(f"Invalid FORCED_SR: {FORCED_SR}. Use 32000/40000/48000")
    expected = int(FORCED_SR_TAG.rstrip("k")) * 1000
    if expected != FORCED_SR:
        raise RuntimeError(f"FORCED_SR_TAG and FORCED_SR mismatch: {FORCED_SR_TAG} vs {FORCED_SR}")


_HELP_CACHE = {}


def get_core_help(subcommand: str) -> str:
    cached = _HELP_CACHE.get(subcommand)
    if cached is not None:
        return cached
    try:
        out = run(["python", "core.py", subcommand, "--help"], cwd=str(APPLIO_DIR))
    except Exception:
        out = ""
    _HELP_CACHE[subcommand] = out or ""
    return _HELP_CACHE[subcommand]


def core_supports_flag(subcommand: str, flag: str) -> bool:
    return flag in get_core_help(subcommand)


def probe_audio(path: Path):
    try:
        out = run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
        )
        data = json.loads(out)
        duration = float(((data.get("format") or {}).get("duration")) or 0)
        if duration <= 0:
            raise RuntimeError("Audio duration invalid (<= 0).")
        return {"durationSec": duration}
    except FileNotFoundError:
        print("ffprobe not found; skipping audio probe.")
        return {"durationSec": None}
    except Exception:
        return {"durationSec": None}


def ffmpeg_convert_to_flac(src_wav: Path, dest_flac: Path):
    dest_flac.parent.mkdir(parents=True, exist_ok=True)
    if dest_flac.exists():
        dest_flac.unlink()
    # Lossless compression; keep original sample rate/channels.
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_wav),
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            str(dest_flac),
        ]
    )
    if not dest_flac.exists() or dest_flac.stat().st_size == 0:
        raise RuntimeError("FLAC conversion failed (empty output)")


def export_inference_zip(model_name: str, out_zip: Path):
    logs_dir = APPLIO_DIR / "logs" / model_name
    if not logs_dir.exists():
        raise RuntimeError(f"Model logs not found: {logs_dir}")

    weights = sorted(
        logs_dir.glob(f"{model_name}_*e_*s.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if weights:
        weight_path = weights[0]
    else:
        final = logs_dir / f"{model_name}.pth"
        if final.exists():
            weight_path = final
        else:
            raise RuntimeError("No weight file found after training (.pth missing).")

    index_files = sorted(
        logs_dir.glob("*.index"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    index_path = index_files[0] if index_files else None

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(weight_path, arcname=weight_path.name)
        if index_path:
            zf.write(index_path, arcname=index_path.name)


def clear_model_logs(model_name: str):
    logs_dir = APPLIO_DIR / "logs" / model_name
    if logs_dir.exists():
        shutil.rmtree(logs_dir, ignore_errors=True)


def _client_error_code(exc: ClientError):
    try:
        return str((exc.response or {}).get("Error", {}).get("Code", ""))
    except Exception:
        return ""


def restore_checkpoint_archive_if_exists(client, bucket: str, checkpoint_key: str, local_zip: Path, model_name: str):
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    if local_zip.exists():
        local_zip.unlink()

    try:
        print(json.dumps({"event": "checkpoint_restore_start", "key": checkpoint_key}))
        client.download_file(bucket, checkpoint_key, str(local_zip))
    except ClientError as e:
        code = _client_error_code(e)
        if code in ("404", "NoSuchKey", "NotFound"):
            print(json.dumps({"event": "checkpoint_restore_miss", "key": checkpoint_key}))
            return {"used": False, "reason": "missing", "files": []}
        raise

    if (not local_zip.exists()) or local_zip.stat().st_size == 0:
        print(json.dumps({"event": "checkpoint_restore_empty", "key": checkpoint_key}))
        return {"used": False, "reason": "empty", "files": []}

    logs_dir = APPLIO_DIR / "logs" / model_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    with zipfile.ZipFile(local_zip, "r") as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if not name:
                continue
            if not (name.startswith("G_") or name.startswith("D_")):
                continue
            if not name.endswith(".pth"):
                continue

            dest = logs_dir / name
            with zf.open(member, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(name)

    used = len(extracted) > 0
    print(
        json.dumps(
            {
                "event": "checkpoint_restore_done",
                "key": checkpoint_key,
                "used": used,
                "files": extracted,
            }
        )
    )
    return {"used": used, "reason": "ok" if used else "no_checkpoint_files", "files": extracted}


def create_checkpoint_archive(model_name: str, out_zip: Path):
    logs_dir = APPLIO_DIR / "logs" / model_name
    if not logs_dir.exists():
        return None

    g_ckpts = sorted(logs_dir.glob("G_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    d_ckpts = sorted(logs_dir.glob("D_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not g_ckpts or not d_ckpts:
        return None

    g = g_ckpts[0]
    d = d_ckpts[0]

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    meta = {
        "modelName": model_name,
        "createdAt": int(time.time()),
        "generator": g.name,
        "discriminator": d.name,
    }

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(g, arcname=g.name)
        zf.write(d, arcname=d.name)
        zf.writestr("checkpoint_meta.json", json.dumps(meta))

    return {
        "zip": out_zip,
        "g": g.name,
        "d": d.name,
        "bytes": out_zip.stat().st_size,
    }


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def extract_model_artifact(zip_path: Path, dest_dir: Path):
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise RuntimeError("Model artifact zip is missing or empty.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    pth_files = []
    index_files = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if not name:
                continue
            lower = name.lower()
            if not (lower.endswith(".pth") or lower.endswith(".index")):
                continue

            target = dest_dir / name
            with zf.open(member, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if lower.endswith(".pth"):
                pth_files.append(target)
            elif lower.endswith(".index"):
                index_files.append(target)

    if not pth_files:
        raise RuntimeError("Model zip does not include a .pth file.")
    if not index_files:
        raise RuntimeError("Model zip does not include a .index file.")

    pth = sorted(pth_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    idx = sorted(index_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return {"pth": pth, "index": idx}


def separate_vocals_and_instrumental(
    *,
    input_audio: Path,
    out_dir: Path,
    model_filename: str,
    vocals_name: str,
    instrumental_name: str,
):
    try:
        from audio_separator.separator import Separator
    except Exception as e:
        raise RuntimeError(
            "audio-separator import failed. Ensure this runner image includes "
            "audio-separator and onnxruntime-gpu (or onnxruntime). "
            f"Root cause: {type(e).__name__}: {e}"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    model_cache = WORK_DIR / "audio_separator_models"
    model_cache.mkdir(parents=True, exist_ok=True)

    separator_kwargs = {
        "log_level": logging.WARNING,
        "model_file_dir": str(model_cache),
        "output_dir": str(out_dir),
        "output_format": "WAV",
        "sample_rate": 44100,
        "use_autocast": True,
    }

    ctor_params = inspect.signature(Separator.__init__).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in ctor_params.values()
    )
    if accepts_var_kwargs:
        effective_kwargs = separator_kwargs
    else:
        effective_kwargs = {k: v for k, v in separator_kwargs.items() if k in ctor_params}

    separator = Separator(**effective_kwargs)
    separator.load_model(model_filename=model_filename)

    produced = []
    try:
        produced = separator.separate(
            str(input_audio),
            {
                "Vocals": vocals_name,
                "Instrumental": instrumental_name,
            },
        )
    except TypeError:
        produced = separator.separate(str(input_audio))

    vocals_path = _resolve_stem_path(
        out_dir=out_dir,
        produced=produced,
        preferred_tokens=[vocals_name, "vocals"],
    )
    instrumental_path = _resolve_stem_path(
        out_dir=out_dir,
        produced=produced,
        preferred_tokens=[instrumental_name, "instrumental", "no_vocals"],
    )

    if vocals_path is None:
        raise RuntimeError("Stem separation failed: vocals stem is missing.")
    if instrumental_path is None:
        raise RuntimeError("Stem separation failed: instrumental stem is missing.")

    return {
        "vocals": vocals_path,
        "instrumental": instrumental_path,
    }


def _resolve_stem_path(*, out_dir: Path, produced, preferred_tokens):
    candidates = []
    for item in produced or []:
        try:
            p = Path(str(item))
            if p.exists() and p.is_file():
                candidates.append(p)
        except Exception:
            continue

    if not candidates:
        candidates = sorted(
            out_dir.glob("**/*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    lowered = [t.lower() for t in preferred_tokens if t]
    for token in lowered:
        for p in candidates:
            if token in p.name.lower():
                return p

    return candidates[0] if candidates else None


def run_rvc_infer_file(
    *,
    input_audio: Path,
    output_wav: Path,
    model_files,
    pitch: int,
    index_rate: float,
    protect: float,
    f0_method: str,
    split_audio: bool,
    export_format: str,
    embedder_model: str,
):
    infer_cmd = [
        "python",
        "core.py",
        "infer",
        "--pitch",
        str(pitch),
        "--index_rate",
        str(index_rate),
        "--volume_envelope",
        "1",
        "--protect",
        str(protect),
        "--f0_method",
        f0_method,
        "--input_path",
        str(input_audio),
        "--output_path",
        str(output_wav),
        "--pth_path",
        str(model_files["pth"]),
        "--index_path",
        str(model_files["index"]),
        "--split_audio",
        str(split_audio),
        "--f0_autotune",
        "False",
        "--proposed_pitch",
        "False",
        "--clean_audio",
        "False",
        "--formant_shifting",
        "False",
        "--post_process",
        "False",
        "--export_format",
        export_format,
        "--embedder_model",
        embedder_model,
        "--embedder_model_custom",
        "",
        "--sid",
        "0",
    ]

    run(infer_cmd, cwd=str(APPLIO_DIR))

    if export_format == "WAV":
        result_path = output_wav
    else:
        result_path = Path(str(output_wav).replace(".wav", f".{export_format.lower()}"))
    if not result_path.exists() or result_path.stat().st_size == 0:
        raise RuntimeError("RVC inference finished but output audio is missing.")
    return result_path


def mix_cover_tracks(*, lead_path: Path, inst_path: Path, backing_path: Path | None, out_path: Path, back_volume: float):
    if backing_path is None:
        filter_graph = "[0:a]volume=1.15[lead];[1:a]volume=0.95[inst];[lead][inst]amix=inputs=2:weights='1 1':normalize=0:dropout_transition=0[mix]"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(lead_path),
            "-i",
            str(inst_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[mix]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out_path),
        ]
    else:
        filter_graph = f"[0:a]volume=1.15[lead];[1:a]volume=0.95[inst];[2:a]volume={back_volume}[back];[inst][back]amix=inputs=2:weights='1 1':normalize=0:dropout_transition=0[bed];[lead][bed]amix=inputs=2:weights='1 1':normalize=0:dropout_transition=0[mix]"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(lead_path),
            "-i",
            str(inst_path),
            "-i",
            str(backing_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[mix]",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out_path),
        ]

    run(cmd)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Final mix output is missing.")
    return out_path


def normalize_audio_to_wav(*, src_path: Path, out_path: Path):
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_path),
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out_path),
        ]
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Could not normalize audio to wav: {src_path}")
    return out_path


def stem_key_from_out_key(out_key: str, stem_name: str):
    base = out_key.rsplit(".", 1)[0] if "." in out_key else out_key
    return f"{base}__{stem_name}.wav"


def handle_infer_job(job, inp, bucket: str, client):
    if "modelKey" not in inp:
        raise RuntimeError("Missing required input: modelKey")
    if "inputKey" not in inp:
        raise RuntimeError("Missing required input: inputKey")
    if "outKey" not in inp:
        raise RuntimeError("Missing required input: outKey")

    model_key = inp["modelKey"]
    input_key = inp["inputKey"]
    out_key = inp["outKey"]

    pitch = as_int(inp.get("pitch"), 0)
    if pitch < -24:
        pitch = -24
    if pitch > 24:
        pitch = 24

    index_rate = _clamp(as_float(inp.get("searchFeatureRatio"), 0.75), 0.0, 1.0)
    split_audio = as_bool(inp.get("splitAudio"), True)

    protect = _clamp(as_float(inp.get("protect"), 0.33), 0.0, 0.5)
    f0_method = str(inp.get("f0Method") or "rmvpe")
    embedder_model = str(inp.get("embedderModel") or "contentvec")
    export_format = str(inp.get("exportFormat") or "WAV").upper()
    if export_format not in ("WAV", "MP3", "FLAC", "OGG", "M4A"):
        export_format = "WAV"

    add_back_vocals = as_bool(inp.get("addBackVocals"), False)
    convert_back_vocals = as_bool(inp.get("convertBackVocals"), False)
    mix_with_input = as_bool(inp.get("mixWithInput"), True)

    req_id = (job or {}).get("id", "infer")
    work = WORK_DIR / f"infer_{str(req_id)[:12]}"
    work.mkdir(parents=True, exist_ok=True)

    input_suffix = Path(str(input_key)).suffix or ".wav"
    input_path = work / f"input_audio{input_suffix}"
    model_zip = work / "model.zip"
    model_dir = work / "model"
    output_path = work / "converted.wav"

    gpu = get_gpu_diagnostics()
    print(json.dumps({"event": "gpu_diagnostics", **gpu}))
    if not gpu.get("cudaAvailable"):
        raise RuntimeError(
            "CUDA is not available in this worker. "
            "Voice conversion is configured to run on GPU; please attach a GPU worker."
        )

    print(
        json.dumps(
            {
                "event": "infer_start",
                "modelKey": model_key,
                "inputKey": input_key,
                "outKey": out_key,
                "pitch": pitch,
                "searchFeatureRatio": index_rate,
                "splitAudio": split_audio,
                "f0Method": f0_method,
                "embedderModel": embedder_model,
                "exportFormat": export_format,
                "addBackVocals": add_back_vocals,
                "convertBackVocals": convert_back_vocals,
                "mixWithInput": mix_with_input,
            }
        )
    )

    try:
        print(json.dumps({"event": "infer_download_model_start", "key": model_key}))
        client.download_file(bucket, model_key, str(model_zip))
        print(json.dumps({"event": "infer_download_model_done", "bytes": model_zip.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download model zip: s3://{bucket}/{model_key}\n{e}") from e

    try:
        print(json.dumps({"event": "infer_download_input_start", "key": input_key}))
        client.download_file(bucket, input_key, str(input_path))
        if not input_path.exists() or input_path.stat().st_size == 0:
            raise RuntimeError("Input audio file is missing or empty after download.")
        print(json.dumps({"event": "infer_download_input_done", "bytes": input_path.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download input audio: s3://{bucket}/{input_key}\n{e}") from e

    model_files = extract_model_artifact(model_zip, model_dir)
    stems_root = work / "stems"
    print(json.dumps({"event": "stem_separation_main_start", "model": VOCAL_SEP_MODEL}))
    main_stems = separate_vocals_and_instrumental(
        input_audio=input_path,
        out_dir=stems_root / "main",
        model_filename=VOCAL_SEP_MODEL,
        vocals_name="main_vocals",
        instrumental_name="instrumental",
    )
    lead_source = main_stems["vocals"]
    instrumental_source = main_stems["instrumental"]
    print(
        json.dumps(
            {
                "event": "stem_separation_main_done",
                "leadSource": str(lead_source),
                "instrumentalSource": str(instrumental_source),
            }
        )
    )

    backing_source = None
    if add_back_vocals:
        print(json.dumps({"event": "stem_separation_backing_start", "model": KARAOKE_SEP_MODEL}))
        backing_stems = separate_vocals_and_instrumental(
            input_audio=lead_source,
            out_dir=stems_root / "backing",
            model_filename=KARAOKE_SEP_MODEL,
            vocals_name="lead_main",
            instrumental_name="backing_vocals",
        )
        lead_source = backing_stems["vocals"]
        backing_source = backing_stems["instrumental"]
        print(
            json.dumps(
                {
                    "event": "stem_separation_backing_done",
                    "leadSource": str(lead_source),
                    "backingSource": str(backing_source),
                }
            )
        )

    print(json.dumps({"event": "infer_convert_main_start", "input": str(lead_source)}))
    lead_converted = run_rvc_infer_file(
        input_audio=lead_source,
        output_wav=output_path,
        model_files=model_files,
        pitch=pitch,
        index_rate=index_rate,
        protect=protect,
        f0_method=f0_method,
        split_audio=split_audio,
        export_format=export_format,
        embedder_model=embedder_model,
    )
    print(json.dumps({"event": "infer_convert_main_done", "output": str(lead_converted)}))

    backing_for_mix = None
    if add_back_vocals:
        if backing_source is None:
            raise RuntimeError("Backing vocal stem is missing.")
        if convert_back_vocals:
            print(json.dumps({"event": "infer_convert_backing_start", "input": str(backing_source)}))
            backing_output = work / "backing_converted.wav"
            backing_for_mix = run_rvc_infer_file(
                input_audio=backing_source,
                output_wav=backing_output,
                model_files=model_files,
                pitch=pitch,
                index_rate=index_rate,
                protect=protect,
                f0_method=f0_method,
                split_audio=split_audio,
                export_format=export_format,
                embedder_model=embedder_model,
            )
            print(json.dumps({"event": "infer_convert_backing_done", "output": str(backing_for_mix)}))
        else:
            backing_for_mix = backing_source

    final_path = lead_converted
    if mix_with_input:
        back_volume = 0.55 if not convert_back_vocals else 0.42
        if not add_back_vocals:
            back_volume = 0.0

        mix_path = work / f"cover_mix.{export_format.lower()}"
        final_path = mix_cover_tracks(
            lead_path=lead_converted,
            inst_path=instrumental_source,
            backing_path=backing_for_mix if add_back_vocals else None,
            out_path=mix_path,
            back_volume=back_volume,
        )
        print(
            json.dumps(
                {
                    "event": "infer_mix_done",
                    "output": str(final_path),
                    "includeBackVocals": add_back_vocals,
                    "convertBackVocals": convert_back_vocals,
                }
            )
        )

    stem_keys = {
        "mainVocalsKey": stem_key_from_out_key(out_key, "main_vocals"),
        "backVocalsKey": stem_key_from_out_key(out_key, "back_vocals") if add_back_vocals else None,
        "instrumentalKey": stem_key_from_out_key(out_key, "instrumental"),
    }

    try:
        main_preview = normalize_audio_to_wav(
            src_path=lead_converted,
            out_path=work / "preview_main_vocals.wav",
        )
        client.upload_file(str(main_preview), bucket, stem_keys["mainVocalsKey"])

        instrumental_preview = normalize_audio_to_wav(
            src_path=instrumental_source,
            out_path=work / "preview_instrumental.wav",
        )
        client.upload_file(str(instrumental_preview), bucket, stem_keys["instrumentalKey"])

        if add_back_vocals and backing_for_mix is not None and stem_keys["backVocalsKey"]:
            backing_preview = normalize_audio_to_wav(
                src_path=backing_for_mix,
                out_path=work / "preview_back_vocals.wav",
            )
            client.upload_file(str(backing_preview), bucket, stem_keys["backVocalsKey"])
    except Exception as e:
        print(json.dumps({"event": "infer_stem_preview_upload_failed", "error": str(e)}))

    try:
        print(json.dumps({"event": "infer_upload_start", "key": out_key, "bytes": final_path.stat().st_size}))
        client.upload_file(str(final_path), bucket, out_key)
        print(json.dumps({"event": "infer_upload_done"}))
    except ClientError as e:
        raise RuntimeError(f"Failed to upload conversion output: s3://{bucket}/{out_key}\n{e}") from e

    return {
        "ok": True,
        "mode": "infer",
        "outputKey": out_key,
        "outputBytes": final_path.stat().st_size,
        "pitch": pitch,
        "searchFeatureRatio": index_rate,
        "splitAudio": split_audio,
        "exportFormat": export_format,
        "addBackVocals": add_back_vocals,
        "convertBackVocals": convert_back_vocals,
        "mixedWithInput": mix_with_input,
        "stemKeys": stem_keys,
    }


def handler(job):
    ensure_applio()
    validate_forced_sample_rate()

    bucket = require_env("R2_BUCKET")
    inp = (job or {}).get("input") or {}
    mode = str(inp.get("mode") or "train").strip().lower()
    client = s3()

    if mode == "infer":
        return handle_infer_job(job=job, inp=inp, bucket=bucket, client=client)

    if "datasetKey" not in inp:
        raise RuntimeError("Missing required input: datasetKey")
    if "outKey" not in inp:
        raise RuntimeError("Missing required input: outKey")

    dataset_key = inp["datasetKey"]
    out_key = inp["outKey"]
    dataset_archive_key = inp.get("datasetArchiveKey")
    checkpoint_key = inp.get("checkpointKey")
    if not isinstance(checkpoint_key, str) or not checkpoint_key:
        checkpoint_key = None
    resume_from_checkpoint = as_bool(inp.get("resumeFromCheckpoint"), True)

    model_name = inp.get("modelName")

    # Force sample rate (advanced pretrains are trained for this SR)
    requested_sr = inp.get("sampleRate")
    sr_tag = FORCED_SR_TAG
    sr = FORCED_SR
    if requested_sr and str(requested_sr).strip().lower() not in (sr_tag, str(sr)):
        print(json.dumps({"event": "sample_rate_forced", "requested": requested_sr, "using": sr_tag}))

    # Initialize requested baseline settings (force to reduce variance)
    cut_preprocess = FORCE_CUT_PREPROCESS
    chunk_len = as_float(inp.get("chunkLen"), 3.0)
    overlap_len = as_float(inp.get("overlapLen"), 0.3)
    normalization_mode = FORCE_NORMALIZATION_MODE

    f0_method = FORCE_F0_METHOD
    include_mutes = FORCE_INCLUDE_MUTES
    embedder_model = FORCE_EMBEDDER_MODEL
    index_algorithm = FORCE_INDEX_ALGORITHM

    # Test mode: default 1 epoch unless explicitly overridden.
    total_epoch = as_int(inp.get("totalEpoch"), 1)
    if total_epoch < 1:
        total_epoch = 1

    batch_size = FORCE_BATCH_SIZE

    save_every_epoch = as_int(inp.get("saveEveryEpoch"), 1)
    if save_every_epoch < 1:
        save_every_epoch = 1
    if save_every_epoch > 100:
        save_every_epoch = 100
    save_only_latest = as_bool(inp.get("saveOnlyLatest"), True)

    vocoder = FORCE_VOCODER

    # Always use custom advanced pretrained (ignore Applio defaults)
    pretrained = True
    custom_pretrained = True

    if not model_name:
        req_id = (job or {}).get("id", "job")
        model_name = f"ogvoice_{str(req_id)[:12]}"

    print(
        json.dumps(
            {
                "event": "job_start",
                "modelName": model_name,
                "datasetKey": dataset_key,
                "outKey": out_key,
                "sampleRate": sr_tag,
                "sr": sr,
                "totalEpoch": total_epoch,
                "batchSize": batch_size,
                "saveEveryEpoch": save_every_epoch,
                "vocoder": vocoder,
                "cutPreprocess": cut_preprocess,
                "processEffects": FORCE_PROCESS_EFFECTS,
                "normalizationMode": normalization_mode,
                "f0Method": f0_method,
                "embedderModel": embedder_model,
                "includeMutes": include_mutes,
                "indexAlgorithm": index_algorithm,
                "pretrained": pretrained,
                "customPretrained": custom_pretrained,
                "saveOnlyLatest": save_only_latest,
                "checkpointKey": checkpoint_key,
                "resumeFromCheckpoint": resume_from_checkpoint,
            }
        )
    )

    gpu = get_gpu_diagnostics()
    print(json.dumps({"event": "gpu_diagnostics", **gpu}))
    if not gpu.get("cudaAvailable"):
        raise RuntimeError(
            "CUDA is not available in this worker. "
            "Training is configured to run on GPU; please attach a GPU worker."
        )
    if not gpu.get("deviceNames"):
        raise RuntimeError("No CUDA devices detected.")
    # Human-friendly confirmation line.
    print(f"[{gpu['deviceNames'][0]}] is active for current training.")

    # Ensure advanced pretrained exists (download if missing)
    ensure_custom_pretrained()
    print(
        json.dumps(
            {
                "event": "custom_pretrained_selected",
                "g": str(CUSTOM_PRETRAIN_G_PATH),
                "d": str(CUSTOM_PRETRAIN_D_PATH),
            }
        )
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = WORK_DIR / model_name
    dataset_dir = job_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.wav"
    dataset_flac_path = dataset_dir / "dataset.flac"
    checkpoint_zip = job_dir / "checkpoint_resume.zip"

    resume_info = {"used": False, "reason": "disabled", "files": []}
    if checkpoint_key and resume_from_checkpoint:
        try:
            resume_info = restore_checkpoint_archive_if_exists(
                client=client,
                bucket=bucket,
                checkpoint_key=checkpoint_key,
                local_zip=checkpoint_zip,
                model_name=model_name,
            )
        except Exception as e:
            print(json.dumps({"event": "checkpoint_restore_failed", "key": checkpoint_key, "error": str(e)[:500]}))
            resume_info = {"used": False, "reason": "restore_error", "files": []}

    if not resume_info.get("used"):
        clear_model_logs(model_name)

    # Download dataset from R2
    try:
        print(json.dumps({"event": "download_start", "bucket": bucket, "key": dataset_key}))
        # Support either .wav or .flac keys.
        if str(dataset_key).lower().endswith(".flac"):
            client.download_file(bucket, dataset_key, str(dataset_flac_path))
            if not dataset_flac_path.exists() or dataset_flac_path.stat().st_size == 0:
                raise RuntimeError("Downloaded dataset.flac is missing or empty.")
            # Decode to WAV for training
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(dataset_flac_path),
                    str(dataset_path),
                ]
            )
            if not dataset_path.exists() or dataset_path.stat().st_size == 0:
                raise RuntimeError("Failed to decode FLAC to WAV.")
            print(
                json.dumps(
                    {
                        "event": "download_done",
                        "bytes": dataset_flac_path.stat().st_size,
                        "decodedWavBytes": dataset_path.stat().st_size,
                    }
                )
            )
        else:
            client.download_file(bucket, dataset_key, str(dataset_path))
            if not dataset_path.exists() or dataset_path.stat().st_size == 0:
                raise RuntimeError("Downloaded dataset.wav is missing or empty.")
            print(json.dumps({"event": "download_done", "bytes": dataset_path.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download from R2: s3://{bucket}/{dataset_key}\n{e}") from e

    audio_info = probe_audio(dataset_path)
    print(json.dumps({"event": "audio_probe", **audio_info}))

    # Storage optimization: archive dataset as FLAC (lossless, smaller) for long-term storage.
    if isinstance(dataset_archive_key, str) and dataset_archive_key:
        try:
            print(json.dumps({"event": "dataset_archive_start", "key": dataset_archive_key}))
            ffmpeg_convert_to_flac(dataset_path, dataset_flac_path)
            client.upload_file(str(dataset_flac_path), bucket, dataset_archive_key)
            print(
                json.dumps(
                    {
                        "event": "dataset_archive_done",
                        "archiveBytes": dataset_flac_path.stat().st_size,
                    }
                )
            )
        except Exception as e:
            # Don't block training if archiving fails; we can retry on next run.
            print(json.dumps({"event": "dataset_archive_failed", "error": str(e)[:500]}))

    if PREREQ_MARKER.exists():
        print(json.dumps({"event": "applio_prerequisites_skip", "reason": "baked_into_image"}))
    else:
        # Fallback: older images or custom builds.
        print(json.dumps({"event": "applio_prerequisites_start"}))
        run(
            [
                "python",
                "core.py",
                "prerequisites",
                "--models",
                "True",
                "--pretraineds_hifigan",
                "True",
                "--exe",
                "False",
            ],
            cwd=str(APPLIO_DIR),
        )
        PREREQ_MARKER.write_text("ok\n")
        print(json.dumps({"event": "applio_prerequisites_done"}))

    # Applio's CLI limits cpu_cores to a fixed range (commonly max 64).
    # Some RunPod machines report higher counts (e.g. 128), which would crash.
    cpu_cores_raw = os.cpu_count() or 2
    cpu_cores = max(1, min(int(cpu_cores_raw), 64))
    if cpu_cores != cpu_cores_raw:
        print(json.dumps({"event": "cpu_cores_clamped", "raw": cpu_cores_raw, "using": cpu_cores}))

    print(json.dumps({"event": "preprocess_start"}))
    preprocess_cmd = [
        "python",
        "core.py",
        "preprocess",
        "--model_name",
        model_name,
        "--dataset_path",
        str(dataset_dir),
        "--sample_rate",
        str(sr),
        "--cpu_cores",
        str(cpu_cores),
        "--cut_preprocess",
        str(cut_preprocess),
        "--chunk_len",
        str(chunk_len),
        "--overlap_len",
        str(overlap_len),
        "--normalization_mode",
        str(normalization_mode),
    ]

    # Preprocess flags are version-dependent; only pass if supported by this Applio build.
    preprocess_flags = []

    # Applio UI: "Noise filter" == process_effects
    if core_supports_flag("preprocess", "--process_effects"):
        preprocess_cmd += ["--process_effects", str(FORCE_PROCESS_EFFECTS)]
        preprocess_flags.append("process_effects")
    else:
        print(json.dumps({"event": "preprocess_warn", "missingFlag": "--process_effects"}))

    if core_supports_flag("preprocess", "--noise_reduction"):
        preprocess_cmd += ["--noise_reduction", str(FORCE_NOISE_REDUCTION)]
        preprocess_flags.append("noise_reduction")
    if core_supports_flag("preprocess", "--noise_reduction_strength"):
        # Keep a reasonable default even though noise reduction is off.
        preprocess_cmd += ["--noise_reduction_strength", "0.7"]
        preprocess_flags.append("noise_reduction_strength")

    print(json.dumps({"event": "preprocess_flags", "flags": preprocess_flags}))
    run(preprocess_cmd, cwd=str(APPLIO_DIR))
    print(json.dumps({"event": "preprocess_done"}))

    print(json.dumps({"event": "extract_start"}))
    run(
        [
            "python",
            "core.py",
            "extract",
            "--model_name",
            model_name,
            "--f0_method",
            f0_method,
            "--sample_rate",
            str(sr),
            "--cpu_cores",
            str(cpu_cores),
            "--gpu",
            "0",
            "--embedder_model",
            embedder_model,
            "--embedder_model_custom",
            "",
            "--include_mutes",
            str(include_mutes),
        ],
        cwd=str(APPLIO_DIR),
    )
    print(json.dumps({"event": "extract_done"}))

    print(json.dumps({"event": "index_start"}))
    run(
        [
            "python",
            "core.py",
            "index",
            "--model_name",
            model_name,
            "--index_algorithm",
            index_algorithm,
        ],
        cwd=str(APPLIO_DIR),
    )
    print(json.dumps({"event": "index_done"}))

    print(json.dumps({"event": "train_start"}))
    # Keep train flags minimal; unspecified settings follow Applio defaults.
    train_cmd = [
        "python",
        "core.py",
        "train",
        "--model_name",
        model_name,
        "--save_every_epoch",
        str(save_every_epoch),
        "--save_only_latest",
        str(save_only_latest),
        "--total_epoch",
        str(total_epoch),
        "--sample_rate",
        str(sr),
        "--batch_size",
        str(batch_size),
        "--gpu",
        "0",
        "--pretrained",
        str(pretrained),
        "--custom_pretrained",
        str(custom_pretrained),
        "--g_pretrained_path",
        str(CUSTOM_PRETRAIN_G_PATH),
        "--d_pretrained_path",
        str(CUSTOM_PRETRAIN_D_PATH),
        "--vocoder",
        vocoder,
    ]
    train_error = None
    try:
        run(train_cmd, cwd=str(APPLIO_DIR))
        print(json.dumps({"event": "train_done"}))
    except Exception as e:
        train_error = e
        print(json.dumps({"event": "train_failed", "error": str(e)[:1000]}))

    checkpoint_upload = None
    if checkpoint_key and train_error is not None:
        try:
            checkpoint_bundle = create_checkpoint_archive(model_name, checkpoint_zip)
            if checkpoint_bundle:
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_upload_start",
                            "key": checkpoint_key,
                            "bytes": checkpoint_bundle["bytes"],
                            "g": checkpoint_bundle["g"],
                            "d": checkpoint_bundle["d"],
                        }
                    )
                )
                client.upload_file(str(checkpoint_bundle["zip"]), bucket, checkpoint_key)
                checkpoint_upload = {
                    "key": checkpoint_key,
                    "bytes": checkpoint_bundle["bytes"],
                    "g": checkpoint_bundle["g"],
                    "d": checkpoint_bundle["d"],
                }
                print(json.dumps({"event": "checkpoint_upload_done", **checkpoint_upload}))
            else:
                print(json.dumps({"event": "checkpoint_upload_skip", "reason": "no_checkpoint_files"}))
        except Exception as e:
            print(json.dumps({"event": "checkpoint_upload_failed", "key": checkpoint_key, "error": str(e)[:500]}))
    elif checkpoint_key:
        print(json.dumps({"event": "checkpoint_upload_skip", "reason": "success_not_needed"}))

    if train_error is not None:
        raise train_error

    out_zip = job_dir / "model.zip"
    print(json.dumps({"event": "export_start", "zip": str(out_zip)}))
    export_inference_zip(model_name, out_zip)
    print(json.dumps({"event": "export_done", "bytes": out_zip.stat().st_size}))

    try:
        print(json.dumps({"event": "upload_start", "bucket": bucket, "key": out_key}))
        client.upload_file(str(out_zip), bucket, out_key)
        print(json.dumps({"event": "upload_done"}))
    except ClientError as e:
        raise RuntimeError(f"Failed to upload to R2: s3://{bucket}/{out_key}\n{e}") from e

    return {
        "ok": True,
        "modelName": model_name,
        "artifactKey": out_key,
        "sampleRate": sr_tag,
        "sr": sr,
        "audio": audio_info,
        "pretrained": {
            "g": str(CUSTOM_PRETRAIN_G_PATH),
            "d": str(CUSTOM_PRETRAIN_D_PATH),
        },
        "resume": {
            "used": bool(resume_info.get("used")),
            "reason": resume_info.get("reason"),
            "files": resume_info.get("files"),
        },
        "checkpoint": checkpoint_upload,
        "datasetArchiveKey": dataset_archive_key,
        "datasetArchiveBytes": dataset_flac_path.stat().st_size if dataset_flac_path.exists() else None,
    }


def log_runtime_dependency_info() -> None:
    info = {}

    try:
        import audio_separator  # type: ignore

        info["audio_separator"] = getattr(audio_separator, "__version__", "unknown")
    except Exception as e:
        info["audio_separator_error"] = f"{type(e).__name__}: {e}"

    try:
        import onnxruntime as ort  # type: ignore

        info["onnxruntime"] = getattr(ort, "__version__", "unknown")
    except Exception as e:
        info["onnxruntime_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps({"event": "runtime_dependency_info", **info}))


if __name__ == "__main__":
    log_runtime_dependency_info()
    runpod.serverless.start({"handler": handler})
