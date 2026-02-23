import os
import json
import hashlib
import shutil
import subprocess
import zipfile
import time
import re
import logging
from collections import deque
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
# Downloaded on demand and cached per worker at /content/Applio/pretrained_custom/*.pth
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
FORCE_TRAINING_PRECISION = os.environ.get("APPLIO_TRAIN_PRECISION", "bf16").strip().lower()
FORCE_INFER_PRECISION = os.environ.get("APPLIO_INFER_PRECISION", "fp16").strip().lower()

EPOCH_PATTERNS = [
    re.compile(r"(?i)\bepoch\s*[:\[\(]?\s*(\d+)\s*(?:/|of)\s*(\d+)"),
    re.compile(r"(?i)\[(\d+)\s*/\s*(\d+)\]"),
    re.compile(r"(?i)\bepoch\s*=\s*(\d+)\b"),
]
TRAINING_SPEED_PATTERN = re.compile(r"(?i)training_speed\s*=\s*(\d{1,2}:\d{2}:\d{2})")

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
    "Mel-Roformer Becruily Deux",
)
KARAOKE_SEP_MODEL = os.environ.get(
    "KARAOKE_SEP_MODEL",
    "Mel-Roformer Karaoke by aufr33 and viperx",
)


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except Exception:
        return default
    return max(minimum, value)


INFER_STEM_TIMEOUT_SEC = env_int("INFER_STEM_TIMEOUT_SEC", 1800)
INFER_RVC_TIMEOUT_SEC = env_int("INFER_RVC_TIMEOUT_SEC", 1500)
INFER_MIX_TIMEOUT_SEC = env_int("INFER_MIX_TIMEOUT_SEC", 600)
INFER_SPLIT_AUDIO_FORCE_AT_SECONDS = env_int("INFER_SPLIT_AUDIO_FORCE_AT_SECONDS", 210, minimum=0)
INFER_MODEL_CACHE_DIR = WORK_DIR / "model_cache"
INFER_MODEL_CACHE_MAX_ENTRIES = env_int("INFER_MODEL_CACHE_MAX_ENTRIES", 24)
_STEM_CACHE_PREFIX_RAW = os.environ.get("INFER_STEM_CACHE_PREFIX", "cache/stems/v1")
INFER_STEM_CACHE_PREFIX = _STEM_CACHE_PREFIX_RAW.strip().strip("/") or "cache/stems/v1"

MUSIC_SEPARATION_DIR = Path("/app/music_separation_code")
MUSIC_SEPARATION_INFER = MUSIC_SEPARATION_DIR / "inference.py"
MUSIC_SEPARATION_MODELS_DIR = Path(
    os.environ.get("MUSIC_SEPARATION_MODELS_DIR", "/app/music_separation_models")
)
AUDIO_SEPARATOR_MODEL_DIR = Path(
    os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", str(WORK_DIR / "audio_separator_models"))
)

DEFAULT_DEREVERB_MODEL_NAME = "UVR-Deecho-Dereverb"
DEFAULT_DEECHO_MODEL_NAME = "UVR-Deecho-Normal"

DEFAULT_DEREVERB_MODEL_FILE = "UVR-DeEcho-DeReverb.pth"
DEFAULT_DEECHO_MODEL_FILE = "UVR-De-Echo-Normal.pth"

STEM_MODEL_SPECS = {
    "vocals": {
        "id": "mel_vocals_becruily_deux",
        "name": "Mel-Roformer Becruily Deux",
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/config_deux_becruily.yaml?download=true",
        "model_url": "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/becruily_deux.ckpt?download=true",
        "aliases": (
            "mel-roformer becruily deux",
            "becruily_deux.ckpt",
            "config_deux_becruily.yaml",
        ),
    },
    "karaoke": {
        "id": "mel_karaoke_aufr33_viperx",
        "name": "Mel-Roformer Karaoke by aufr33 and viperx",
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
        "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "aliases": (
            "mel-roformer karaoke by aufr33 and viperx",
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "config_mel_band_roformer_karaoke.yaml",
        ),
    },
}


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


INFER_STEM_OUTPUT_FLAC = as_bool(os.environ.get("INFER_STEM_OUTPUT_FLAC"), False)
INFER_STEM_CACHE_ENABLED = as_bool(os.environ.get("INFER_STEM_CACHE_ENABLED"), False)
INFER_STEM_CACHE_UPLOAD_ENABLED = as_bool(os.environ.get("INFER_STEM_CACHE_UPLOAD_ENABLED"), True)
AUDIO_SEPARATOR_VR_BATCH_SIZE = env_int("AUDIO_SEPARATOR_VR_BATCH_SIZE", 1)
AUDIO_SEPARATOR_ENABLE_TTA = as_bool(os.environ.get("AUDIO_SEPARATOR_ENABLE_TTA"), False)


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


def normalized_model_name(v) -> str:
    return str(v or "").strip().lower()


def model_alias_key(v) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalized_model_name(v))


UVR_DEREVERB_MODEL_ALIAS = {
    model_alias_key(DEFAULT_DEREVERB_MODEL_NAME): (DEFAULT_DEREVERB_MODEL_NAME, DEFAULT_DEREVERB_MODEL_FILE),
    model_alias_key(DEFAULT_DEREVERB_MODEL_FILE): (DEFAULT_DEREVERB_MODEL_NAME, DEFAULT_DEREVERB_MODEL_FILE),
    model_alias_key("UVR-DeEcho-DeReverb"): (DEFAULT_DEREVERB_MODEL_NAME, DEFAULT_DEREVERB_MODEL_FILE),
}

UVR_DEECHO_MODEL_ALIAS = {
    model_alias_key(DEFAULT_DEECHO_MODEL_NAME): (DEFAULT_DEECHO_MODEL_NAME, DEFAULT_DEECHO_MODEL_FILE),
    model_alias_key(DEFAULT_DEECHO_MODEL_FILE): (DEFAULT_DEECHO_MODEL_NAME, DEFAULT_DEECHO_MODEL_FILE),
    model_alias_key("UVR-De-Echo-Normal"): (DEFAULT_DEECHO_MODEL_NAME, DEFAULT_DEECHO_MODEL_FILE),
    model_alias_key("UVR-Deecho-Agggressive"): ("UVR-Deecho-Agggressive", "UVR-De-Echo-Aggressive.pth"),
    model_alias_key("UVR-De-Echo-Aggressive"): ("UVR-Deecho-Agggressive", "UVR-De-Echo-Aggressive.pth"),
}


def resolve_uvr_model(stage: str, requested: str):
    if stage == "dereverb":
        alias = UVR_DEREVERB_MODEL_ALIAS
        fallback = (DEFAULT_DEREVERB_MODEL_NAME, DEFAULT_DEREVERB_MODEL_FILE)
    elif stage == "deecho":
        alias = UVR_DEECHO_MODEL_ALIAS
        fallback = (DEFAULT_DEECHO_MODEL_NAME, DEFAULT_DEECHO_MODEL_FILE)
    else:
        raise RuntimeError(f"Unknown UVR stage: {stage}")

    key = model_alias_key(requested)
    resolved = alias.get(key)
    if resolved is not None:
        return {
            "name": resolved[0],
            "file": resolved[1],
        }

    if str(requested or "").strip():
        print(
            json.dumps(
                {
                    "event": "uvr_model_forced",
                    "stage": stage,
                    "requested": requested,
                    "using": fallback[0],
                }
            )
        )

    return {
        "name": fallback[0],
        "file": fallback[1],
    }


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


def run(cmd, cwd=None, timeout_sec=None):
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout if e.stdout is not None else e.output
        tail = str(out or "")[-8000:]
        raise RuntimeError(
            f"Command timed out after {timeout_sec}s: {' '.join(cmd)}\n\n{tail}"
        ) from e

    if p.returncode != 0:
        tail = (p.stdout or "")[-8000:]
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{tail}")
    return p.stdout or ""


def run_stream(cmd, cwd=None, on_line=None):
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tail = deque(maxlen=1200)
    if process.stdout is not None:
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.rstrip("\n")
            if line:
                print(line)
                tail.append(line)
                if on_line is not None:
                    try:
                        on_line(line)
                    except Exception:
                        pass
        process.stdout.close()

    process.wait()
    if process.returncode != 0:
        tail_text = "\n".join(tail)[-8000:]
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(cmd)}\n\n{tail_text}")


def extract_epoch_progress(line, total_epoch):
    if not line:
        return None
    text = str(line)

    for pattern in EPOCH_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        try:
            current = int(match.group(1))
        except Exception:
            continue

        total = None
        if match.lastindex and match.lastindex >= 2:
            try:
                total = int(match.group(2))
            except Exception:
                total = None

        if total_epoch > 0 and total is not None and total != total_epoch:
            # Prefer exact total-epoch matches for this job when total is present.
            continue
        if current < 1:
            continue
        if total_epoch > 0 and current > total_epoch:
            continue
        return current

    return None


def extract_training_speed_seconds(line):
    if not line:
        return None
    match = TRAINING_SPEED_PATTERN.search(str(line))
    if not match:
        return None
    token = match.group(1)
    parts = token.split(":")
    if len(parts) != 3:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2])
    except Exception:
        return None
    if h < 0 or m < 0 or m > 59 or s < 0 or s > 59:
        return None
    return h * 3600 + m * 60 + s


def upload_training_progress(client, bucket, progress_key, payload):
    if not progress_key:
        return

    try:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        client.put_object(
            Bucket=bucket,
            Key=progress_key,
            Body=body,
            ContentType="application/json",
            CacheControl="no-store",
        )
    except Exception as e:
        print(
            json.dumps(
                {
                    "event": "training_progress_upload_failed",
                    "key": progress_key,
                    "error": str(e)[:300],
                }
            )
        )


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


def normalize_precision(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in ("fp32", "fp16", "bf16"):
        return v
    return "bf16"


def resolve_precision(preferred: str) -> tuple[str, str, str | None]:
    requested = normalize_precision(preferred)
    if requested == "bf16":
        bf16_supported = (
            _TORCH_AVAILABLE
            and torch is not None
            and bool(torch.cuda.is_available())
            and bool(torch.cuda.is_bf16_supported())
        )
        if bf16_supported:
            return requested, "bf16", None
        fp16_supported = _TORCH_AVAILABLE and torch is not None and bool(torch.cuda.is_available())
        return requested, "fp16" if fp16_supported else "fp32", "bf16_not_supported"

    if requested == "fp16":
        fp16_supported = _TORCH_AVAILABLE and torch is not None and bool(torch.cuda.is_available())
        if fp16_supported:
            return requested, "fp16", None
        return requested, "fp32", "fp16_not_supported"

    return requested, "fp32", None


def force_applio_precision(preferred: str) -> str:
    config_path = APPLIO_DIR / "assets" / "config.json"
    requested, effective, fallback_reason = resolve_precision(preferred)

    cfg = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    cfg["precision"] = effective
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "precision_forced",
                "requested": requested,
                "effective": effective,
                "configPath": str(config_path),
                "fallbackReason": fallback_reason,
            }
        )
    )

    return effective


def ensure_applio():
    if not APPLIO_DIR.exists():
        raise RuntimeError("Applio directory not found in image (expected /content/Applio).")
    core = APPLIO_DIR / "core.py"
    if not core.exists():
        raise RuntimeError(f"Applio core.py not found: {core}")


def download_file_http(
    url: str,
    dest: Path,
    retries: int = 3,
    timeout_sec: int = 120,
    min_bytes: int = 5_000_000,
):
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

            if not tmp.exists() or tmp.stat().st_size < min_bytes:
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


def restore_feature_cache_if_exists(client, bucket: str, feature_cache_key: str, local_zip: Path, model_name: str):
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    if local_zip.exists():
        local_zip.unlink()

    try:
        print(json.dumps({"event": "feature_cache_restore_start", "key": feature_cache_key}))
        client.download_file(bucket, feature_cache_key, str(local_zip))
    except ClientError as e:
        code = _client_error_code(e)
        if code in ("404", "NoSuchKey", "NotFound"):
            print(json.dumps({"event": "feature_cache_restore_miss", "key": feature_cache_key}))
            return {"used": False, "reason": "missing", "files": 0}
        raise

    if (not local_zip.exists()) or local_zip.stat().st_size == 0:
        print(json.dumps({"event": "feature_cache_restore_empty", "key": feature_cache_key}))
        return {"used": False, "reason": "empty", "files": 0}

    logs_dir = APPLIO_DIR / "logs" / model_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(local_zip, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel = Path(member.filename)
            parts = [p for p in rel.parts if p not in ("", ".")]
            if not parts or any(p == ".." for p in parts):
                continue

            dest = logs_dir.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    has_index = any(logs_dir.glob("*.index"))
    used = extracted > 0 and has_index
    print(
        json.dumps(
            {
                "event": "feature_cache_restore_done",
                "key": feature_cache_key,
                "used": used,
                "files": extracted,
                "hasIndex": has_index,
            }
        )
    )
    if not used:
        return {"used": False, "reason": "invalid", "files": extracted}
    return {"used": True, "reason": "ok", "files": extracted}


def create_feature_cache_archive(model_name: str, out_zip: Path):
    logs_dir = APPLIO_DIR / "logs" / model_name
    if not logs_dir.exists():
        return None

    files = []
    for p in logs_dir.rglob("*"):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if lower.endswith(".pth"):
            continue
        files.append(p)

    if not files:
        return None

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            rel = p.relative_to(logs_dir).as_posix()
            zf.write(p, arcname=rel)

    return {
        "zip": out_zip,
        "bytes": out_zip.stat().st_size,
        "files": len(files),
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


def _file_ready(path: Path, min_bytes: int) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except Exception:
        return False


def _model_cache_slot(model_key: str) -> Path:
    digest = hashlib.sha1(str(model_key).encode("utf-8")).hexdigest()[:24]
    return INFER_MODEL_CACHE_DIR / digest


def prune_model_cache(max_entries: int):
    if max_entries < 1:
        return
    try:
        if not INFER_MODEL_CACHE_DIR.exists():
            return
        slots = [p for p in INFER_MODEL_CACHE_DIR.iterdir() if p.is_dir()]
        if len(slots) <= max_entries:
            return
        slots_sorted = sorted(slots, key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in slots_sorted[max_entries:]:
            shutil.rmtree(stale, ignore_errors=True)
            print(json.dumps({"event": "infer_model_cache_pruned", "slot": stale.name}))
    except Exception as e:
        print(json.dumps({"event": "infer_model_cache_prune_failed", "error": str(e)[:300]}))


def resolve_infer_model_files(*, client, bucket: str, model_key: str, work: Path):
    slot = _model_cache_slot(model_key)
    cached_pth = slot / "model.pth"
    cached_index = slot / "model.index"

    if _file_ready(cached_pth, 5_000_000) and _file_ready(cached_index, 1_000):
        try:
            os.utime(slot, None)
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "event": "infer_model_cache_hit",
                    "slot": slot.name,
                    "pthBytes": cached_pth.stat().st_size,
                    "indexBytes": cached_index.stat().st_size,
                }
            )
        )
        return {
            "pth": cached_pth,
            "index": cached_index,
            "cache": "hit",
            "cacheSlot": slot.name,
        }

    started = time.time()
    print(json.dumps({"event": "infer_model_cache_miss", "slot": slot.name, "key": model_key}))

    model_zip = work / "model.zip"
    model_extract_dir = work / "model_extract"
    if model_zip.exists():
        model_zip.unlink()
    if model_extract_dir.exists():
        shutil.rmtree(model_extract_dir, ignore_errors=True)

    try:
        print(json.dumps({"event": "infer_download_model_start", "key": model_key}))
        client.download_file(bucket, model_key, str(model_zip))
        print(json.dumps({"event": "infer_download_model_done", "bytes": model_zip.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download model zip: s3://{bucket}/{model_key}\n{e}") from e

    extracted = extract_model_artifact(model_zip, model_extract_dir)
    slot.mkdir(parents=True, exist_ok=True)

    tmp_pth = slot / "model.pth.tmp"
    tmp_index = slot / "model.index.tmp"
    for p in (tmp_pth, tmp_index):
        if p.exists():
            p.unlink()

    shutil.copy2(extracted["pth"], tmp_pth)
    shutil.copy2(extracted["index"], tmp_index)

    tmp_pth.replace(cached_pth)
    tmp_index.replace(cached_index)

    meta = {
        "modelKey": model_key,
        "cachedAt": int(time.time()),
        "sourcePth": extracted["pth"].name,
        "sourceIndex": extracted["index"].name,
        "pthBytes": cached_pth.stat().st_size,
        "indexBytes": cached_index.stat().st_size,
    }
    (slot / "meta.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "infer_model_cache_store_done",
                "slot": slot.name,
                "durationSec": round(max(0.0, time.time() - started), 3),
                "pthBytes": cached_pth.stat().st_size,
                "indexBytes": cached_index.stat().st_size,
            }
        )
    )
    prune_model_cache(INFER_MODEL_CACHE_MAX_ENTRIES)

    return {
        "pth": cached_pth,
        "index": cached_index,
        "cache": "miss",
        "cacheSlot": slot.name,
    }


def sha1_file_hex(path: Path, chunk_bytes: int = 2 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def user_scope_from_input_key(input_key: str) -> str:
    text = str(input_key or "")
    m = re.match(r"^u/([^/]+)/", text)
    if not m:
        return "global"
    scope = re.sub(r"[^a-zA-Z0-9_-]", "-", m.group(1).strip())
    return scope or "global"


def build_main_stem_cache_prefix(input_sha1: str, model_id: str, user_scope: str) -> str:
    model_slug = re.sub(r"[^a-z0-9_-]", "-", str(model_id or "").strip().lower())
    if not model_slug:
        model_slug = "default"
    key_hash = str(input_sha1 or "").strip().lower()
    if not key_hash:
        key_hash = "unknown"
    scope = re.sub(r"[^a-zA-Z0-9_-]", "-", str(user_scope or "").strip()) or "global"
    return f"{INFER_STEM_CACHE_PREFIX}/main/{scope}/{model_slug}/{key_hash}"


def _s3_download_if_exists(client, bucket: str, key: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()

    try:
        client.download_file(bucket, key, str(dest))
    except ClientError as e:
        code = _client_error_code(e)
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

    return _file_ready(dest, 1000)


def _s3_json_if_exists(client, bucket: str, key: str):
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = _client_error_code(e)
        if code in ("404", "NoSuchKey", "NotFound"):
            return None
        raise

    try:
        body = obj["Body"].read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def restore_main_stem_cache_if_exists(client, bucket: str, cache_prefix: str, dest_dir: Path):
    meta_key = f"{cache_prefix}/meta.json"
    print(json.dumps({"event": "stem_cache_main_lookup_start", "prefix": cache_prefix}))

    meta = _s3_json_if_exists(client, bucket, meta_key)
    name_pairs = []
    if isinstance(meta, dict):
        vocals_name = str(meta.get("vocalsName") or "").strip()
        inst_name = str(meta.get("instrumentalName") or "").strip()
        if vocals_name and inst_name:
            name_pairs.append((vocals_name, inst_name))

    name_pairs.extend(
        [
            ("vocals.wav", "instrumental.wav"),
            ("vocals.flac", "instrumental.flac"),
        ]
    )

    seen = set()
    dedup_pairs = []
    for vocals_name, inst_name in name_pairs:
        pair = (vocals_name, inst_name)
        if pair in seen:
            continue
        seen.add(pair)
        dedup_pairs.append(pair)

    dest_dir.mkdir(parents=True, exist_ok=True)

    for vocals_name, inst_name in dedup_pairs:
        if "/" in vocals_name or "\\" in vocals_name:
            continue
        if "/" in inst_name or "\\" in inst_name:
            continue

        vocals_key = f"{cache_prefix}/{vocals_name}"
        inst_key = f"{cache_prefix}/{inst_name}"
        vocals_path = dest_dir / vocals_name
        inst_path = dest_dir / inst_name

        ok_vocals = _s3_download_if_exists(client, bucket, vocals_key, vocals_path)
        ok_inst = _s3_download_if_exists(client, bucket, inst_key, inst_path)
        if ok_vocals and ok_inst and _file_ready(vocals_path, 1000) and _file_ready(inst_path, 1000):
            print(
                json.dumps(
                    {
                        "event": "stem_cache_main_lookup_done",
                        "prefix": cache_prefix,
                        "used": True,
                        "vocalsKey": vocals_key,
                        "instrumentalKey": inst_key,
                        "vocalsBytes": vocals_path.stat().st_size,
                        "instrumentalBytes": inst_path.stat().st_size,
                    }
                )
            )
            return {
                "used": True,
                "vocals": vocals_path,
                "instrumental": inst_path,
                "vocalsKey": vocals_key,
                "instrumentalKey": inst_key,
            }

        for p in (vocals_path, inst_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    print(json.dumps({"event": "stem_cache_main_lookup_done", "prefix": cache_prefix, "used": False}))
    return {"used": False, "vocals": None, "instrumental": None, "vocalsKey": None, "instrumentalKey": None}


def upload_main_stem_cache_best_effort(
    client,
    bucket: str,
    cache_prefix: str,
    vocals_path: Path,
    instrumental_path: Path,
    meta: dict,
):
    if not INFER_STEM_CACHE_UPLOAD_ENABLED:
        print(json.dumps({"event": "stem_cache_main_upload_skip", "reason": "disabled", "prefix": cache_prefix}))
        return {"uploaded": False, "reason": "disabled"}

    try:
        def safe_suffix(path: Path) -> str:
            suffix = str(path.suffix or "").strip().lower()
            if re.fullmatch(r"\.[a-z0-9]+", suffix):
                return suffix
            return ".wav"

        started = time.time()
        vocals_name = f"vocals{safe_suffix(vocals_path)}"
        instrumental_name = f"instrumental{safe_suffix(instrumental_path)}"
        vocals_key = f"{cache_prefix}/{vocals_name}"
        inst_key = f"{cache_prefix}/{instrumental_name}"

        print(json.dumps({"event": "stem_cache_main_upload_start", "prefix": cache_prefix}))
        client.upload_file(str(vocals_path), bucket, vocals_key)
        client.upload_file(str(instrumental_path), bucket, inst_key)

        payload = {
            "version": 1,
            "createdAt": int(time.time()),
            "vocalsName": vocals_name,
            "instrumentalName": instrumental_name,
            "vocalsKey": vocals_key,
            "instrumentalKey": inst_key,
            "vocalsBytes": vocals_path.stat().st_size,
            "instrumentalBytes": instrumental_path.stat().st_size,
            "meta": meta,
        }
        client.put_object(
            Bucket=bucket,
            Key=f"{cache_prefix}/meta.json",
            Body=json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store",
        )

        print(
            json.dumps(
                {
                    "event": "stem_cache_main_upload_done",
                    "prefix": cache_prefix,
                    "durationSec": round(max(0.0, time.time() - started), 3),
                }
            )
        )
        return {"uploaded": True, "vocalsKey": vocals_key, "instrumentalKey": inst_key}
    except Exception as e:
        print(json.dumps({"event": "stem_cache_main_upload_failed", "prefix": cache_prefix, "error": str(e)[:500]}))
        return {"uploaded": False, "reason": "error"}


def separate_vocals_and_instrumental(
    *,
    input_audio: Path,
    out_dir: Path,
    model_filename: str,
    vocals_name: str,
    instrumental_name: str,
):
    if not MUSIC_SEPARATION_INFER.exists():
        raise RuntimeError(
            "music_separation_code is missing in this runner image. "
            "Please rebuild with updated Dockerfile."
        )

    role = "karaoke" if instrumental_name == "backing_vocals" else "vocals"
    model_spec = STEM_MODEL_SPECS[role]
    requested_model = str(model_filename or "")
    requested_model_norm = normalized_model_name(requested_model)
    accepted_aliases = set(model_spec["aliases"])
    accepted_aliases.add(normalized_model_name(model_spec["name"]))

    if requested_model_norm and requested_model_norm not in accepted_aliases:
        print(
            json.dumps(
                {
                    "event": "stem_model_forced",
                    "role": role,
                    "requested": requested_model,
                    "using": model_spec["name"],
                }
            )
        )

    stem_model_dir = MUSIC_SEPARATION_MODELS_DIR / model_spec["id"]
    config_path = stem_model_dir / "config.yaml"
    checkpoint_path = stem_model_dir / "model.ckpt"

    if (not config_path.exists()) or config_path.stat().st_size < 100:
        print(json.dumps({"event": "stem_model_config_download_start", "role": role, "url": model_spec["config_url"]}))
        download_file_http(model_spec["config_url"], config_path, min_bytes=100)
        print(json.dumps({"event": "stem_model_config_download_done", "role": role, "bytes": config_path.stat().st_size}))

    if (not checkpoint_path.exists()) or checkpoint_path.stat().st_size < 5_000_000:
        print(json.dumps({"event": "stem_model_checkpoint_download_start", "role": role, "url": model_spec["model_url"]}))
        download_file_http(model_spec["model_url"], checkpoint_path, min_bytes=5_000_000)
        print(json.dumps({"event": "stem_model_checkpoint_download_done", "role": role, "bytes": checkpoint_path.stat().st_size}))

    if model_spec["type"] == "bs_roformer":
        ensure_bs_roformer_runtime_compat()
        sanitize_bs_roformer_config(config_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    sep_cmd = [
        "python",
        str(MUSIC_SEPARATION_INFER),
        "--model_type",
        model_spec["type"],
        "--config_path",
        str(config_path),
        "--start_check_point",
        str(checkpoint_path),
        "--input_file",
        str(input_audio),
        "--store_dir",
        str(out_dir),
        "--pcm_type",
        "PCM_16",
        "--extract_instrumental",
        "--disable_detailed_pbar",
    ]
    if INFER_STEM_OUTPUT_FLAC:
        sep_cmd.append("--flac_file")

    print(
        json.dumps(
            {
                "event": "stem_separation_command_start",
                "role": role,
                "timeoutSec": INFER_STEM_TIMEOUT_SEC,
                "input": str(input_audio),
                "outputFlac": INFER_STEM_OUTPUT_FLAC,
            }
        )
    )
    run(sep_cmd, timeout_sec=INFER_STEM_TIMEOUT_SEC)
    print(json.dumps({"event": "stem_separation_command_done", "role": role}))

    produced = sorted(
        [p for p in out_dir.glob("**/*") if p.exists() and p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

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
        "modelUsed": model_spec["name"],
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
            [p for p in out_dir.glob("**/*") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    lowered = [t.lower() for t in preferred_tokens if t]
    for token in lowered:
        for p in candidates:
            if token in p.name.lower():
                return p

    return candidates[0] if candidates else None


def sanitize_bs_roformer_config(config_path: Path):
    try:
        if not config_path.exists() or config_path.stat().st_size < 10:
            return

        raw = config_path.read_text(encoding="utf-8")
        lines = raw.splitlines(keepends=True)

        removable_keys = {
            "mlp_expansion_factor",
            "use_torch_checkpoint",
            "skip_connection",
        }

        removed = []
        out = []
        in_model = False
        model_indent = 0

        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))

            if re.match(r"^model\s*:\s*$", stripped):
                in_model = True
                model_indent = indent
                out.append(line)
                continue

            if in_model and stripped and indent <= model_indent:
                in_model = False

            if in_model:
                m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:", line)
                if m:
                    key = m.group(1)
                    if key in removable_keys:
                        removed.append(key)
                        continue

            out.append(line)

        if not removed:
            return

        config_path.write_text("".join(out), encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "stem_model_config_sanitized",
                    "path": str(config_path),
                    "removedKeys": sorted(set(removed)),
                }
            )
        )
    except Exception as e:
        print(
            json.dumps(
                {
                    "event": "stem_model_config_sanitize_failed",
                    "path": str(config_path),
                    "error": str(e)[:500],
                }
            )
        )


def ensure_bs_roformer_runtime_compat():
    bs_file = MUSIC_SEPARATION_DIR / "models" / "bs_roformer" / "bs_roformer.py"
    if not bs_file.exists():
        print(
            json.dumps(
                {
                    "event": "stem_bs_roformer_patch_skip",
                    "reason": "missing_file",
                    "path": str(bs_file),
                }
            )
        )
        return

    try:
        text = bs_file.read_text(encoding="utf-8")
        changed = False

        # Repair broken partial patch state that causes:
        # NameError: name 'mlp_expansion_factor' is not defined
        # Safe in all variants because MaskEstimator already has default=4.
        bad_line = "                mlp_expansion_factor=mlp_expansion_factor,\n"
        if bad_line in text:
            text = text.replace(bad_line, "", 1)
            changed = True

        # Also remove accidental tab-indented variant if present.
        bad_line_tab = "\t\t\t\tmlp_expansion_factor=mlp_expansion_factor,\n"
        if bad_line_tab in text:
            text = text.replace(bad_line_tab, "", 1)
            changed = True

        if changed:
            bs_file.write_text(text, encoding="utf-8")
            print(json.dumps({"event": "stem_bs_roformer_patch_applied", "path": str(bs_file), "mode": "repair_nameerror"}))
        else:
            print(json.dumps({"event": "stem_bs_roformer_patch_ok", "path": str(bs_file)}))
    except Exception as e:
        print(
            json.dumps(
                {
                    "event": "stem_bs_roformer_patch_failed",
                    "path": str(bs_file),
                    "error": str(e)[:500],
                }
            )
        )


_AUDIO_SEPARATOR_CLASS = None


def get_audio_separator_class():
    global _AUDIO_SEPARATOR_CLASS
    if _AUDIO_SEPARATOR_CLASS is not None:
        return _AUDIO_SEPARATOR_CLASS

    try:
        from audio_separator.separator import Separator  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "audio-separator dependency is required for deecho/dereverb inference stages."
        ) from e

    _AUDIO_SEPARATOR_CLASS = Separator
    return _AUDIO_SEPARATOR_CLASS


def run_uvr_single_stem(
    *,
    input_audio: Path,
    out_dir: Path,
    stage: str,
    model_file: str,
    output_single_stem: str,
):
    Separator = get_audio_separator_class()

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = AUDIO_SEPARATOR_MODEL_DIR / stage
    model_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=str(model_dir),
        log_level=logging.WARNING,
        normalization_threshold=1.0,
        output_format="wav",
        output_dir=str(out_dir),
        output_single_stem=output_single_stem,
        vr_params={
            "batch_size": AUDIO_SEPARATOR_VR_BATCH_SIZE,
            "enable_tta": AUDIO_SEPARATOR_ENABLE_TTA,
        },
    )

    separator.load_model(model_filename=model_file)
    produced = separator.separate(str(input_audio))
    if not isinstance(produced, (list, tuple)):
        produced = []

    preferred_tokens = [
        output_single_stem,
        output_single_stem.replace(" ", ""),
        stage,
    ]
    if stage == "dereverb":
        preferred_tokens.extend(["noreverb", "no_reverb"])
    if stage == "deecho":
        preferred_tokens.extend(["noecho", "no_echo"])

    stem_path = _resolve_stem_path(
        out_dir=out_dir,
        produced=produced,
        preferred_tokens=preferred_tokens,
    )
    if stem_path is None:
        raise RuntimeError(f"{stage} stage finished but output stem is missing.")
    return stem_path


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
    def build_infer_cmd(split_flag: bool):
        return [
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
            str(split_flag),
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

    def resolve_rvc_output(started_at: float):
        primary = output_wav if export_format == "WAV" else Path(str(output_wav).replace(".wav", f".{export_format.lower()}"))
        candidates = [primary, output_wav]
        for p in candidates:
            try:
                if p.exists() and p.is_file() and p.stat().st_size > 0:
                    return p
            except Exception:
                pass

        stem = output_wav.stem
        parent = output_wav.parent
        fuzzy = sorted(
            [
                p
                for p in parent.glob(f"{stem}*")
                if p.is_file()
                and p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
                and p.stat().st_size > 0
                and p.stat().st_mtime >= (started_at - 2.0)
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return fuzzy[0] if fuzzy else None

    first_start = time.time()
    first_out = run(build_infer_cmd(split_audio), cwd=str(APPLIO_DIR), timeout_sec=INFER_RVC_TIMEOUT_SEC)
    first_result = resolve_rvc_output(first_start)
    if first_result is not None:
        return first_result

    if split_audio:
        print(json.dumps({"event": "infer_split_audio_retry", "reason": "missing_output", "retrySplitAudio": False}))
        second_start = time.time()
        second_out = run(build_infer_cmd(False), cwd=str(APPLIO_DIR), timeout_sec=INFER_RVC_TIMEOUT_SEC)
        second_result = resolve_rvc_output(second_start)
        if second_result is not None:
            return second_result
        debug_tail = (first_out or "")[-2000:] + "\n--- retry ---\n" + (second_out or "")[-2000:]
        raise RuntimeError(f"RVC inference finished but output audio is missing.\n\n{debug_tail}")

    raise RuntimeError(f"RVC inference finished but output audio is missing.\n\n{(first_out or '')[-3000:]}")


def _build_atempo_chain(target: float):
    # ffmpeg atempo only supports [0.5, 2.0]; chain multiple filters when needed.
    if target <= 0:
        return [1.0]

    factors = []
    remain = float(target)
    while remain < 0.5:
        factors.append(0.5)
        remain /= 0.5
    while remain > 2.0:
        factors.append(2.0)
        remain /= 2.0

    if not factors or abs(remain - 1.0) > 1e-6:
        factors.append(remain)
    return factors


def pitch_shift_audio_preserve_duration(*, src_path: Path, out_path: Path, semitones: int):
    if semitones == 0:
        return src_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    pitch_factor = 2 ** (float(semitones) / 12.0)
    tempo_target = 1.0 / pitch_factor
    atempo_chain = ",".join([f"atempo={f:.8f}" for f in _build_atempo_chain(tempo_target)])
    filter_chain = f"asetrate=44100*{pitch_factor:.10f},{atempo_chain},aresample=44100"

    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_path),
            "-filter:a",
            filter_chain,
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out_path),
        ],
        timeout_sec=INFER_MIX_TIMEOUT_SEC,
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Instrumental pitch shift output is missing.")
    return out_path


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

    run(cmd, timeout_sec=INFER_MIX_TIMEOUT_SEC)
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


def handle_infer_job(job, inp, bucket: str, client, effective_precision: str):
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

    instrumental_pitch = as_int(inp.get("instrumentalPitch"), pitch)
    if instrumental_pitch < -24:
        instrumental_pitch = -24
    if instrumental_pitch > 24:
        instrumental_pitch = 24

    index_rate = _clamp(as_float(inp.get("searchFeatureRatio"), 0.75), 0.0, 1.0)
    split_audio = as_bool(inp.get("splitAudio"), False)
    protect = _clamp(as_float(inp.get("protect"), 0.33), 0.0, 0.5)
    f0_method = str(inp.get("f0Method") or "rmvpe")
    embedder_model = str(inp.get("embedderModel") or "contentvec")
    export_format = str(inp.get("exportFormat") or "WAV").upper()
    if export_format not in ("WAV", "MP3", "FLAC", "OGG", "M4A"):
        export_format = "WAV"

    add_back_vocals = as_bool(inp.get("addBackVocals"), False)
    convert_back_vocals = as_bool(inp.get("convertBackVocals"), False)
    mix_with_input = as_bool(inp.get("mixWithInput"), True)
    deecho_enabled = as_bool(inp.get("deechoEnabled"), True)

    dereverb_model = resolve_uvr_model("dereverb", str(inp.get("dereverbModel") or DEFAULT_DEREVERB_MODEL_NAME))
    deecho_model = resolve_uvr_model("deecho", str(inp.get("deechoModel") or DEFAULT_DEECHO_MODEL_NAME))

    vocal_sep_model = str(
        inp.get("vocalStemModel")
        or inp.get("vocalModel")
        or inp.get("vocalSepModel")
        or VOCAL_SEP_MODEL
    ).strip()
    karaoke_sep_model = str(
        inp.get("karaokeStemModel")
        or inp.get("karaokeModel")
        or inp.get("karaokeSepModel")
        or KARAOKE_SEP_MODEL
    ).strip()

    req_id = (job or {}).get("id", "infer")
    work = WORK_DIR / f"infer_{str(req_id)[:12]}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    infer_started_at = time.time()

    input_suffix = Path(str(input_key)).suffix or ".wav"
    input_path = work / f"input_audio{input_suffix}"
    model_zip = work / "model.zip"
    model_dir = work / "model"
    output_path = work / "converted.wav"

    gpu = get_gpu_diagnostics()
    print(json.dumps({"event": "gpu_diagnostics", **gpu}))
    if not gpu.get("cudaAvailable"):
        raise RuntimeError("CUDA is not available in this worker. GPU is required for inference.")

    print(
        json.dumps(
            {
                "event": "infer_start",
                "modelKey": model_key,
                "inputKey": input_key,
                "outKey": out_key,
                "pitch": pitch,
                "instrumentalPitch": instrumental_pitch,
                "searchFeatureRatio": index_rate,
                "splitAudio": split_audio,
                "f0Method": f0_method,
                "embedderModel": embedder_model,
                "exportFormat": export_format,
                "addBackVocals": add_back_vocals,
                "convertBackVocals": convert_back_vocals,
                "mixWithInput": mix_with_input,
                "deechoEnabled": deecho_enabled,
                "vocalSepModel": vocal_sep_model,
                "karaokeSepModel": karaoke_sep_model,
                "dereverbModel": dereverb_model["name"],
                "deechoModel": deecho_model["name"],
                "precision": effective_precision,
            }
        )
    )

    try:
        print(json.dumps({"event": "infer_download_model_start", "key": model_key}))
        client.download_file(bucket, model_key, str(model_zip))
        print(json.dumps({"event": "infer_download_model_done", "bytes": model_zip.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download model zip: s3://{bucket}/{model_key}\n{e}") from e

    model_files = extract_model_artifact(model_zip, model_dir)

    try:
        print(json.dumps({"event": "infer_download_input_start", "key": input_key}))
        client.download_file(bucket, input_key, str(input_path))
        if not input_path.exists() or input_path.stat().st_size == 0:
            raise RuntimeError("Input audio file is missing or empty after download.")
        print(json.dumps({"event": "infer_download_input_done", "bytes": input_path.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download input audio: s3://{bucket}/{input_key}\n{e}") from e

    stems_root = work / "stems"

    print(json.dumps({"event": "stem_separation_main_start", "model": vocal_sep_model}))
    main_sep_started = time.time()
    main_stems = separate_vocals_and_instrumental(
        input_audio=input_path,
        out_dir=stems_root / "main",
        model_filename=vocal_sep_model,
        vocals_name="main_vocals",
        instrumental_name="instrumental",
    )
    main_sep_duration_sec = max(0.0, time.time() - main_sep_started)
    lead_source = main_stems["vocals"]
    instrumental_source = main_stems["instrumental"]
    print(
        json.dumps(
            {
                "event": "stem_separation_main_done",
                "leadSource": str(lead_source),
                "instrumentalSource": str(instrumental_source),
                "modelUsed": main_stems.get("modelUsed"),
                "durationSec": round(main_sep_duration_sec, 3),
            }
        )
    )

    print(json.dumps({"event": "stem_separation_backing_start", "model": karaoke_sep_model}))
    backing_sep_started = time.time()
    backing_stems = separate_vocals_and_instrumental(
        input_audio=lead_source,
        out_dir=stems_root / "backing",
        model_filename=karaoke_sep_model,
        vocals_name="lead_main",
        instrumental_name="backing_vocals",
    )
    backing_sep_duration_sec = max(0.0, time.time() - backing_sep_started)
    lead_source = backing_stems["vocals"]
    backing_source = backing_stems["instrumental"]
    print(
        json.dumps(
            {
                "event": "stem_separation_backing_done",
                "leadSource": str(lead_source),
                "backingSource": str(backing_source),
                "modelUsed": backing_stems.get("modelUsed"),
                "durationSec": round(backing_sep_duration_sec, 3),
            }
        )
    )

    print(json.dumps({"event": "stem_clean_dereverb_start", "model": dereverb_model["name"], "input": str(lead_source)}))
    lead_source = run_uvr_single_stem(
        input_audio=lead_source,
        out_dir=stems_root / "dereverb",
        stage="dereverb",
        model_file=dereverb_model["file"],
        output_single_stem="No Reverb",
    )
    print(json.dumps({"event": "stem_clean_dereverb_done", "output": str(lead_source)}))

    if deecho_enabled:
        print(json.dumps({"event": "stem_clean_deecho_start", "model": deecho_model["name"], "input": str(lead_source)}))
        lead_source = run_uvr_single_stem(
            input_audio=lead_source,
            out_dir=stems_root / "deecho",
            stage="deecho",
            model_file=deecho_model["file"],
            output_single_stem="No Echo",
        )
        print(json.dumps({"event": "stem_clean_deecho_done", "output": str(lead_source)}))

    lead_rvc_input = normalize_audio_to_wav(src_path=lead_source, out_path=work / "lead_for_rvc.wav")
    print(json.dumps({"event": "infer_convert_main_start", "input": str(lead_rvc_input)}))
    lead_converted = run_rvc_infer_file(
        input_audio=lead_rvc_input,
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
        if convert_back_vocals:
            backing_rvc_input = normalize_audio_to_wav(src_path=backing_source, out_path=work / "backing_for_rvc.wav")
            print(json.dumps({"event": "infer_convert_backing_start", "input": str(backing_rvc_input)}))
            backing_for_mix = run_rvc_infer_file(
                input_audio=backing_rvc_input,
                output_wav=work / "backing_converted.wav",
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

    instrumental_for_mix = instrumental_source
    if mix_with_input and instrumental_pitch != 0:
        instrumental_for_mix = pitch_shift_audio_preserve_duration(
            src_path=instrumental_source,
            out_path=work / "instrumental_pitch_shifted.wav",
            semitones=instrumental_pitch,
        )

    final_path = lead_converted
    if mix_with_input:
        back_volume = 0.55 if not convert_back_vocals else 0.42
        if not add_back_vocals:
            back_volume = 0.0
        final_path = mix_cover_tracks(
            lead_path=lead_converted,
            inst_path=instrumental_for_mix,
            backing_path=backing_for_mix if add_back_vocals else None,
            out_path=work / f"cover_mix.{export_format.lower()}",
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

    try:
        print(json.dumps({"event": "infer_upload_start", "key": out_key, "bytes": final_path.stat().st_size}))
        client.upload_file(str(final_path), bucket, out_key)
        print(json.dumps({"event": "infer_upload_done"}))
    except ClientError as e:
        raise RuntimeError(f"Failed to upload conversion output: s3://{bucket}/{out_key}\n{e}") from e

    total_duration_sec = max(0.0, time.time() - infer_started_at)
    print(json.dumps({"event": "infer_total_done", "durationSec": round(total_duration_sec, 3)}))

    return {
        "ok": True,
        "mode": "infer",
        "outputKey": out_key,
        "outputBytes": final_path.stat().st_size,
        "pitch": pitch,
        "instrumentalPitch": instrumental_pitch,
        "splitAudio": split_audio,
        "exportFormat": export_format,
        "precision": effective_precision,
        "addBackVocals": add_back_vocals,
        "convertBackVocals": convert_back_vocals,
        "mixedWithInput": mix_with_input,
        "deechoEnabled": deecho_enabled,
    }


def handler(job):
    print(json.dumps({"event": "runner_build", "build": "stemflow-20260223-vanilla-infer-reset-v8"}))
    log_runtime_dependency_info()

    ensure_applio()
    validate_forced_sample_rate()

    bucket = require_env("R2_BUCKET")
    inp = (job or {}).get("input") or {}
    mode = str(inp.get("mode") or "train").strip().lower()
    preferred_precision = FORCE_INFER_PRECISION if mode == "infer" else FORCE_TRAINING_PRECISION
    effective_precision = force_applio_precision(preferred_precision)
    if effective_precision != preferred_precision:
        raise RuntimeError(
            f"Precision requirement mismatch for mode={mode}: "
            f"requested={preferred_precision}, effective={effective_precision}"
        )
    client = s3()

    if mode == "infer":
        return handle_infer_job(
            job=job,
            inp=inp,
            bucket=bucket,
            client=client,
            effective_precision=effective_precision,
        )

    if "datasetKey" not in inp:
        raise RuntimeError("Missing required input: datasetKey")
    if "outKey" not in inp:
        raise RuntimeError("Missing required input: outKey")

    dataset_key = inp["datasetKey"]
    out_key = inp["outKey"]
    progress_key = inp.get("progressKey")
    if not isinstance(progress_key, str) or not progress_key:
        progress_key = None
    dataset_archive_key = inp.get("datasetArchiveKey")
    checkpoint_key = inp.get("checkpointKey")
    if not isinstance(checkpoint_key, str) or not checkpoint_key:
        checkpoint_key = None
    feature_cache_key = inp.get("featureCacheKey")
    if not isinstance(feature_cache_key, str) or not feature_cache_key:
        feature_cache_key = None
    upload_feature_cache = as_bool(inp.get("uploadFeatureCache"), True)
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
                "progressKey": progress_key,
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
                "precision": effective_precision,
                "pretrained": pretrained,
                "customPretrained": custom_pretrained,
                "saveOnlyLatest": save_only_latest,
                "checkpointKey": checkpoint_key,
                "featureCacheKey": feature_cache_key,
                "uploadFeatureCache": upload_feature_cache,
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
    feature_cache_zip = job_dir / "feature_cache_restore.zip"

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

    feature_cache_info = {"used": False, "reason": "disabled", "files": 0}
    if feature_cache_key:
        try:
            feature_cache_info = restore_feature_cache_if_exists(
                client=client,
                bucket=bucket,
                feature_cache_key=feature_cache_key,
                local_zip=feature_cache_zip,
                model_name=model_name,
            )
        except Exception as e:
            print(json.dumps({"event": "feature_cache_restore_failed", "key": feature_cache_key, "error": str(e)[:500]}))
            feature_cache_info = {"used": False, "reason": "restore_error", "files": 0}

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

    # Defer dataset FLAC archiving until after training artifact is uploaded.
    # This keeps training startup path focused on clone-critical steps.

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

    if feature_cache_info.get("used"):
        print(
            json.dumps(
                {
                    "event": "feature_cache_used",
                    "key": feature_cache_key,
                    "reason": feature_cache_info.get("reason"),
                    "files": feature_cache_info.get("files"),
                }
            )
        )
    else:
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

        if feature_cache_key and upload_feature_cache:
            try:
                feature_cache_bundle = create_feature_cache_archive(model_name, feature_cache_zip)
                if feature_cache_bundle:
                    print(
                        json.dumps(
                            {
                                "event": "feature_cache_upload_start",
                                "key": feature_cache_key,
                                "bytes": feature_cache_bundle["bytes"],
                                "files": feature_cache_bundle["files"],
                            }
                        )
                    )
                    client.upload_file(str(feature_cache_bundle["zip"]), bucket, feature_cache_key)
                    print(
                        json.dumps(
                            {
                                "event": "feature_cache_upload_done",
                                "key": feature_cache_key,
                                "bytes": feature_cache_bundle["bytes"],
                                "files": feature_cache_bundle["files"],
                            }
                        )
                    )
                else:
                    print(json.dumps({"event": "feature_cache_upload_skip", "reason": "no_cache_files"}))
            except Exception as e:
                print(json.dumps({"event": "feature_cache_upload_failed", "key": feature_cache_key, "error": str(e)[:500]}))
        elif feature_cache_key:
            print(json.dumps({"event": "feature_cache_upload_skip", "reason": "disabled"}))

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

    train_started_at = time.time()
    last_epoch_seen = 0
    last_epoch_timestamp = None
    epoch_durations = []
    steady_epoch_seconds = None
    first_epoch_seconds = None

    upload_training_progress(
        client,
        bucket,
        progress_key,
        {
            "version": 1,
            "status": "running",
            "phase": "training",
            "totalEpoch": total_epoch,
            "currentEpoch": 0,
            "progressPercent": 0,
            "elapsedSeconds": 0,
            "etaSeconds": None,
            "epochSecondsEstimate": None,
            "firstEpochSeconds": None,
            "updatedAt": int(train_started_at),
        },
    )

    def on_train_line(line):
        nonlocal last_epoch_seen, last_epoch_timestamp, steady_epoch_seconds, first_epoch_seconds

        epoch = extract_epoch_progress(line, total_epoch)
        if epoch is None or epoch <= last_epoch_seen:
            return

        now_ts = time.time()

        epoch_seconds = extract_training_speed_seconds(line)
        if epoch_seconds is None and last_epoch_timestamp is not None:
            epoch_seconds = max(0.001, now_ts - last_epoch_timestamp)

        if epoch_seconds is not None:
            epoch_durations.append(float(epoch_seconds))
            if epoch == 1 and first_epoch_seconds is None:
                first_epoch_seconds = float(epoch_seconds)

        last_epoch_timestamp = now_ts
        last_epoch_seen = epoch

        stable_slice = epoch_durations[1:] if len(epoch_durations) >= 2 else []

        if len(stable_slice) > 0:
            steady_epoch_seconds = sum(stable_slice) / len(stable_slice)

        eta_seconds = None
        if steady_epoch_seconds is not None:
            remaining_epochs = max(0, total_epoch - epoch)
            eta_seconds = int(round(remaining_epochs * steady_epoch_seconds))

        progress_percent = int(max(1, min(99, round((epoch / max(1, total_epoch)) * 100))))

        upload_training_progress(
            client,
            bucket,
            progress_key,
            {
                "version": 1,
                "status": "running",
                "phase": "training",
                "totalEpoch": total_epoch,
                "currentEpoch": epoch,
                "progressPercent": progress_percent,
                "elapsedSeconds": int(max(0, now_ts - train_started_at)),
                "etaSeconds": eta_seconds,
                "epochSecondsEstimate": round(steady_epoch_seconds, 3) if steady_epoch_seconds is not None else None,
                "firstEpochSeconds": round(first_epoch_seconds, 3) if first_epoch_seconds is not None else None,
                "updatedAt": int(now_ts),
            },
        )

        print(
            json.dumps(
                {
                    "event": "training_epoch_progress",
                    "epoch": epoch,
                    "totalEpoch": total_epoch,
                    "etaSeconds": eta_seconds,
                    "epochSecondsEstimate": round(steady_epoch_seconds, 3) if steady_epoch_seconds is not None else None,
                }
            )
        )

    train_error = None
    try:
        run_stream(train_cmd, cwd=str(APPLIO_DIR), on_line=on_train_line)
        done_at = time.time()
        upload_training_progress(
            client,
            bucket,
            progress_key,
            {
                "version": 1,
                "status": "succeeded",
                "phase": "training",
                "totalEpoch": total_epoch,
                "currentEpoch": total_epoch,
                "progressPercent": 100,
                "elapsedSeconds": int(max(0, done_at - train_started_at)),
                "etaSeconds": 0,
                "epochSecondsEstimate": round(steady_epoch_seconds, 3) if steady_epoch_seconds is not None else None,
                "firstEpochSeconds": round(first_epoch_seconds, 3) if first_epoch_seconds is not None else None,
                "updatedAt": int(done_at),
            },
        )
        print(json.dumps({"event": "train_done"}))
    except Exception as e:
        train_error = e
        failed_at = time.time()
        upload_training_progress(
            client,
            bucket,
            progress_key,
            {
                "version": 1,
                "status": "failed",
                "phase": "training",
                "totalEpoch": total_epoch,
                "currentEpoch": last_epoch_seen,
                "progressPercent": int(max(1, min(99, round((last_epoch_seen / max(1, total_epoch)) * 100)))),
                "elapsedSeconds": int(max(0, failed_at - train_started_at)),
                "etaSeconds": None,
                "epochSecondsEstimate": round(steady_epoch_seconds, 3) if steady_epoch_seconds is not None else None,
                "firstEpochSeconds": round(first_epoch_seconds, 3) if first_epoch_seconds is not None else None,
                "updatedAt": int(failed_at),
                "error": str(e)[:240],
            },
        )
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

    dataset_archive_bytes = None
    if isinstance(dataset_archive_key, str) and dataset_archive_key:
        try:
            print(json.dumps({"event": "dataset_archive_start", "key": dataset_archive_key}))
            ffmpeg_convert_to_flac(dataset_path, dataset_flac_path)
            client.upload_file(str(dataset_flac_path), bucket, dataset_archive_key)
            dataset_archive_bytes = dataset_flac_path.stat().st_size
            print(
                json.dumps(
                    {
                        "event": "dataset_archive_done",
                        "archiveBytes": dataset_archive_bytes,
                    }
                )
            )
        except Exception as e:
            # Best-effort only. Training artifact already uploaded.
            print(json.dumps({"event": "dataset_archive_failed", "error": str(e)[:500]}))

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
        "precision": effective_precision,
        "resume": {
            "used": bool(resume_info.get("used")),
            "reason": resume_info.get("reason"),
            "files": resume_info.get("files"),
        },
        "featureCache": {
            "used": bool(feature_cache_info.get("used")),
            "reason": feature_cache_info.get("reason"),
            "files": feature_cache_info.get("files"),
            "key": feature_cache_key,
        },
        "checkpoint": checkpoint_upload,
        "datasetArchiveKey": dataset_archive_key,
        "datasetArchiveBytes": dataset_archive_bytes,
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
        try:
            info["onnxruntimeProviders"] = list(ort.get_available_providers())
        except Exception:
            info["onnxruntimeProviders"] = []
    except Exception as e:
        info["onnxruntime_error"] = f"{type(e).__name__}: {e}"

    info["music_separation_infer"] = str(MUSIC_SEPARATION_INFER)
    info["music_separation_ready"] = bool(MUSIC_SEPARATION_INFER.exists())

    print(json.dumps({"event": "runtime_dependency_info", **info}))


if __name__ == "__main__":
    log_runtime_dependency_info()
    runpod.serverless.start({"handler": handler})
