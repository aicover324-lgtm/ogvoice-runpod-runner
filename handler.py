import os
import json
import shutil
import subprocess
import zipfile
import time
import re
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

EPOCH_PATTERNS = [
    re.compile(r"(?i)\bepoch\s*[:\[\(]?\s*(\d+)\s*(?:/|of)\s*(\d+)"),
    re.compile(r"(?i)\[(\d+)\s*/\s*(\d+)\]"),
    re.compile(r"(?i)\bepoch\s*=\s*(\d+)\b"),
]
TRAINING_SPEED_PATTERN = re.compile(r"(?i)training_speed\s*=\s*(\d{1,2}:\d{2}:\d{2})")

# "Noise filter" in Applio UI maps to the preprocess flag `--process_effects`.
# This applies a simple filter during preprocessing.
FORCE_PROCESS_EFFECTS = True

# Noise settings:
# - noise reduction: off
FORCE_NOISE_REDUCTION = False


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


def export_model_zip(model_name: str, out_zip: Path):
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


def handler(job):
    print(json.dumps({"event": "runner_build", "build": "stemflow-20260223-training-only-nocache-v11"}))
    log_runtime_dependency_info()

    ensure_applio()
    validate_forced_sample_rate()

    bucket = require_env("R2_BUCKET")
    inp = (job or {}).get("input") or {}
    mode = str(inp.get("mode") or "train").strip().lower()
    if mode == "infer":
        raise RuntimeError("Inference mode has been removed from this runner by explicit request.")
    if mode != "train":
        raise RuntimeError(f"Unsupported mode: {mode}")

    preferred_precision = FORCE_TRAINING_PRECISION
    effective_precision = force_applio_precision(preferred_precision)
    if effective_precision != preferred_precision:
        raise RuntimeError(
            f"Precision requirement mismatch for mode={mode}: "
            f"requested={preferred_precision}, effective={effective_precision}"
        )
    client = s3()

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

    # Defer dataset FLAC archiving until after training artifact is uploaded.
    # This keeps training startup path focused on clone-critical steps.

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

    if train_error is not None:
        raise train_error

    out_zip = job_dir / "model.zip"
    print(json.dumps({"event": "export_start", "zip": str(out_zip)}))
    export_model_zip(model_name, out_zip)
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
        "datasetArchiveKey": dataset_archive_key,
        "datasetArchiveBytes": dataset_archive_bytes,
    }


def log_runtime_dependency_info() -> None:
    info = {}

    try:
        import onnxruntime as ort  # type: ignore

        info["onnxruntime"] = getattr(ort, "__version__", "unknown")
        try:
            info["onnxruntimeProviders"] = list(ort.get_available_providers())
        except Exception:
            info["onnxruntimeProviders"] = []
    except Exception as e:
        info["onnxruntime_error"] = f"{type(e).__name__}: {e}"

    print(json.dumps({"event": "runtime_dependency_info", **info}))


if __name__ == "__main__":
    log_runtime_dependency_info()
    runpod.serverless.start({"handler": handler})
