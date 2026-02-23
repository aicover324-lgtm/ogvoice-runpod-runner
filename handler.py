import os
import json
import shutil
import subprocess
import zipfile
import time
import re
import sys
import logging
from collections import deque
from contextlib import contextmanager
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

try:
    from audio_separator.separator import Separator  # type: ignore

    _AUDIO_SEPARATOR_AVAILABLE = True
except Exception:
    Separator = None  # type: ignore
    _AUDIO_SEPARATOR_AVAILABLE = False


APPLIO_DIR = Path("/content/Applio")
APPLIO_COVER_ROOT = Path("/app")
APPLIO_COVER_DIR = APPLIO_COVER_ROOT / "programs" / "applio_code"
WORK_DIR = Path("/workspace")
PREREQ_MARKER = APPLIO_DIR / ".prerequisites_ready"
MUSIC_SEPARATION_DIR = Path("/app/music_separation_code")
MUSIC_MODELS_DIR = Path("/app/music_separation_models")
RUNNER_BUILD = "stemflow-20260223-infer-phase2-v9"

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

# Inference defaults and model sources (phase-1 cover pipeline).
INFER_DEFAULT_EXPORT_FORMAT = "WAV"
INFER_DEFAULT_PITCH_EXTRACTOR = "rmvpe"
INFER_DEFAULT_EMBEDDER = "contentvec"
INFER_DEFAULT_FILTER_RADIUS = 3
INFER_DEFAULT_INDEX_RATE = 0.75
INFER_DEFAULT_RMS_MIX_RATE = 0.25
INFER_DEFAULT_PROTECT = 0.33
INFER_DEFAULT_HOP_LENGTH = 64
INFER_DEFAULT_SPLIT_AUDIO = True
INFER_DEFAULT_AUTOTUNE = False
INFER_DEFAULT_USE_TTA = False
INFER_DEFAULT_BATCH_SIZE = 1

# Stem-separation model sources are locked to RVC-AI-Cover-Maker references.
VOCALS_MODEL_CONFIG_URL = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml"
VOCALS_MODEL_CKPT_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
KARAOKE_MODEL_CONFIG_URL = "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml"
KARAOKE_MODEL_CKPT_URL = "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

UVR_MODEL_FILE_DEREVERB = "UVR-DeEcho-DeReverb.pth"
UVR_MODEL_FILE_DEECHO = "UVR-De-Echo-Normal.pth"
UVR_MODELS_DIR = MUSIC_MODELS_DIR / "uvr"

COVER_RESOURCES_BASE_URL = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
COVER_RMVPE_URL = f"{COVER_RESOURCES_BASE_URL}/predictors/rmvpe.pt"
COVER_FCPE_URL = f"{COVER_RESOURCES_BASE_URL}/predictors/fcpe.pt"

INFER_FIXED_VOCALS_MODEL = "Mel-Roformer"
INFER_FIXED_KARAOKE_MODEL = "Mel-Roformer Karaoke"
INFER_FIXED_DEREVERB_MODEL = "UVR-Deecho-Dereverb"
INFER_FIXED_DEECHO_MODEL = "UVR-Deecho-Normal"

ALLOWED_EXPORT_FORMATS = {"WAV", "MP3", "FLAC", "OGG", "M4A"}
ALLOWED_PITCH_EXTRACTORS = {"rmvpe", "crepe", "crepe-tiny", "fcpe"}
ALLOWED_EMBEDDERS = {
    "contentvec",
    "chinese-hubert-base",
    "japanese-hubert-base",
    "korean-hubert-base",
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


@contextmanager
def pushd(path: Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


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

    cfg_dir = APPLIO_DIR / "rvc" / "configs"
    if not cfg_dir.exists():
        raise RuntimeError(f"Applio configs dir not found: {cfg_dir}")

    required_configs = {
        "32000.json": ("32000.json", "32k.json"),
        "40000.json": ("40000.json", "40k.json"),
        "48000.json": ("48000.json", "48k.json"),
    }
    for required_name, aliases in required_configs.items():
        required_path = cfg_dir / required_name
        if required_path.exists() and required_path.stat().st_size > 0:
            continue

        copied_from = None
        for alias in aliases:
            alias_path = cfg_dir / alias
            if alias_path.exists() and alias_path.stat().st_size > 0:
                if alias_path.resolve() != required_path.resolve():
                    shutil.copyfile(alias_path, required_path)
                    copied_from = alias_path
                break

        if not required_path.exists() or required_path.stat().st_size <= 0:
            url = f"https://raw.githubusercontent.com/IAHispano/Applio/3.6.0/rvc/configs/{required_name}"
            download_file_http(url, required_path, retries=2, timeout_sec=40, min_bytes=200)
            print(
                json.dumps(
                    {
                        "event": "applio_config_downloaded",
                        "config": str(required_path),
                        "url": url,
                    }
                )
            )
        elif copied_from is not None:
            print(
                json.dumps(
                    {
                        "event": "applio_config_copied_alias",
                        "from": str(copied_from),
                        "to": str(required_path),
                    }
                )
            )

    # Applio internals sometimes use relative path "rvc/..." from process CWD.
    # Keep a legacy alias at /app/rvc so calls still work even outside /content/Applio.
    applio_rvc = APPLIO_DIR / "rvc"
    app_rvc_alias = Path("/app/rvc")
    if applio_rvc.exists() and not app_rvc_alias.exists():
        try:
            app_rvc_alias.symlink_to(applio_rvc, target_is_directory=True)
            print(json.dumps({"event": "applio_alias_created", "alias": str(app_rvc_alias), "target": str(applio_rvc)}))
        except Exception as e:
            print(json.dumps({"event": "applio_alias_create_failed", "alias": str(app_rvc_alias), "error": str(e)[:300]}))


def ensure_cover_applio():
    if not APPLIO_COVER_DIR.exists():
        raise RuntimeError(f"Cover applio directory not found: {APPLIO_COVER_DIR}")
    cover_rvc = APPLIO_COVER_DIR / "rvc"
    infer_py = cover_rvc / "infer" / "infer.py"
    if not infer_py.exists():
        raise RuntimeError(f"Cover applio infer.py not found: {infer_py}")

    # Force /app/rvc to point to cover applio rvc so relative "rvc/..." paths resolve correctly.
    app_rvc_alias = APPLIO_COVER_ROOT / "rvc"
    alias_ok = False
    try:
        if app_rvc_alias.exists() or app_rvc_alias.is_symlink():
            if app_rvc_alias.is_symlink() and app_rvc_alias.resolve() == cover_rvc.resolve():
                alias_ok = True
            elif app_rvc_alias.resolve() == cover_rvc.resolve():
                alias_ok = True
    except Exception:
        alias_ok = False

    if not alias_ok:
        try:
            if app_rvc_alias.is_symlink() or app_rvc_alias.is_file():
                app_rvc_alias.unlink()
            elif app_rvc_alias.exists() and app_rvc_alias.is_dir():
                shutil.rmtree(app_rvc_alias, ignore_errors=True)
            app_rvc_alias.symlink_to(cover_rvc, target_is_directory=True)
            print(
                json.dumps(
                    {
                        "event": "cover_rvc_alias_created",
                        "alias": str(app_rvc_alias),
                        "target": str(cover_rvc),
                    }
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create cover rvc alias {app_rvc_alias} -> {cover_rvc}: {e}"
            ) from e

    cfg_dir = cover_rvc / "configs"
    required = [
        cfg_dir / "v1" / "32000.json",
        cfg_dir / "v1" / "40000.json",
        cfg_dir / "v1" / "48000.json",
        cfg_dir / "v2" / "32000.json",
        cfg_dir / "v2" / "40000.json",
        cfg_dir / "v2" / "48000.json",
    ]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size <= 0]
    if missing:
        raise RuntimeError("Cover applio config files missing: " + ", ".join(missing))

    predictors_dir = cover_rvc / "models" / "predictors"
    rmvpe_path = predictors_dir / "rmvpe.pt"
    fcpe_path = predictors_dir / "fcpe.pt"
    if not rmvpe_path.exists() or rmvpe_path.stat().st_size < 1_000_000:
        download_file_http(COVER_RMVPE_URL, rmvpe_path, retries=3, timeout_sec=180, min_bytes=1_000_000)
        print(json.dumps({"event": "cover_predictor_ready", "name": "rmvpe", "path": str(rmvpe_path)}))
    if not fcpe_path.exists() or fcpe_path.stat().st_size < 1_000_000:
        download_file_http(COVER_FCPE_URL, fcpe_path, retries=3, timeout_sec=180, min_bytes=1_000_000)
        print(json.dumps({"event": "cover_predictor_ready", "name": "fcpe", "path": str(fcpe_path)}))


def force_cover_applio_precision(preferred: str) -> str:
    ensure_cover_applio()
    requested = normalize_precision(preferred)

    if requested == "fp32":
        effective = "fp32"
        fallback_reason = None
    else:
        fp16_supported = _TORCH_AVAILABLE and torch is not None and bool(torch.cuda.is_available())
        effective = "fp16" if fp16_supported else "fp32"
        fallback_reason = None
        if requested == "bf16":
            fallback_reason = "bf16_not_supported_in_cover_applio"
        if not fp16_supported:
            fallback_reason = "fp16_not_supported"

    fp16_run = effective == "fp16"
    cfg_dir = APPLIO_COVER_DIR / "rvc" / "configs"
    targets = [
        cfg_dir / "v1" / "32000.json",
        cfg_dir / "v1" / "40000.json",
        cfg_dir / "v1" / "48000.json",
        cfg_dir / "v2" / "32000.json",
        cfg_dir / "v2" / "40000.json",
        cfg_dir / "v2" / "48000.json",
    ]
    for p in targets:
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
        train_cfg = cfg.get("train")
        if not isinstance(train_cfg, dict):
            train_cfg = {}
            cfg["train"] = train_cfg
        train_cfg["fp16_run"] = fp16_run
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "cover_precision_forced",
                "requested": requested,
                "effective": effective,
                "configRoot": str(cfg_dir),
                "fallbackReason": fallback_reason,
            }
        )
    )
    return effective


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


_VOICE_CONVERTER_INSTANCE = None


def as_dict(value):
    return value if isinstance(value, dict) else {}


def clamp_int(value: int, min_value: int, max_value: int):
    return max(min_value, min(max_value, int(value)))


def clamp_float(value: float, min_value: float, max_value: float):
    return max(min_value, min(max_value, float(value)))


def normalize_export_format(value):
    if not isinstance(value, str):
        return "WAV"
    fmt = value.strip().upper()
    return fmt if fmt in ALLOWED_EXPORT_FORMATS else "WAV"


def normalize_pitch_extractor(value):
    if not isinstance(value, str):
        return INFER_DEFAULT_PITCH_EXTRACTOR
    extractor = value.strip().lower()
    return extractor if extractor in ALLOWED_PITCH_EXTRACTORS else INFER_DEFAULT_PITCH_EXTRACTOR


def normalize_embedder_model(value):
    if not isinstance(value, str):
        return INFER_DEFAULT_EMBEDDER
    embedder = value.strip()
    return embedder if embedder in ALLOWED_EMBEDDERS else INFER_DEFAULT_EMBEDDER


def normalize_rvc_config(raw: dict):
    return {
        "pitch": clamp_int(as_int(raw.get("pitch"), 0), -24, 24),
        "pitchExtractor": normalize_pitch_extractor(raw.get("pitchExtractor")),
        "embedderModel": normalize_embedder_model(raw.get("embedderModel")),
        "filterRadius": clamp_int(as_int(raw.get("filterRadius"), INFER_DEFAULT_FILTER_RADIUS), 0, 7),
        "searchFeatureRatio": clamp_float(as_float(raw.get("searchFeatureRatio"), INFER_DEFAULT_INDEX_RATE), 0.0, 1.0),
        "volumeEnvelope": clamp_float(as_float(raw.get("volumeEnvelope"), INFER_DEFAULT_RMS_MIX_RATE), 0.0, 1.0),
        "protectVoicelessConsonants": clamp_float(
            as_float(raw.get("protectVoicelessConsonants"), INFER_DEFAULT_PROTECT), 0.0, 0.5
        ),
        "hopLength": clamp_int(as_int(raw.get("hopLength"), INFER_DEFAULT_HOP_LENGTH), 1, 512),
        # Keep split enabled in cover mode for better consistency on long inputs.
        "splitAudio": True,
        "autotune": False,
        "inferBackingVocals": False,
    }


def normalize_audio_separation_config(raw: dict):
    return {
        "addBackVocals": as_bool(raw.get("addBackVocals"), False),
        "backVocalMode": "do_not_convert",
        "useTta": False,
        "batchSize": 1,
        "vocalsModel": INFER_FIXED_VOCALS_MODEL,
        "karaokeModel": INFER_FIXED_KARAOKE_MODEL,
        "dereverbModel": INFER_FIXED_DEREVERB_MODEL,
        "deechoEnabled": True,
        "deechoModel": INFER_FIXED_DEECHO_MODEL,
        "denoiseEnabled": False,
    }


def normalize_post_process_config(raw: dict):
    return {
        "exportFormat": "WAV",
        "deleteIntermediateAudios": as_bool(raw.get("deleteIntermediateAudios"), True),
        "reverb": False,
        "instrumentalPitch": clamp_int(as_int(raw.get("instrumentalPitch"), 0), -24, 24),
        "instrumentalPitchFollowsMainPitch": True,
        "mixGainEnabled": False,
    }


def normalize_infer_config(raw_config: dict):
    config = as_dict(raw_config)
    return {
        "rvc": normalize_rvc_config(as_dict(config.get("rvc"))),
        "audioSeparation": normalize_audio_separation_config(as_dict(config.get("audioSeparation"))),
        "postProcess": normalize_post_process_config(as_dict(config.get("postProcess"))),
    }


def get_voice_converter():
    global _VOICE_CONVERTER_INSTANCE
    if _VOICE_CONVERTER_INSTANCE is not None:
        return _VOICE_CONVERTER_INSTANCE

    ensure_cover_applio()
    cover_root = str(APPLIO_COVER_ROOT)
    if cover_root not in sys.path:
        sys.path.append(cover_root)
    with pushd(APPLIO_COVER_ROOT):
        from programs.applio_code.rvc.infer.infer import VoiceConverter  # type: ignore

        _VOICE_CONVERTER_INSTANCE = VoiceConverter()
    return _VOICE_CONVERTER_INSTANCE


def ensure_music_separation_models():
    vocals_dir = MUSIC_MODELS_DIR / "vocals_mel_roformer"
    karaoke_dir = MUSIC_MODELS_DIR / "karaoke_mel_roformer"
    vocals_cfg = vocals_dir / "config.yaml"
    vocals_ckpt = vocals_dir / "model.ckpt"
    karaoke_cfg = karaoke_dir / "config.yaml"
    karaoke_ckpt = karaoke_dir / "model.ckpt"

    if not vocals_cfg.exists():
        download_file_http(VOCALS_MODEL_CONFIG_URL, vocals_cfg, min_bytes=200)
    if not vocals_ckpt.exists() or vocals_ckpt.stat().st_size < 5_000_000:
        download_file_http(VOCALS_MODEL_CKPT_URL, vocals_ckpt, min_bytes=5_000_000)

    if not karaoke_cfg.exists():
        download_file_http(KARAOKE_MODEL_CONFIG_URL, karaoke_cfg, min_bytes=200)
    if not karaoke_ckpt.exists() or karaoke_ckpt.stat().st_size < 5_000_000:
        download_file_http(KARAOKE_MODEL_CKPT_URL, karaoke_ckpt, min_bytes=5_000_000)

    models = {
        "vocals": {"config": vocals_cfg, "ckpt": vocals_ckpt, "model_type": "mel_band_roformer"},
        "karaoke": {"config": karaoke_cfg, "ckpt": karaoke_ckpt, "model_type": "mel_band_roformer"},
    }
    print(
        json.dumps(
            {
                "event": "infer_stem_models_locked",
                "profile": "rvc_ai_cover_maker",
                "vocals": {
                    "name": INFER_FIXED_VOCALS_MODEL,
                    "configUrl": VOCALS_MODEL_CONFIG_URL,
                    "ckptUrl": VOCALS_MODEL_CKPT_URL,
                    "configPath": str(vocals_cfg),
                    "ckptPath": str(vocals_ckpt),
                },
                "karaoke": {
                    "name": INFER_FIXED_KARAOKE_MODEL,
                    "configUrl": KARAOKE_MODEL_CONFIG_URL,
                    "ckptUrl": KARAOKE_MODEL_CKPT_URL,
                    "configPath": str(karaoke_cfg),
                    "ckptPath": str(karaoke_ckpt),
                },
                "dereverb": {"name": INFER_FIXED_DEREVERB_MODEL, "modelFile": UVR_MODEL_FILE_DEREVERB},
                "deecho": {"name": INFER_FIXED_DEECHO_MODEL, "modelFile": UVR_MODEL_FILE_DEECHO},
                "denoiseEnabled": False,
            }
        )
    )
    return models


def run_music_separation(input_path: Path, store_dir: Path, model_type: str, config_path: Path, ckpt_path: Path, use_tta: bool):
    store_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(MUSIC_SEPARATION_DIR / "inference.py"),
        "--model_type",
        model_type,
        "--config_path",
        str(config_path),
        "--start_check_point",
        str(ckpt_path),
        "--input_file",
        str(input_path),
        "--store_dir",
        str(store_dir),
        "--flac_file",
        "--pcm_type",
        "PCM_16",
        "--extract_instrumental",
    ]
    if use_tta:
        cmd.append("--use_tta")

    if _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        cmd += ["--device_ids", "0"]
    else:
        cmd.append("--force_cpu")

    run(cmd, cwd=str(WORK_DIR))


def run_uvr_single_stem(
    input_path: Path,
    store_dir: Path,
    model_filename: str,
    output_single_stem: str,
    use_tta: bool,
    batch_size: int,
):
    if not _AUDIO_SEPARATOR_AVAILABLE or Separator is None:
        raise RuntimeError("audio-separator is not available in this worker image.")

    if store_dir.exists():
        shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    UVR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=str(UVR_MODELS_DIR),
        log_level=logging.WARNING,
        normalization_threshold=1.0,
        output_format="flac",
        output_dir=str(store_dir),
        output_single_stem=output_single_stem,
        vr_params={
            "batch_size": int(max(1, min(24, batch_size))),
            "enable_tta": bool(use_tta),
        },
    )
    separator.load_model(model_filename=model_filename)
    separator.separate(str(input_path))

    result = find_latest_stem_file(store_dir)
    if not result:
        raise RuntimeError(
            f"UVR stage failed for model={model_filename} stem={output_single_stem}."
        )
    return result


def find_stem_file(store_dir: Path, contains: str):
    files = sorted(store_dir.glob("*.flac"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if contains.lower() in f.name.lower():
            return f
    return None


def find_karaoke_vocal_stem(store_dir: Path):
    files = sorted(store_dir.glob("*.flac"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None

    preferred_tokens = ("_karaoke", "karaoke", "_vocals", "vocals")
    for token in preferred_tokens:
        for f in files:
            name = f.name.lower()
            if "instrumental" in name:
                continue
            if token in name:
                return f

    # Fallback: first non-instrumental stem.
    for f in files:
        if "instrumental" not in f.name.lower():
            return f
    return None


def find_latest_stem_file(store_dir: Path):
    files = sorted(store_dir.glob("*.flac"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def get_audio_sample_rate(path: Path):
    try:
        out = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
        value = int(str(out).strip().splitlines()[0])
        if value > 0:
            return value
    except Exception:
        pass
    return 44100


def build_atempo_filters(target_ratio: float):
    ratio = float(target_ratio)
    if ratio <= 0:
        return "atempo=1.0"

    factors = []
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    factors.append(ratio)
    return ",".join([f"atempo={f:.8f}" for f in factors])


def shift_audio_pitch_to_semitones(
    input_path: Path,
    output_path: Path,
    semitones: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shift = int(semitones)
    if shift == 0:
        shutil.copyfile(input_path, output_path)
        return

    factor = 2 ** (float(shift) / 12.0)
    sample_rate = get_audio_sample_rate(input_path)
    tempo_ratio = 1.0 / factor
    atempo_chain = build_atempo_filters(tempo_ratio)
    filter_chain = f"asetrate={int(round(sample_rate * factor))},aresample={sample_rate},{atempo_chain}"

    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-filter:a",
            filter_chain,
            str(output_path),
        ]
    )


def extract_model_artifact(model_zip_path: Path, extract_dir: Path):
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(model_zip_path, "r") as zf:
        zf.extractall(extract_dir)

    pth_candidates = sorted(
        [p for p in extract_dir.rglob("*.pth") if not p.name.startswith(("G_", "D_"))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pth_candidates:
        raise RuntimeError("Model artifact does not include a valid .pth file.")

    index_candidates = sorted(
        [p for p in extract_dir.rglob("*.index") if "trained" not in p.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return {
        "model_path": pth_candidates[0],
        "index_path": index_candidates[0] if index_candidates else None,
    }


def ensure_wav_for_rvc_input(input_audio_path: Path, temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_input = temp_dir / f"{input_audio_path.stem}_for_rvc.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_audio_path),
            "-ar",
            "44100",
            "-ac",
            "1",
            str(wav_input),
        ]
    )
    return wav_input


def recover_rvc_output(
    output_audio_path: Path,
    input_audio_path: Path,
    started_at: float,
    attempt: str,
):
    if output_audio_path.exists() and output_audio_path.stat().st_size > 0:
        return True

    # Applio sometimes writes output into CWD or with a rewritten extension/name.
    # Recover by probing likely locations and normalizing to requested output path.
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    probe_candidates = []
    for root in (
        output_audio_path.parent,
        input_audio_path.parent,
        APPLIO_COVER_ROOT,
        APPLIO_COVER_DIR,
        APPLIO_DIR,
        APPLIO_DIR / "assets",
        APPLIO_DIR / "outputs",
        APPLIO_DIR / "audios",
    ):
        for ext in exts:
            probe_candidates.append(root / f"{output_audio_path.stem}{ext}")
        probe_candidates.append(root / output_audio_path.name)

    seen = set()
    for cand in probe_candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.exists() and cand.stat().st_size > 0:
                if cand.resolve() != output_audio_path.resolve():
                    shutil.copyfile(cand, output_audio_path)
                print(
                    json.dumps(
                        {
                            "event": "infer_rvc_output_recovered",
                            "attempt": attempt,
                            "from": str(cand),
                            "to": str(output_audio_path),
                            "bytes": output_audio_path.stat().st_size if output_audio_path.exists() else None,
                        }
                    )
                )
                if output_audio_path.exists() and output_audio_path.stat().st_size > 0:
                    return True
        except Exception:
            continue

    # Last-resort: scan Applio tree for recently-written audio that matches output stem.
    stem = output_audio_path.stem.lower()
    recent = []
    try:
        for p in APPLIO_COVER_ROOT.rglob("*"):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            if suffix not in exts:
                continue
            name = p.name.lower()
            if stem not in name:
                continue
            st = p.stat()
            if st.st_size <= 0:
                continue
            if st.st_mtime < (started_at - 3):
                continue
            recent.append((st.st_mtime, p))
        recent.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        recent = []

    if recent:
        cand = recent[0][1]
        shutil.copyfile(cand, output_audio_path)
        print(
            json.dumps(
                {
                    "event": "infer_rvc_output_recovered_recent",
                    "attempt": attempt,
                    "from": str(cand),
                    "to": str(output_audio_path),
                    "bytes": output_audio_path.stat().st_size if output_audio_path.exists() else None,
                }
            )
        )
        if output_audio_path.exists() and output_audio_path.stat().st_size > 0:
            return True

    # Final fallback: any very recent audio file created during conversion.
    recent_any = []
    try:
        for root in (output_audio_path.parent, input_audio_path.parent, APPLIO_COVER_ROOT, APPLIO_DIR):
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                suffix = p.suffix.lower()
                if suffix not in exts:
                    continue
                if p.resolve() == input_audio_path.resolve():
                    continue
                st = p.stat()
                if st.st_size <= 0:
                    continue
                if st.st_mtime < (started_at - 3):
                    continue
                recent_any.append((st.st_mtime, st.st_size, p))
        recent_any.sort(key=lambda x: (x[0], x[1]), reverse=True)
    except Exception:
        recent_any = []

    if recent_any:
        cand = recent_any[0][2]
        shutil.copyfile(cand, output_audio_path)
        print(
            json.dumps(
                {
                    "event": "infer_rvc_output_recovered_any_recent",
                    "attempt": attempt,
                    "from": str(cand),
                    "to": str(output_audio_path),
                    "bytes": output_audio_path.stat().st_size if output_audio_path.exists() else None,
                }
            )
        )
        if output_audio_path.exists() and output_audio_path.stat().st_size > 0:
            return True

    return False


def convert_with_rvc(
    input_audio_path: Path,
    output_audio_path: Path,
    model_path: Path,
    index_path: Path | None,
    rvc_cfg: dict,
):
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    requested_split_audio = as_bool(rvc_cfg.get("splitAudio"), INFER_DEFAULT_SPLIT_AUDIO)
    attempts: list[tuple[str, Path, bool]] = [
        ("requested", input_audio_path, requested_split_audio),
    ]
    if requested_split_audio:
        attempts.append(("fallback_split_disabled", input_audio_path, False))

    wav_input = output_audio_path.parent / f"{input_audio_path.stem}_for_rvc.wav"
    attempts.append(("fallback_wav_input", wav_input, False))
    if requested_split_audio:
        attempts.append(("fallback_wav_input_with_split", wav_input, True))

    failures = []
    for idx, (attempt, attempt_input_path, split_audio) in enumerate(attempts, start=1):
        started_at = time.time()
        if output_audio_path.exists():
            try:
                output_audio_path.unlink()
            except Exception:
                pass

        if attempt_input_path == wav_input:
            try:
                attempt_input_path = ensure_wav_for_rvc_input(input_audio_path, output_audio_path.parent)
            except Exception as e:
                failures.append(f"{attempt}: wav_prepare_exception={type(e).__name__}: {str(e)[:240]}")
                print(
                    json.dumps(
                        {
                            "event": "infer_rvc_attempt_prepare_exception",
                            "attemptIndex": idx,
                            "attempt": attempt,
                            "error": str(e)[:600],
                        }
                    )
                )
                continue

        if not attempt_input_path.exists() or attempt_input_path.stat().st_size <= 0:
            failures.append(f"{attempt}: input_missing={attempt_input_path}")
            continue

        print(
            json.dumps(
                {
                    "event": "infer_rvc_attempt_start",
                    "attemptIndex": idx,
                    "attempt": attempt,
                    "input": str(attempt_input_path),
                    "splitAudio": bool(split_audio),
                }
            )
        )

        try:
            vc = get_voice_converter()
            with pushd(APPLIO_COVER_ROOT):
                vc.convert_audio(
                    audio_input_path=str(attempt_input_path),
                    audio_output_path=str(output_audio_path),
                    model_path=str(model_path),
                    index_path=str(index_path) if index_path else "",
                    embedder_model=str(rvc_cfg.get("embedderModel") or INFER_DEFAULT_EMBEDDER),
                    pitch=as_int(rvc_cfg.get("pitch"), 0),
                    f0_file=None,
                    f0_method=str(rvc_cfg.get("pitchExtractor") or INFER_DEFAULT_PITCH_EXTRACTOR),
                    filter_radius=as_int(rvc_cfg.get("filterRadius"), INFER_DEFAULT_FILTER_RADIUS),
                    index_rate=as_float(rvc_cfg.get("searchFeatureRatio"), INFER_DEFAULT_INDEX_RATE),
                    volume_envelope=as_float(rvc_cfg.get("volumeEnvelope"), INFER_DEFAULT_RMS_MIX_RATE),
                    protect=as_float(rvc_cfg.get("protectVoicelessConsonants"), INFER_DEFAULT_PROTECT),
                    split_audio=bool(split_audio),
                    f0_autotune=as_bool(rvc_cfg.get("autotune"), INFER_DEFAULT_AUTOTUNE),
                    hop_length=as_int(rvc_cfg.get("hopLength"), INFER_DEFAULT_HOP_LENGTH),
                    export_format=INFER_DEFAULT_EXPORT_FORMAT,
                    embedder_model_custom=None,
                )
        except Exception as e:
            failures.append(f"{attempt}: convert_exception={type(e).__name__}: {str(e)[:240]}")
            print(
                json.dumps(
                    {
                        "event": "infer_rvc_attempt_exception",
                        "attemptIndex": idx,
                        "attempt": attempt,
                        "error": str(e)[:600],
                    }
                )
            )
            continue

        if recover_rvc_output(
            output_audio_path=output_audio_path,
            input_audio_path=attempt_input_path,
            started_at=started_at,
            attempt=attempt,
        ):
            print(
                json.dumps(
                    {
                        "event": "infer_rvc_attempt_success",
                        "attemptIndex": idx,
                        "attempt": attempt,
                        "output": str(output_audio_path),
                        "bytes": output_audio_path.stat().st_size if output_audio_path.exists() else None,
                    }
                )
            )
            return

        failures.append(f"{attempt}: no_output")
        print(
            json.dumps(
                {
                    "event": "infer_rvc_attempt_no_output",
                    "attemptIndex": idx,
                    "attempt": attempt,
                }
            )
        )

    raise RuntimeError(
        "RVC conversion completed but output file is missing: "
        f"{output_audio_path} (attempts={'; '.join(failures)})"
    )


def mix_tracks(main_vocal_path: Path, instrumental_path: Path, output_path: Path, backing_vocal_path: Path | None = None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(main_vocal_path),
        "-i",
        str(instrumental_path),
    ]
    if backing_vocal_path:
        cmd += ["-i", str(backing_vocal_path)]
        filter_complex = "[0:a][1:a][2:a]amix=inputs=3:duration=longest:normalize=0[mix]"
    else:
        filter_complex = "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[mix]"

    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[mix]",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    run(cmd)


def export_audio_format(src_wav_path: Path, output_path: Path, export_format: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = normalize_export_format(export_format)
    if fmt == "WAV":
        shutil.copyfile(src_wav_path, output_path)
        return

    codec_args = []
    if fmt == "MP3":
        codec_args = ["-codec:a", "libmp3lame", "-b:a", "320k"]
    elif fmt == "FLAC":
        codec_args = ["-codec:a", "flac"]
    elif fmt == "OGG":
        codec_args = ["-codec:a", "libvorbis", "-q:a", "6"]
    elif fmt == "M4A":
        codec_args = ["-codec:a", "aac", "-b:a", "256k"]

    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_wav_path),
            *codec_args,
            str(output_path),
        ]
    )


def normalize_stem_keys(raw: dict):
    def clean(value):
        if not isinstance(value, str):
            return None
        v = value.strip()
        return v if v else None

    return {
        "mainVocalsKey": clean(raw.get("mainVocalsKey")),
        "backVocalsKey": clean(raw.get("backVocalsKey")),
        "instrumentalKey": clean(raw.get("instrumentalKey")),
    }


def upload_if_present(client, bucket: str, local_path: Path | None, key: str | None):
    if not local_path or not key:
        return
    if not local_path.exists() or local_path.stat().st_size == 0:
        return
    client.upload_file(str(local_path), bucket, key)


def handle_infer_job(job: dict, inp: dict, client, bucket: str):
    if "inputAudioKey" not in inp and "inputStorageKey" not in inp:
        raise RuntimeError("Missing required input: inputAudioKey")
    if "modelArtifactKey" not in inp:
        raise RuntimeError("Missing required input: modelArtifactKey")
    if "outKey" not in inp:
        raise RuntimeError("Missing required input: outKey")

    input_audio_key = str(inp.get("inputAudioKey") or inp.get("inputStorageKey") or "").strip()
    model_artifact_key = str(inp.get("modelArtifactKey") or "").strip()
    out_key = str(inp.get("outKey") or "").strip()
    if not input_audio_key:
        raise RuntimeError("inputAudioKey is empty.")
    if not model_artifact_key:
        raise RuntimeError("modelArtifactKey is empty.")
    if not model_artifact_key.lower().endswith(".zip"):
        raise RuntimeError("modelArtifactKey must point to a .zip artifact.")
    if not out_key:
        raise RuntimeError("outKey is empty.")

    config = normalize_infer_config(as_dict(inp.get("config")))
    rvc_cfg = config["rvc"]
    sep_cfg = config["audioSeparation"]
    post_cfg = config["postProcess"]
    stem_keys = normalize_stem_keys(as_dict(inp.get("stemKeys")))

    add_back_vocals = as_bool(sep_cfg.get("addBackVocals"), False)
    use_tta = as_bool(sep_cfg.get("useTta"), INFER_DEFAULT_USE_TTA)
    batch_size = clamp_int(as_int(sep_cfg.get("batchSize"), INFER_DEFAULT_BATCH_SIZE), 1, 24)
    instrumental_pitch = clamp_int(as_int(rvc_cfg.get("pitch"), 0), -24, 24)

    infer_precision = os.environ.get("APPLIO_INFER_PRECISION", "fp16").strip().lower()
    force_cover_applio_precision(infer_precision)

    gpu = get_gpu_diagnostics()
    print(json.dumps({"event": "infer_gpu_diagnostics", **gpu}))
    print(
        json.dumps(
            {
                "event": "infer_config_normalized",
                "config": config,
                "fixedModels": {
                    "vocalsModel": INFER_FIXED_VOCALS_MODEL,
                    "karaokeModel": INFER_FIXED_KARAOKE_MODEL,
                    "dereverbModel": INFER_FIXED_DEREVERB_MODEL,
                    "deechoModel": INFER_FIXED_DEECHO_MODEL,
                },
            }
        )
    )

    model_defs = ensure_music_separation_models()

    req_id = str((job or {}).get("id") or inp.get("jobId") or f"infer_{int(time.time())}")
    work_dir = WORK_DIR / "infer" / req_id
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_audio_path = work_dir / "input_audio.wav"
    model_zip_path = work_dir / "model.zip"
    model_extract_dir = work_dir / "model_artifact"
    stems_vocals_dir = work_dir / "stems_vocals"
    stems_karaoke_dir = work_dir / "stems_karaoke"
    stems_dereverb_dir = work_dir / "stems_dereverb"
    stems_deecho_dir = work_dir / "stems_deecho"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "infer_download_start", "inputAudioKey": input_audio_key, "modelArtifactKey": model_artifact_key}))
    client.download_file(bucket, input_audio_key, str(input_audio_path))
    client.download_file(bucket, model_artifact_key, str(model_zip_path))
    if not input_audio_path.exists() or input_audio_path.stat().st_size < 1024:
        raise RuntimeError("Input audio is missing or empty.")
    if not model_zip_path.exists() or model_zip_path.stat().st_size < 1024:
        raise RuntimeError("Model artifact zip is missing or empty.")
    print(json.dumps({"event": "infer_download_done", "inputBytes": input_audio_path.stat().st_size, "modelZipBytes": model_zip_path.stat().st_size}))

    model_info = extract_model_artifact(model_zip_path, model_extract_dir)
    model_path = model_info["model_path"]
    index_path = model_info["index_path"]
    print(json.dumps({"event": "infer_model_ready", "modelPath": str(model_path), "indexPath": str(index_path) if index_path else None}))

    print(json.dumps({"event": "infer_separation_vocals_start"}))
    run_music_separation(
        input_path=input_audio_path,
        store_dir=stems_vocals_dir,
        model_type=model_defs["vocals"]["model_type"],
        config_path=model_defs["vocals"]["config"],
        ckpt_path=model_defs["vocals"]["ckpt"],
        use_tta=use_tta,
    )
    vocals_stem = find_stem_file(stems_vocals_dir, "_vocals")
    instrumental_stem = find_stem_file(stems_vocals_dir, "_instrumental")
    if not vocals_stem or not instrumental_stem:
        raise RuntimeError("Could not find separated vocals/instrumental stems.")
    print(json.dumps({"event": "infer_separation_vocals_done", "vocalsStem": vocals_stem.name, "instrumentalStem": instrumental_stem.name}))

    print(json.dumps({"event": "infer_separation_backing_start"}))
    run_music_separation(
        input_path=vocals_stem,
        store_dir=stems_karaoke_dir,
        model_type=model_defs["karaoke"]["model_type"],
        config_path=model_defs["karaoke"]["config"],
        ckpt_path=model_defs["karaoke"]["ckpt"],
        use_tta=use_tta,
    )
    # Keep chain aligned with reference cover maker:
    # vocals -> karaoke vocal -> dereverb -> deecho -> RVC
    karaoke_vocal_stem = find_karaoke_vocal_stem(stems_karaoke_dir)
    backing_stem = find_stem_file(stems_karaoke_dir, "_instrumental") or find_stem_file(
        stems_karaoke_dir, "instrumental"
    )
    if not karaoke_vocal_stem:
        raise RuntimeError(
            "Could not find karaoke vocal stem after karaoke separation "
            "(expected a non-instrumental stem)."
        )
    if not backing_stem:
        print(
            json.dumps(
                {
                    "event": "infer_separation_backing_warn",
                    "message": "backing stem missing, continuing without backing",
                    "karaokeVocalStem": karaoke_vocal_stem.name,
                }
            )
        )
    else:
        print(
            json.dumps(
                {
                    "event": "infer_separation_backing_done",
                    "karaokeVocalStem": karaoke_vocal_stem.name,
                    "backingStem": backing_stem.name,
                }
            )
        )

    print(json.dumps({"event": "infer_dereverb_start", "model": INFER_FIXED_DEREVERB_MODEL}))
    dereverb_stem = run_uvr_single_stem(
        input_path=karaoke_vocal_stem,
        store_dir=stems_dereverb_dir,
        model_filename=UVR_MODEL_FILE_DEREVERB,
        output_single_stem="No Reverb",
        use_tta=use_tta,
        batch_size=batch_size,
    )
    print(json.dumps({"event": "infer_dereverb_done", "output": dereverb_stem.name}))

    print(json.dumps({"event": "infer_deecho_start", "model": INFER_FIXED_DEECHO_MODEL}))
    deecho_stem = run_uvr_single_stem(
        input_path=dereverb_stem,
        store_dir=stems_deecho_dir,
        model_filename=UVR_MODEL_FILE_DEECHO,
        output_single_stem="No Echo",
        use_tta=use_tta,
        batch_size=batch_size,
    )
    print(json.dumps({"event": "infer_deecho_done", "output": deecho_stem.name}))

    converted_main_vocals = output_dir / "main_vocals_converted.wav"
    print(json.dumps({"event": "infer_rvc_main_start"}))
    convert_with_rvc(
        input_audio_path=deecho_stem,
        output_audio_path=converted_main_vocals,
        model_path=model_path,
        index_path=index_path,
        rvc_cfg=rvc_cfg,
    )
    if not converted_main_vocals.exists() or converted_main_vocals.stat().st_size <= 0:
        raise RuntimeError(
            f"RVC main vocal output missing after conversion: {converted_main_vocals}"
        )
    print(
        json.dumps(
            {
                "event": "infer_rvc_main_done",
                "output": converted_main_vocals.name,
                "bytes": converted_main_vocals.stat().st_size,
            }
        )
    )

    final_backing_stem = None
    if add_back_vocals and backing_stem:
        backing_for_mix = output_dir / "backing_pitch_shifted.flac"
        print(
            json.dumps(
                {
                    "event": "infer_backing_pitch_shift_start",
                    "semitones": instrumental_pitch,
                }
            )
        )
        shift_audio_pitch_to_semitones(
            input_path=backing_stem,
            output_path=backing_for_mix,
            semitones=instrumental_pitch,
        )
        final_backing_stem = backing_for_mix
        print(
            json.dumps(
                {
                    "event": "infer_backing_pitch_shift_done",
                    "output": backing_for_mix.name,
                }
            )
        )

    instrumental_for_mix = instrumental_stem
    if instrumental_pitch != 0:
        shifted_instrumental = output_dir / "instrumental_pitch_shifted.flac"
        print(
            json.dumps(
                {
                    "event": "infer_instrumental_pitch_shift_start",
                    "semitones": instrumental_pitch,
                }
            )
        )
        shift_audio_pitch_to_semitones(
            input_path=instrumental_stem,
            output_path=shifted_instrumental,
            semitones=instrumental_pitch,
        )
        instrumental_for_mix = shifted_instrumental
        print(
            json.dumps(
                {
                    "event": "infer_instrumental_pitch_shift_done",
                    "output": shifted_instrumental.name,
                }
            )
        )

    final_mix_path = output_dir / "cover_final.wav"
    print(json.dumps({"event": "infer_mix_start"}))
    mix_tracks(
        main_vocal_path=converted_main_vocals,
        instrumental_path=instrumental_for_mix,
        output_path=final_mix_path,
        backing_vocal_path=final_backing_stem if add_back_vocals else None,
    )
    print(json.dumps({"event": "infer_mix_done", "output": final_mix_path.name, "bytes": final_mix_path.stat().st_size}))

    final_export_ext = post_cfg["exportFormat"].lower()
    final_upload_path = output_dir / f"cover_final.{final_export_ext}"
    if post_cfg["exportFormat"] == "WAV":
        final_upload_path = final_mix_path
    else:
        export_audio_format(
            src_wav_path=final_mix_path,
            output_path=final_upload_path,
            export_format=post_cfg["exportFormat"],
        )

    print(json.dumps({"event": "infer_upload_start", "outKey": out_key}))
    client.upload_file(str(final_upload_path), bucket, out_key)
    upload_if_present(client, bucket, converted_main_vocals, stem_keys.get("mainVocalsKey"))
    upload_if_present(client, bucket, instrumental_for_mix, stem_keys.get("instrumentalKey"))
    upload_if_present(client, bucket, final_backing_stem if add_back_vocals else None, stem_keys.get("backVocalsKey"))
    print(json.dumps({"event": "infer_upload_done"}))
    output_bytes = final_upload_path.stat().st_size if final_upload_path.exists() else None

    if post_cfg["deleteIntermediateAudios"]:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            print(json.dumps({"event": "infer_cleanup_done", "workDir": str(work_dir)}))
        except Exception as cleanup_error:
            print(json.dumps({"event": "infer_cleanup_failed", "error": str(cleanup_error)[:500]}))

    return {
        "ok": True,
        "mode": "infer",
        "outputKey": out_key,
        "outputBytes": output_bytes,
        "stemKeys": {
            "mainVocalsKey": stem_keys.get("mainVocalsKey"),
            "backVocalsKey": stem_keys.get("backVocalsKey") if add_back_vocals else None,
            "instrumentalKey": stem_keys.get("instrumentalKey"),
        },
        "modelArtifactKey": model_artifact_key,
        "inputAudioKey": input_audio_key,
        "effectiveConfig": config,
    }


def handler(job):
    print(json.dumps({"event": "runner_build", "build": RUNNER_BUILD}))
    log_runtime_dependency_info()

    bucket = require_env("R2_BUCKET")
    inp = (job or {}).get("input") or {}
    mode = str(inp.get("mode") or "train").strip().lower()
    client = s3()

    if mode == "infer":
        ensure_cover_applio()
        print(json.dumps({"event": "runtime_mode_source", "mode": "infer", "source": str(APPLIO_COVER_DIR)}))
        return handle_infer_job(job=job or {}, inp=inp, client=client, bucket=bucket)
    if mode != "train":
        raise RuntimeError(f"Unsupported mode: {mode}")

    ensure_applio()
    validate_forced_sample_rate()
    print(json.dumps({"event": "runtime_mode_source", "mode": "train", "source": str(APPLIO_DIR)}))

    preferred_precision = FORCE_TRAINING_PRECISION
    effective_precision = force_applio_precision(preferred_precision)
    if effective_precision != preferred_precision:
        raise RuntimeError(
            f"Precision requirement mismatch for mode={mode}: "
            f"requested={preferred_precision}, effective={effective_precision}"
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
