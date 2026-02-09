import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import boto3
import runpod
from botocore.config import Config
from botocore.exceptions import ClientError


APPLIO_DIR = Path("/content/Applio")
WORK_DIR = Path("/workspace")


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


def parse_sample_rate(v, default_tag="40k"):
    """
    Accepts: "32k"/"40k"/"48k", 32000/40000/48000, "40000"
    Returns: (sr_int, tag_str like "40k")
    """
    if v is None:
        tag = default_tag
    elif isinstance(v, (int, float)):
        sr = int(v)
        if sr in (32000, 40000, 48000):
            tag = f"{sr // 1000}k"
        else:
            raise RuntimeError(f"Invalid sampleRate: {v}. Use 32000/40000/48000 or '32k'/'40k'/'48k'.")
    elif isinstance(v, str):
        s = v.strip().lower()
        if s.endswith("k"):
            n = s[:-1].strip()
            if n not in ("32", "40", "48"):
                raise RuntimeError(f"Invalid sampleRate: {v}. Use '32k'/'40k'/'48k'.")
            tag = f"{n}k"
        else:
            # "40000"
            sr = int(float(s))
            if sr not in (32000, 40000, 48000):
                raise RuntimeError(f"Invalid sampleRate: {v}. Use 32000/40000/48000 or '32k'/'40k'/'48k'.")
            tag = f"{sr // 1000}k"
    else:
        tag = default_tag

    sr = int(tag.rstrip("k")) * 1000
    return sr, tag


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


def ensure_applio():
    if not APPLIO_DIR.exists():
        raise RuntimeError("Applio directory not found in image (expected /content/Applio).")
    core = APPLIO_DIR / "core.py"
    if not core.exists():
        raise RuntimeError(f"Applio core.py not found: {core}")


def probe_audio(path: Path):
    # Optional sanity check. If ffprobe doesn't exist, we skip.
    try:
        out = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(out)
        duration = float(((data.get("format") or {}).get("duration")) or 0)
        if duration <= 0:
            raise RuntimeError("Audio duration invalid (<= 0).")
        return {"durationSec": duration}
    except FileNotFoundError:
        print("ffprobe not found; skipping audio probe.")
        return {"durationSec": None}
    except json.JSONDecodeError:
        print("ffprobe returned non-JSON; skipping audio probe.")
        return {"durationSec": None}


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


def handler(job):
    ensure_applio()

    bucket = require_env("R2_BUCKET")
    inp = (job or {}).get("input") or {}

    # Required keys (clear errors)
    if "datasetKey" not in inp:
        raise RuntimeError("Missing required input: datasetKey")
    if "outKey" not in inp:
        raise RuntimeError("Missing required input: outKey")

    dataset_key = inp["datasetKey"]
    out_key = inp["outKey"]

    model_name = inp.get("modelName")
    sr, sr_tag = parse_sample_rate(inp.get("sampleRate"), default_tag="40k")

    # UI defaults’e yakin:
    cut_preprocess = inp.get("cutPreprocess", "Automatic")
    chunk_len = as_float(inp.get("chunkLen"), 3.0)
    overlap_len = as_float(inp.get("overlapLen"), 0.3)
    normalization_mode = inp.get("normalizationMode", "post")

    f0_method = inp.get("f0Method", "rmvpe")
    include_mutes = as_int(inp.get("includeMutes"), 2)
    embedder_model = inp.get("embedderModel", "contentvec")
    index_algorithm = inp.get("indexAlgorithm", "Auto")

    total_epoch = as_int(inp.get("totalEpoch"), 200)
    batch_size = as_int(inp.get("batchSize"), 4)

    pretrained = as_bool(inp.get("pretrained"), True)
    save_every_epoch = as_int(inp.get("saveEveryEpoch"), 10)
    save_only_latest = as_bool(inp.get("saveOnlyLatest"), True)

    vocoder = inp.get("vocoder", "HiFi-GAN")

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
                "pretrained": pretrained,
                "saveOnlyLatest": save_only_latest,
            }
        )
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = WORK_DIR / model_name
    dataset_dir = job_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.wav"

    client = s3()

    # Download dataset.wav from R2
    try:
        print(json.dumps({"event": "download_start", "bucket": bucket, "key": dataset_key}))
        client.download_file(bucket, dataset_key, str(dataset_path))
        if not dataset_path.exists() or dataset_path.stat().st_size == 0:
            raise RuntimeError("Downloaded dataset.wav is missing or empty.")
        print(json.dumps({"event": "download_done", "bytes": dataset_path.stat().st_size}))
    except ClientError as e:
        raise RuntimeError(f"Failed to download from R2: s3://{bucket}/{dataset_key}\n{e}") from e

    # Optional sanity check
    audio_info = probe_audio(dataset_path)
    print(json.dumps({"event": "audio_probe", **audio_info}))

    print(json.dumps({"event": "applio_prerequisites_start"}))
    run(
        ["python", "core.py", "prerequisites", "--models", "True", "--pretraineds_hifigan", "True"],
        cwd=str(APPLIO_DIR),
    )
    print(json.dumps({"event": "applio_prerequisites_done"}))

    cpu_cores = os.cpu_count() or 2

    print(json.dumps({"event": "preprocess_start"}))
    run(
        [
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
            "--process_effects",
            "False",
            "--noise_reduction",
            "False",
            "--noise_reduction_strength",
            "0.7",
        ],
        cwd=str(APPLIO_DIR),
    )
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

    # Choose pretrained paths by sample rate tag; fallback to 48k if missing
    pretrained_root = APPLIO_DIR / "rvc/models/pretraineds/pretraineds_custom"
    g_pre = pretrained_root / f"G{sr_tag}.pth"
    d_pre = pretrained_root / f"D{sr_tag}.pth"

    if not g_pre.exists() or not d_pre.exists():
        g48 = pretrained_root / "G48k.pth"
        d48 = pretrained_root / "D48k.pth"
        if g48.exists() and d48.exists():
            print(
                json.dumps(
                    {
                        "event": "pretrained_fallback",
                        "reason": "missing_sr_specific_pretrained",
                        "wanted": {"g": str(g_pre), "d": str(d_pre)},
                        "using": {"g": str(g48), "d": str(d48)},
                    }
                )
            )
            g_pre, d_pre = g48, d48
        else:
            print(
                json.dumps(
                    {
                        "event": "pretrained_missing",
                        "reason": "no_pretrained_files_found",
                        "wanted": {"g": str(g_pre), "d": str(d_pre)},
                    }
                )
            )

    print(json.dumps({"event": "train_start"}))
    run(
        [
            "python",
            "core.py",
            "train",
            "--model_name",
            model_name,
            "--save_every_epoch",
            str(save_every_epoch),
            "--save_only_latest",
            str(save_only_latest),
            "--save_every_weights",
            "False",
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
            "False",
            "--g_pretrained_path",
            str(g_pre),
            "--d_pretrained_path",
            str(d_pre),
            "--overtraining_detector",
            "False",
            "--overtraining_threshold",
            "50",
            "--cleanup",
            "False",
            "--cache_data_in_gpu",
            "False",
            "--vocoder",
            vocoder,
            "--checkpointing",
            "False",
        ],
        cwd=str(APPLIO_DIR),
    )
    print(json.dumps({"event": "train_done"}))

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
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
