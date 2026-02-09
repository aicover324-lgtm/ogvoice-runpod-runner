import os
import json
import shutil
import subprocess
from pathlib import Path

import boto3
import runpod
from botocore.config import Config


APPLIO_DIR = Path("/content/Applio")
WORK_DIR = Path("/workspace")


def s3():
    endpoint = os.environ["R2_ENDPOINT"].rstrip("/")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(s3={"addressing_style": "path"}),
    )


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout[-4000:]}")
    return p.stdout


def ensure_applio():
    # Applio build-time clone edilecek; yine de kontrol edelim
    if not APPLIO_DIR.exists():
        raise RuntimeError("Applio directory not found in image")


def export_inference_zip(model_name: str, out_zip: Path):
    logs_dir = APPLIO_DIR / "logs" / model_name
    if not logs_dir.exists():
        raise RuntimeError(f"Model logs not found: {logs_dir}")

    # En yeni weight: <model>_*e_*s.pth
    weight = sorted(logs_dir.glob(f"{model_name}_*e_*s.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not weight:
        # bazen final weight <model>.pth olabilir
        final = logs_dir / f"{model_name}.pth"
        if final.exists():
            weight_path = final
        else:
            raise RuntimeError("No weight file found after training")
    else:
        weight_path = weight[0]

    index_files = sorted(logs_dir.glob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
    index_path = index_files[0] if index_files else None

    tmp_dir = out_zip.parent / f"export_{model_name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(weight_path, tmp_dir / weight_path.name)
    if index_path:
        shutil.copy2(index_path, tmp_dir / index_path.name)

    # zip
    if out_zip.exists():
        out_zip.unlink()

    run(["bash", "-lc", f"cd {tmp_dir} && zip -q -r {out_zip} ."])
    shutil.rmtree(tmp_dir, ignore_errors=True)


def handler(job):
    ensure_applio()

    bucket = os.environ["R2_BUCKET"]
    inp = (job or {}).get("input") or {}

    dataset_key = inp["datasetKey"]              # R2 key -> dataset.wav
    out_key = inp["outKey"]                      # R2 key -> model.zip
    model_name = inp.get("modelName")            # opsiyonel
    sample_rate = inp.get("sampleRate", "40k")   # 32k / 40k / 48k
    sr = int(sample_rate.rstrip("k")) * 1000

    # UI defaults’e yakin:
    cut_preprocess = inp.get("cutPreprocess", "Automatic")
    chunk_len = float(inp.get("chunkLen", 3))
    overlap_len = float(inp.get("overlapLen", 0.3))
    normalization_mode = inp.get("normalizationMode", "post")

    f0_method = inp.get("f0Method", "rmvpe")
    include_mutes = int(inp.get("includeMutes", 2))
    embedder_model = inp.get("embedderModel", "contentvec")
    index_algorithm = inp.get("indexAlgorithm", "Auto")

    total_epoch = int(inp.get("totalEpoch", 200))
    batch_size = int(inp.get("batchSize", 4))
    pretrained = bool(inp.get("pretrained", True))
    save_every_epoch = int(inp.get("saveEveryEpoch", 10))
    save_only_latest = bool(inp.get("saveOnlyLatest", True))
    vocoder = inp.get("vocoder", "HiFi-GAN")

    # Unique model_name
    if not model_name:
        req_id = (job or {}).get("id", "job")
        model_name = f"ogvoice_{req_id[:12]}"

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = WORK_DIR / model_name
    dataset_dir = job_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "dataset.wav"

    # Download dataset.wav from R2
    client = s3()
    client.download_file(bucket, dataset_key, str(dataset_path))

    # Applio prerequisites (ilk seferde uzun surebilir; warm worker’da hizlanir)
    run(["python", "core.py", "prerequisites", "--models", "True", "--pretraineds_hifigan", "True"], cwd=str(APPLIO_DIR))

    cpu_cores = os.cpu_count() or 2

    # 1) preprocess (dataset_path bir klasor)
    run([
        "python", "core.py", "preprocess",
        "--model_name", model_name,
        "--dataset_path", str(dataset_dir),
        "--sample_rate", str(sr),
        "--cpu_cores", str(cpu_cores),
        "--cut_preprocess", str(cut_preprocess),
        "--chunk_len", str(chunk_len),
        "--overlap_len", str(overlap_len),
        "--normalization_mode", str(normalization_mode),
        "--process_effects", "False",
        "--noise_reduction", "False",
        "--noise_reduction_strength", "0.7",
    ], cwd=str(APPLIO_DIR))

    # 2) extract
    run([
        "python", "core.py", "extract",
        "--model_name", model_name,
        "--f0_method", f0_method,
        "--sample_rate", str(sr),
        "--cpu_cores", str(cpu_cores),
        "--gpu", "0",
        "--embedder_model", embedder_model,
        "--embedder_model_custom", "",
        "--include_mutes", str(include_mutes),
    ], cwd=str(APPLIO_DIR))

    # 3) index
    run([
        "python", "core.py", "index",
        "--model_name", model_name,
        "--index_algorithm", index_algorithm,
    ], cwd=str(APPLIO_DIR))

    # 4) train
    run([
        "python", "core.py", "train",
        "--model_name", model_name,
        "--save_every_epoch", str(save_every_epoch),
        "--save_only_latest", str(save_only_latest),
        "--save_every_weights", "False",
        "--total_epoch", str(total_epoch),
        "--sample_rate", str(sr),
        "--batch_size", str(batch_size),
        "--gpu", "0",
        "--pretrained", str(pretrained),
        "--custom_pretrained", "False",
        "--g_pretrained_path", str(APPLIO_DIR / "rvc/models/pretraineds/pretraineds_custom/G48k.pth"),
        "--d_pretrained_path", str(APPLIO_DIR / "rvc/models/pretraineds/pretraineds_custom/D48k.pth"),
        "--overtraining_detector", "False",
        "--overtraining_threshold", "50",
        "--cleanup", "False",
        "--cache_data_in_gpu", "False",
        "--vocoder", vocoder,
        "--checkpointing", "False",
    ], cwd=str(APPLIO_DIR))

    # Export minimal inference zip
    out_zip = job_dir / "model.zip"
    export_inference_zip(model_name, out_zip)

    # Upload to R2
    client.upload_file(str(out_zip), bucket, out_key)

    return {
        "ok": True,
        "modelName": model_name,
        "artifactKey": out_key,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
