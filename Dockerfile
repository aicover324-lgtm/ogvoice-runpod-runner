FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
# - build-essential: gcc/g++ needed for webrtcvad build
# - ffmpeg/libsndfile1: audio processing dependencies
# - zip: export artifact
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg zip libsndfile1 \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Applio
RUN mkdir -p /content \
  && cd /content \
  && git clone https://github.com/IAHispano/Applio --branch 3.6.0 --single-branch

# Install Applio requirements.
# IMPORTANT: include PyTorch cu128 index so torch==2.7.1+cu128 resolves.
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r /content/Applio/requirements.txt \
     --extra-index-url https://download.pytorch.org/whl/cu128

# Preload Applio prerequisites at build time to avoid runtime downloads.
# This reduces cold-start time and avoids wasting paid worker runtime.
# NOTE: Applio core.py expects to run with CWD=/content/Applio because it reads
# files like rvc/lib/tools/tts_voices.json using relative paths.
RUN cd /content/Applio \
  && python -u core.py prerequisites --models True --pretraineds_hifigan True --exe False \
  && python - <<'PY'
from pathlib import Path
p = Path('/content/Applio/.prerequisites_ready')
p.write_text('ok\n')
print('Wrote', p)
PY

# Our runner deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Download advanced pretrained weights at build time.
# This removes runtime dependency on HuggingFace/network and ensures training
# always uses these pretrains.
RUN python - <<'PY'
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

G_URL = os.environ.get(
    "CUSTOM_PRETRAIN_G_URL",
    "https://huggingface.co/OrcunAICovers/legacy_core_pretrain_v1.5/resolve/main/G_15.pth?download=true",
)
D_URL = os.environ.get(
    "CUSTOM_PRETRAIN_D_URL",
    "https://huggingface.co/OrcunAICovers/legacy_core_pretrain_v1.5/resolve/main/D_15.pth?download=true",
)

dest_dir = Path("/content/Applio/pretrained_custom")
dest_dir.mkdir(parents=True, exist_ok=True)

targets = [
    (G_URL, dest_dir / "G_15.pth"),
    (D_URL, dest_dir / "D_15.pth"),
]

def download(url: str, dest: Path, retries: int = 3, timeout_sec: int = 120):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    last_err = None
    for i in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "ogvoice-runpod-runner-build/1.0"})
            with urlopen(req, timeout=timeout_sec) as r:
                status = getattr(r, "status", 200)
                if status >= 400:
                    raise RuntimeError(f"HTTP {status} while downloading {url}")
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

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}\n{last_err}")


for url, dest in targets:
    print(f"Downloading: {url} -> {dest}")
    download(url, dest)
    print(f"OK: {dest} ({dest.stat().st_size} bytes)")

PY

# Work dir for jobs
RUN mkdir -p /workspace

CMD ["python", "-u", "/app/handler.py"]
