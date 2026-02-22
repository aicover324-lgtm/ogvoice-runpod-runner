FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# System deps
# - build-essential: gcc/g++ needed for webrtcvad build
# - ffmpeg/libsndfile1: audio processing dependencies
# - zip: export artifact
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg zip libsndfile1 \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Applio (shallow clone to keep image smaller)
RUN mkdir -p /content \
  && cd /content \
  && GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/IAHispano/Applio --branch 3.6.0 --single-branch --depth 1 \
  && rm -rf /content/Applio/.git

# Bundle Music Source Separation code used by RVC AI Cover Maker UI.
# We use this path for Mel-Roformer stem models to match UI behavior.
RUN git clone https://github.com/Eddycrack864/RVC-AI-Cover-Maker-UI --depth 1 /tmp/rvc_cover \
  && mkdir -p /app/music_separation_code \
  && mkdir -p /app/music_separation_models \
  && cp -r /tmp/rvc_cover/programs/music_separation_code/* /app/music_separation_code/ \
  && test -f /app/music_separation_code/inference.py \
  && rm -rf /tmp/rvc_cover

# Install Applio requirements.
# IMPORTANT: include PyTorch cu128 index so torch==2.7.1+cu128 resolves.
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r /content/Applio/requirements.txt \
     --extra-index-url https://download.pytorch.org/whl/cu128

# Our runner deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
  && python -c "import onnxruntime, audio_separator; print('onnxruntime', onnxruntime.__version__, 'audio_separator', getattr(audio_separator, '__version__', 'unknown'))" \
  && python -c "import sys; sys.path.append('/app/music_separation_code'); import utils; print('music_separation_code OK')"

# Toolchain needed only for pip builds; keep runtime image lean.
RUN apt-get update \
  && apt-get purge -y --auto-remove build-essential git \
  && rm -rf /var/lib/apt/lists/*

COPY handler.py /app/handler.py

# Heavy model assets are fetched lazily by handler.py and cached per worker.
# This keeps the container image smaller and reduces pull/extract delay.

# Work dir for jobs
RUN mkdir -p /workspace

CMD ["python", "-u", "/app/handler.py"]
