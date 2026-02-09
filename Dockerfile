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

# Our runner deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Work dir for jobs
RUN mkdir -p /workspace

CMD ["python", "-u", "/app/handler.py"]
