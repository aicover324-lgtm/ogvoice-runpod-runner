FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg zip \
  && rm -rf /var/lib/apt/lists/*

# Applio
RUN mkdir -p /content \
  && cd /content \
  && git clone https://github.com/IAHispano/Applio --branch 3.6.0 --single-branch

# Applio deps (pip ile)
RUN pip install --no-cache-dir -r /content/Applio/requirements.txt

# Our runner deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# work dir for jobs
RUN mkdir -p /workspace

CMD ["python", "-u", "/app/handler.py"]
