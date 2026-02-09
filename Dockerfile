FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates ffmpeg zip libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Applio
RUN mkdir -p /content \
  && cd /content \
  && git clone https://github.com/IAHispano/Applio --branch 3.6.0 --single-branch

RUN pip install --no-cache-dir -r /content/Applio/requirements.txt

# Our runner deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

RUN mkdir -p /workspace

CMD ["python", "-u", "/app/handler.py"]
