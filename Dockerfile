FROM python:3.11-slim

WORKDIR /app

# Minimal system deps (boto3 için yeterli)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# RunPod runtime, handler() fonksiyonunu buradan çağırır
CMD ["python", "-c", "import handler; print('container ready')"]
