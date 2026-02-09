import os
import json
import boto3
from botocore.config import Config


def _s3():
    endpoint = os.environ["R2_ENDPOINT"].rstrip("/")
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]

    # R2 için path-style genelde sorunsuz
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(s3={"addressing_style": "path"}),
    )


def handler(event):
    """
    RunPod serverless handler.
    event["input"] içinden payload gelir.
    """
    bucket = os.environ["R2_BUCKET"]
    inp = (event or {}).get("input") or {}

    # Basit test input: datasetKey ver (opsiyonel)
    dataset_key = inp.get("datasetKey")

    s3 = _s3()

    out = {
        "ok": True,
        "bucket": bucket,
        "checked": False,
        "exists": None,
        "datasetKey": dataset_key,
    }

    if dataset_key:
        out["checked"] = True
        try:
            s3.head_object(Bucket=bucket, Key=dataset_key)
            out["exists"] = True
        except Exception:
            out["exists"] = False

    return out
