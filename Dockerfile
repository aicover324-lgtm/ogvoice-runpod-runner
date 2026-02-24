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
  && mkdir -p /app/programs/applio_code \
  && cp -r /tmp/rvc_cover/programs/applio_code/* /app/programs/applio_code/ \
  && mkdir -p /app/music_separation_code \
  && mkdir -p /app/music_separation_models \
  && cp -r /tmp/rvc_cover/programs/music_separation_code/* /app/music_separation_code/ \
  && test -f /app/programs/applio_code/rvc/infer/infer.py \
  && test -f /app/music_separation_code/inference.py \
  && rm -rf /tmp/rvc_cover

# Deterministic preload of fixed inference assets.
# This bakes heavy models into the image so worker cold-start does not spend
# minutes downloading predictors/embedders/separation checkpoints.
RUN python - <<'PY'
from pathlib import Path
from urllib.request import Request, urlopen
import time

assets = [
    # Mel-Roformer vocals
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/vocals/config_deux_becruily.yaml?download=true",
        "/app/music_separation_models/vocals_mel_roformer/config.yaml",
        200,
    ),
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/vocals/becruily_deux.ckpt?download=true",
        "/app/music_separation_models/vocals_mel_roformer/model.ckpt",
        5_000_000,
    ),
    # Mel-Roformer karaoke
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/karaoke/mel_band_roformer_karaoke_aufr33_viperx_config_mel_band_roformer_karaoke.yaml?download=true",
        "/app/music_separation_models/karaoke_mel_roformer/config.yaml",
        200,
    ),
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/karaoke/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt?download=true",
        "/app/music_separation_models/karaoke_mel_roformer/model.ckpt",
        5_000_000,
    ),
    # UVR de-reverb model (deecho stage is disabled)
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/deecho-dereverb/UVR-DeEcho-DeReverb.pth?download=true",
        "/app/music_separation_models/uvr/UVR-DeEcho-DeReverb.pth",
        5_000_000,
    ),
    # tmp_applio predictors and embedder (active RVC runtime path)
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/RVC%20pitch%20predictors/rmvpe.pt?download=true",
        "/content/Applio/rvc/models/predictors/rmvpe.pt",
        1_000_000,
    ),
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/RVC%20pitch%20predictors/fcpe.pt?download=true",
        "/content/Applio/rvc/models/predictors/fcpe.pt",
        1_000_000,
    ),
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/RVC%20embedder%20(contentvec)/pytorch_model.bin?download=true",
        "/content/Applio/rvc/models/embedders/contentvec/pytorch_model.bin",
        1_000_000,
    ),
    (
        "https://huggingface.co/OrcunAICovers/stem_seperation/resolve/main/seperation_models/RVC%20embedder%20(contentvec)/Resources_embedders_contentvec_config.json?download=true",
        "/content/Applio/rvc/models/embedders/contentvec/config.json",
        200,
    ),
]

def download(url: str, dest: Path, min_bytes: int):
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print("cached", dest)
        return True, None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    last = None
    for i in range(5):
        try:
            req = Request(url, headers={"User-Agent": "ogvoice-runpod-runner/1.0"})
            with urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if not tmp.exists() or tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"downloaded file too small for {dest}")
            if dest.exists():
                dest.unlink()
            tmp.replace(dest)
            print("downloaded", dest)
            return True, None
        except Exception as e:
            last = e
            if tmp.exists():
                tmp.unlink()
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to download {url}: {last}")

for url, path, min_bytes in assets:
    download(url, Path(path), min_bytes)

print("asset preload completed successfully")
PY

# Compatibility patch for BS-Roformer karaoke config variants.
# Some checkpoints/configs include `mlp_expansion_factor` for BSRoformer; older
# upstream code does not expose this argument and crashes at runtime.
RUN python - <<'PY'
from pathlib import Path

p = Path('/app/music_separation_code/models/bs_roformer/bs_roformer.py')
if not p.exists():
    raise RuntimeError(f'Missing file: {p}')

text = p.read_text(encoding='utf-8')
orig = text

sig_old = (
    '            mask_estimator_depth=2,\n'
    '            multi_stft_resolution_loss_weight=1.,\n'
)
sig_new = (
    '            mask_estimator_depth=2,\n'
    '            mlp_expansion_factor=4,\n'
    '            multi_stft_resolution_loss_weight=1.,\n'
)

block_old = (
    '            mask_estimator = MaskEstimator(\n'
    '                dim=dim,\n'
    '                dim_inputs=freqs_per_bands_with_complex,\n'
    '                depth=mask_estimator_depth\n'
    '            )\n'
)
block_new = (
    '            mask_estimator = MaskEstimator(\n'
    '                dim=dim,\n'
    '                dim_inputs=freqs_per_bands_with_complex,\n'
    '                depth=mask_estimator_depth,\n'
    '                mlp_expansion_factor=mlp_expansion_factor,\n'
    '            )\n'
)

if 'mlp_expansion_factor=4' not in text:
    if sig_old not in text:
        raise RuntimeError('Could not patch BSRoformer signature block')
    text = text.replace(sig_old, sig_new, 1)

if 'mlp_expansion_factor=mlp_expansion_factor' not in text:
    if block_old not in text:
        raise RuntimeError('Could not patch BSRoformer mask estimator block')
    text = text.replace(block_old, block_new, 1)

if text != orig:
    p.write_text(text, encoding='utf-8')
    print('Patched', p)
else:
    print('No patch needed for', p)
PY

# Compatibility patch for NumPy>=1.24 where np.int alias is removed.
RUN python - <<'PY'
from pathlib import Path

p = Path('/app/programs/applio_code/rvc/infer/pipeline.py')
if not p.exists():
    raise RuntimeError(f'Missing file: {p}')

text = p.read_text(encoding='utf-8')
orig = text
text = text.replace('astype(np.int)', 'astype(int)')

if text != orig:
    p.write_text(text, encoding='utf-8')
    print('Patched', p, 'np.int -> int')
else:
    print('No np.int patch needed for', p)
PY

# Install Applio requirements.
# IMPORTANT: include PyTorch cu128 index so torch==2.7.1+cu128 resolves.
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r /content/Applio/requirements.txt \
     --extra-index-url https://download.pytorch.org/whl/cu128

# Our runner deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
  && python -c "import onnxruntime, audio_separator; print('onnxruntime', onnxruntime.__version__, 'audio_separator', getattr(audio_separator, '__version__', 'unknown'))" \
  && python -c "import sys; sys.path.append('/app/music_separation_code'); import utils; print('music_separation_code OK')" \
  && python -c "import sys; sys.path.append('/app'); from programs.applio_code.rvc.infer.infer import VoiceConverter; print('cover_applio_code OK', bool(VoiceConverter))"

# Toolchain needed only for pip builds; keep runtime image lean.
RUN apt-get update \
  && apt-get purge -y --auto-remove build-essential git \
  && rm -rf /var/lib/apt/lists/*

COPY handler.py /app/handler.py

# Inference assets are preloaded into the image above to minimize cold-start
# latency. Runtime download in handler.py stays as a safety fallback only.

# Work dir for jobs
RUN mkdir -p /workspace

CMD ["python", "-u", "/app/handler.py"]
