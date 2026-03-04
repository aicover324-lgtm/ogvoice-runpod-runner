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

# Training runtime tuning patch for Applio:
# - Make DataLoader worker count configurable
# - Limit thread fan-out inside DataLoader worker processes
# This improves CPU-bound input pipeline stability without changing model quality.
RUN python - <<'PY'
from pathlib import Path

p = Path('/content/Applio/rvc/train/train.py')
if not p.exists():
    raise RuntimeError(f'Missing file: {p}')

text = p.read_text(encoding='utf-8')
orig = text

helper_marker = "def _int_env(name, default, min_value=None, max_value=None):"
if helper_marker not in text:
    anchor = "logging.getLogger(\"torch\").setLevel(logging.ERROR)\n"
    if anchor not in text:
        raise RuntimeError("Could not find logging anchor in train.py for helper insertion")
    helper_block = '''logging.getLogger("torch").setLevel(logging.ERROR)


def _int_env(name, default, min_value=None, max_value=None):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except Exception:
            value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return int(value)


def _available_cpu_count():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _resolve_dataloader_workers():
    available = _available_cpu_count()
    default_workers = max(1, min(4, available))
    return _int_env("OGV_DATALOADER_WORKERS", default_workers, min_value=0, max_value=available)


def _worker_thread_limits():
    return _int_env("OGV_DATALOADER_WORKER_THREADS", 1, min_value=1, max_value=4)


def _dataloader_worker_init_fn(_worker_id):
    worker_threads = _worker_thread_limits()
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(worker_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(worker_threads)
    try:
        torch.set_num_threads(worker_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
'''
    text = text.replace(anchor, helper_block, 1)

old_loader_block = '''    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
'''
new_loader_block = '''    resolved_num_workers = _resolve_dataloader_workers()
    resolved_prefetch_factor = _int_env(
        "OGV_DATALOADER_PREFETCH_FACTOR", 8, min_value=1, max_value=32
    )
    resolved_worker_threads = _worker_thread_limits()
    print(
        "DataLoader config"
        f" | workers={resolved_num_workers}"
        f" | prefetch_factor={resolved_prefetch_factor if resolved_num_workers > 0 else 0}"
        f" | worker_threads={resolved_worker_threads}"
        " | pin_memory=True"
    )

    train_loader_kwargs = {
        "dataset": train_dataset,
        "num_workers": resolved_num_workers,
        "shuffle": False,
        "pin_memory": True,
        "collate_fn": collate_fn,
        "batch_sampler": train_sampler,
        "persistent_workers": resolved_num_workers > 0,
    }
    if resolved_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = resolved_prefetch_factor
        train_loader_kwargs["worker_init_fn"] = _dataloader_worker_init_fn

    train_loader = DataLoader(**train_loader_kwargs)
'''
if old_loader_block in text:
    text = text.replace(old_loader_block, new_loader_block, 1)
elif "train_loader = DataLoader(**train_loader_kwargs)" not in text:
    raise RuntimeError("Could not patch DataLoader block in train.py")

timing_block_old = '''    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu:
'''
timing_block_new = '''    epoch_recorder = EpochRecorder()
    # Timing instrumentation (no behavior change): helps detect CPU-bound input stalls.
    timing_data_wait_total = 0.0
    timing_h2d_total = 0.0
    timing_compute_total = 0.0
    timing_batches = 0
    timing_prev_batch_end = ttime()
    timing_sync_for_metrics = str(os.environ.get("OGV_TRAIN_TIMING_SYNC", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            timing_batch_start = ttime()
            timing_data_wait_total += max(0.0, timing_batch_start - timing_prev_batch_end)

            h2d_start = ttime()
            if device.type == "cuda" and not cache_data_in_gpu:
'''
if timing_block_old in text:
    text = text.replace(timing_block_old, timing_block_new, 1)
elif "timing_data_wait_total = 0.0" not in text:
    raise RuntimeError("Could not patch timing init block in train.py")

compute_block_old = '''            else:
                loss_gen_all.backward()
                grad_norm_g = commons.grad_norm(net_g.parameters())
                optim_g.step()

            global_step += 1
'''
compute_block_new = '''            else:
                loss_gen_all.backward()
                grad_norm_g = commons.grad_norm(net_g.parameters())
                optim_g.step()
            if device.type == "cuda" and timing_sync_for_metrics:
                # Optional strict GPU timing for diagnostics (disabled by default).
                torch.cuda.synchronize(device_id)
            timing_compute_total += max(0.0, ttime() - compute_start)
            timing_batches += 1

            global_step += 1
'''
if compute_block_old in text:
    text = text.replace(compute_block_old, compute_block_new, 1)
elif "timing_compute_total +=" not in text:
    raise RuntimeError("Could not patch compute timing block in train.py")

h2d_block_old = '''            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned

            (
'''
h2d_block_new = '''            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned
            timing_h2d_total += max(0.0, ttime() - h2d_start)

            (
'''
if h2d_block_old in text:
    text = text.replace(h2d_block_old, h2d_block_new, 1)
elif "timing_h2d_total +=" not in text:
    raise RuntimeError("Could not patch H2D timing block in train.py")

forward_block_old = '''            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
'''
forward_block_new = '''            compute_start = ttime()
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
'''
if forward_block_old in text and "compute_start = ttime()" not in text:
    text = text.replace(forward_block_old, forward_block_new, 1)

pbar_block_old = '''            pbar.update(1)
        # end of batch train
'''
pbar_block_new = '''            pbar.update(1)
            timing_prev_batch_end = ttime()
        # end of batch train
'''
if pbar_block_old in text:
    text = text.replace(pbar_block_old, pbar_block_new, 1)
elif "timing_prev_batch_end = ttime()" not in text:
    raise RuntimeError("Could not patch pbar timing tail block in train.py")

record_block_old = '''        if overtraining_detector:
            remaining_epochs_gen = overtraining_threshold - consecutive_increases_gen
            remaining_epochs_disc = (
                overtraining_threshold * 2 - consecutive_increases_disc
            )
            record = (
                record
                + f" | Number of epochs remaining for overtraining: g/total: {remaining_epochs_gen} d/total: {remaining_epochs_disc} | smoothed_loss_gen={smoothed_value_gen:.3f} | smoothed_loss_disc={smoothed_value_disc:.3f}"
            )
        print(record)
'''
record_block_new = '''        if overtraining_detector:
            remaining_epochs_gen = overtraining_threshold - consecutive_increases_gen
            remaining_epochs_disc = (
                overtraining_threshold * 2 - consecutive_increases_disc
            )
            record = (
                record
                + f" | Number of epochs remaining for overtraining: g/total: {remaining_epochs_gen} d/total: {remaining_epochs_disc} | smoothed_loss_gen={smoothed_value_gen:.3f} | smoothed_loss_disc={smoothed_value_disc:.3f}"
            )
        if timing_batches > 0:
            data_wait_avg_ms = (timing_data_wait_total / timing_batches) * 1000.0
            h2d_avg_ms = (timing_h2d_total / timing_batches) * 1000.0
            compute_avg_ms = (timing_compute_total / timing_batches) * 1000.0
            wait_to_compute_ratio = (
                (timing_data_wait_total / timing_compute_total)
                if timing_compute_total > 1e-9
                else 0.0
            )
            record = (
                record
                + f" | data_wait_avg_ms={data_wait_avg_ms:.1f}"
                + f" | h2d_avg_ms={h2d_avg_ms:.1f}"
                + f" | gpu_compute_avg_ms={compute_avg_ms:.1f}"
                + f" | wait_to_compute_ratio={wait_to_compute_ratio:.3f}"
            )
        print(record)
'''
if record_block_old in text:
    text = text.replace(record_block_old, record_block_new, 1)
elif "wait_to_compute_ratio=" not in text:
    raise RuntimeError("Could not patch epoch record timing block in train.py")

if text != orig:
    p.write_text(text, encoding='utf-8')
    print('Patched', p, 'for configurable DataLoader workers and worker thread limits')
else:
    print('No training tuning patch needed for', p)
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
