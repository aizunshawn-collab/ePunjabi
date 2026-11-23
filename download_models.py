"""Downloads all ASR/translation model weights into /app/models at container
startup rather than at Docker build time. /app/models is a persistent Fly
Volume mount (see fly.toml's [[mounts]]) - keeping model weights off the
image itself is what keeps the image under Fly's 8GB uncompressed per-machine
unpack limit (see Dockerfile history/repo memory: baking the ~523MB gated
NeMo RNNT checkpoint into the image pushed the final image back over that
limit, which fails SILENTLY - `flyctl deploy` reports "complete" while the
machine keeps serving its old image forever).

Idempotent: skips any model whose expected path already exists, so restarts
and redeploys reuse whatever is already on the volume instead of
re-downloading several GB every boot.
"""
import os

from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def download_nemo_checkpoint():
    expected = os.path.join(MODELS_DIR, 'indicconformer-nemo', 'indicconformer_stt_pa_hybrid_rnnt_large.nemo')
    if os.path.isfile(expected):
        print(f"[models] NeMo checkpoint already present at {expected}, skipping.")
        return
    token = os.environ.get('HF_TOKEN', '').strip()
    if not token:
        print("[models] HF_TOKEN not set - skipping gated NeMo RNNT checkpoint (will fall back to ONNX CTC).")
        return
    print("[models] Downloading gated IndicConformer .nemo checkpoint...")
    # HF_HUB_DISABLE_XET avoids a Xet-storage quirk where the downloaded blob
    # can end up as a symlink into a cache path outside this directory.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    path = hf_hub_download(
        repo_id='ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large',
        filename='indicconformer_stt_pa_hybrid_rnnt_large.nemo',
        local_dir=os.path.join(MODELS_DIR, 'indicconformer-nemo'),
        token=token,
    )
    if os.path.islink(path):
        import shutil
        real = os.path.realpath(path)
        shutil.copyfile(real, path + '.materialized')
        os.remove(path)
        os.rename(path + '.materialized', path)
    assert os.path.isfile(expected) and not os.path.islink(expected), 'NeMo checkpoint missing/symlinked after download'
    print(f"[models] NeMo checkpoint downloaded OK: {os.path.getsize(expected)} bytes")


def download_public_models():
    # (repo_id, local directory name under models/, optional allow_patterns)
    targets = [
        ('AI-pro11/indictrans2-en-indic-ct2-int8', 'indictrans2-en-indic-ct2-int8', None),
        ('AI-pro11/indictrans2-indic-en-ct2-int8', 'indictrans2-indic-en-ct2-int8', None),
        ('AI-pro11/indicconformer-punjabi-ctc-onnx', 'indicconformer-punjabi-ctc-onnx', None),
        ('eschmidbauer/parakeet-unified-en-0.6b-onnx', 'parakeet-unified-en-onnx', ['onnx_int8/*']),
        ('speechbrain/lang-id-voxlingua107-ecapa', 'lang-id-voxlingua107-ecapa', None),
        # Dedicated English grammar/fluency correction (see
        # grammar_correction.py docstring for why the Google Translate
        # fallback alone didn't fix the "unnatural phrasing" complaint).
        # Small (77M params, T5-small based) - negligible addition against
        # the ~4-5GB already on this volume.
        ('pszemraj/grammar-synthesis-small', 'grammar-synthesis-small', None),
    ]
    for repo_id, local_name, allow_patterns in targets:
        local_dir = os.path.join(MODELS_DIR, local_name)
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"[models] {local_name} already present, skipping.")
            continue
        print(f"[models] Downloading {repo_id} -> {local_name} ...")
        kwargs = {'repo_id': repo_id, 'local_dir': local_dir}
        if allow_patterns:
            kwargs['allow_patterns'] = allow_patterns
        snapshot_download(**kwargs)
        print(f"[models] {local_name} downloaded OK.")


if __name__ == '__main__':
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_nemo_checkpoint()
    download_public_models()
    print("[models] All model downloads complete.")
