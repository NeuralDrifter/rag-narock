"""
indexing/embedder.py — model loading and embedding.
Provides clean public API and handles local caches and API calls.
"""

import os
import sys
import json
import gc
import logging
from typing import List
import config.settings as rag_settings
import rag_backends
from core.integrity import save_index_integrity
from core.index_manager import resolve_index_dir

logger = logging.getLogger(__name__)


def setup_hf_home_fallback():
    """Configure HuggingFace to use a local 'models' folder at the project root.
    Creates the directory if it does not exist, and points cache directories to it."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_models_dir = os.path.join(project_root, "models")
    
    # Create the directory if it is not there
    try:
        os.makedirs(project_models_dir, exist_ok=True)
    except Exception as e:
        logger.debug("Failed to create models directory: %s", e)

    # Point HuggingFace caching variables to the project root models folder
    os.environ["HF_HUB_CACHE"] = project_models_dir
    os.environ["HF_HOME"] = os.path.join(project_models_dir, "hf_home")


setup_hf_home_fallback()

_model = None
_model_name = None
_model_device = None

def _resolve_embedding_device(gpu_flag=None):
    use_gpu = gpu_flag if gpu_flag is not None else rag_settings.get('gpu_indexing')
    if not use_gpu:
        return 'cpu'
    device = _detect_best_device()
    if device == 'cpu':
        try:
            import torch
            build = torch.__version__
        except Exception:
            build = 'unknown'
        if '+cpu' in build.lower():
            print(
                "WARNING: GPU indexing requested but this PyTorch build is CPU-only "
                f"({build}). Install a CUDA or XPU build to use GPU embedding.",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: GPU indexing requested but no GPU detected, falling back to CPU",
                file=sys.stderr,
            )
    return device

def _resolve_embedding_model_name():
    backend = rag_settings.get('embedding_backend')
    model = rag_settings.get('embedding_model') if backend == 'local' else rag_settings.get('api_model')
    return backend, model

def _save_index_metadata(index_dir, meta):
    with open(os.path.join(index_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    save_index_integrity(index_dir)

def _auto_lock_index(index_dir):
    lock_path = os.path.join(index_dir, ".locked")
    if not os.path.exists(lock_path):
        with open(lock_path, 'w') as f:
            f.write("locked\n")

def _release_gpu_memory():
    gc.collect()
    try:
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def _detect_best_device():
    """Auto-detect best available device: xpu > cuda > cpu."""
    saved = {}
    for key in ('CUDA_VISIBLE_DEVICES', 'ONEAPI_DEVICE_SELECTOR', 'SYCL_DEVICE_FILTER'):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        import torch
        if hasattr(torch, 'xpu') and torch.xpu.is_available() and torch.xpu.device_count() > 0:
            return 'xpu'
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 'cuda'
    except Exception:
        pass
    finally:
        os.environ.update(saved)
    return 'cpu'

_DEVICE_ENV = {
    'cpu': {
        'set': {"CUDA_VISIBLE_DEVICES": "", "ONEAPI_DEVICE_SELECTOR": "opencl:cpu"},
        'remove': [],
    },
    'xpu': {
        'set': {},
        'remove': ["CUDA_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER"],
    },
    'cuda': {
        'set': {"ONEAPI_DEVICE_SELECTOR": "opencl:cpu"},
        'remove': ["CUDA_VISIBLE_DEVICES"],
    },
}

def _prepare_env_for_device(device):
    env_config = _DEVICE_ENV.get(device, _DEVICE_ENV['cpu'])
    for key in env_config['remove']:
        os.environ.pop(key, None)
    os.environ.update(env_config['set'])

def _suppress_library_noise():
    import warnings
    warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')
    warnings.filterwarnings(
        'ignore',
        message='.*pin_memory.*no accelerator is found.*',
        category=UserWarning,
    )
    import logging
    logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
    os.environ["TQDM_DISABLE"] = "1"

def _load_from_cache(model_name, device):
    from sentence_transformers import SentenceTransformer
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)
    return model

def _download_model(model_name, device):
    from sentence_transformers import SentenceTransformer
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("TQDM_DISABLE", None)
    model = SentenceTransformer(model_name, device=device, local_files_only=False)
    os.environ["TQDM_DISABLE"] = "1"
    return model

def _get_model(model_name=None, device='cpu'):
    global _model, _model_name, _model_device
    if model_name is None:
        model_name = rag_settings.get('embedding_model')

    if _model is not None and (_model_name != model_name or _model_device != device):
        print(f"Switching model: '{_model_name}' ({_model_device}) -> '{model_name}' ({device})", file=sys.stderr)
        del _model
        _model = None
        _model_name = None
        _model_device = None
        gc.collect()

    if _model is None:
        _suppress_library_noise()
        _prepare_env_for_device(device)

        import torch
        if device == 'cpu':
            torch.set_num_threads(min(8, os.cpu_count() or 4))

        dev_label = device.upper()

        try:
            print(f"Loading embedding model '{model_name}' ({dev_label}, local cache)...", file=sys.stderr)
            _model = _load_from_cache(model_name, device)
            _model_name = model_name
            _model_device = device
            print(f"Model ready on {dev_label} (offline).", file=sys.stderr)
        except Exception:
            print(f"Model '{model_name}' not cached locally. Downloading...", file=sys.stderr)
            _model = _download_model(model_name, device)
            _model_name = model_name
            _model_device = device
            print(f"Model downloaded and cached on {dev_label}. Future loads will be offline.", file=sys.stderr)
    return _model

def _unload_embedding_model():
    global _model, _model_device
    if rag_settings.get('embedding_backend') in ('ollama', 'lmstudio'):
        return
    if _model is not None:
        was_gpu = _model_device and _model_device != 'cpu'
        if was_gpu:
            try:
                _model.cpu()
            except Exception:
                pass
        del _model
        _model = None
        _model_device = None
        if was_gpu:
            _release_gpu_memory()
            _prepare_env_for_device('cpu')
        else:
            gc.collect()

def embed_texts_api(texts: List[str], backend: str, override_model=None, override_url=None):
    import numpy as np
    import urllib.request

    api_model = override_model or rag_settings.get('api_model')
    if override_url:
        base_url = override_url.rstrip('/')
    elif backend == 'ollama':
        base_url = rag_settings.get('ollama_url').rstrip('/')
    else:
        base_url = rag_settings.get('lmstudio_url').rstrip('/')
    url = f"{base_url}/v1/embeddings"

    all_embs = []
    batch_size = 32
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        if total > batch_size:
            print(f"  API embed batch {start // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
                  f"({len(batch)} texts)...", file=sys.stderr)

        payload = json.dumps({"model": api_model, "input": batch}).encode('utf-8')
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach {backend} at {base_url}: {e}\n"
                f"Is {backend} running? Start it and ensure model '{api_model}' is available."
            ) from e

        sorted_data = sorted(result['data'], key=lambda d: d['index'])
        for item in sorted_data:
            all_embs.append(item['embedding'])

    embs = np.array(all_embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs

def embed_texts(texts: List[str], batch_size: int = 64,
                override_backend=None, override_model=None, override_url=None,
                device: str = 'cpu'):
    import numpy as np
    backend = override_backend or rag_settings.get('embedding_backend')
    if backend in ('ollama', 'lmstudio'):
        return embed_texts_api(texts, backend, override_model=override_model, override_url=override_url)
    model = _get_model(model_name=override_model, device=device)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 100,
                        normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)

def resolve_embedding_for_index(index_name):
    index_dir = resolve_index_dir(index_name)
    meta = rag_backends.get_index_meta_with_defaults(index_dir)

    emb_backend = meta.get('embedding_backend', 'local')
    emb_model = meta.get('embedding_model', 'all-MiniLM-L6-v2')
    storage = meta.get('storage_backend', 'faiss')
    api_url = None
    warning = None

    if storage in ('sqlite-vec', 'sqlite-doc'):
        try:
            import sqlite_vec
        except ImportError:
            return {
                'backend': emb_backend, 'model': emb_model, 'api_url': None,
                'storage_backend': storage,
                'warning': (
                    f"ERROR: Index '{index_name}' uses {storage} for storage, but sqlite-vec is not\n"
                    f"installed in your Python environment.\n\n"
                    f"To fix this:\n"
                    f"  1. Install it:  pip install sqlite-vec\n"
                    f"  2. Verify:      python -c \"import sqlite_vec; print('OK')\"\n"
                    f"  3. Or re-index using FAISS (the default): rag.py settings -> RAG tab -> Storage Backend -> FAISS"
                ),
            }

    if emb_backend == 'local':
        pass

    elif emb_backend == 'ollama':
        import urllib.request
        api_url = meta.get('ollama_url', rag_settings.get('ollama_url'))
        base_url = api_url.rstrip('/')
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method='GET')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                model_names = [m.get('name', '').split(':')[0] for m in data.get('models', [])]
                if emb_model not in model_names:
                    try:
                        pull_data = json.dumps({"name": emb_model}).encode('utf-8')
                        pull_req = urllib.request.Request(
                            f"{base_url}/api/pull", data=pull_data,
                            headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(pull_req, timeout=300):
                            pass
                        warning = f"Auto-pulled Ollama model '{emb_model}'"
                    except Exception:
                        warning = (
                            f"WARNING: Ollama model '{emb_model}' not found. "
                            f"Pull it with: ollama pull {emb_model}"
                        )
        except Exception:
            return {
                'backend': emb_backend, 'model': emb_model, 'api_url': api_url,
                'storage_backend': storage,
                'warning': (
                    f"ERROR: Index '{index_name}' requires Ollama model '{emb_model}' but Ollama\n"
                    f"is not reachable at {base_url}.\n\n"
                    f"To fix this:\n"
                    f"  1. Start Ollama:  ollama serve\n"
                    f"  2. Pull the model: ollama pull {emb_model}\n"
                    f"  3. Verify it's running: curl {base_url}/api/tags\n"
                    f"  4. If Ollama is on a different URL, update it in: rag.py settings -> Models tab"
                ),
            }

    elif emb_backend == 'lmstudio':
        import urllib.request
        api_url = meta.get('lmstudio_url', rag_settings.get('lmstudio_url'))
        base_url = api_url.rstrip('/')
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", method='GET')
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            return {
                'backend': emb_backend, 'model': emb_model, 'api_url': api_url,
                'storage_backend': storage,
                'warning': (
                    f"ERROR: Index '{index_name}' requires LM Studio model '{emb_model}' but\n"
                    f"LM Studio is not reachable at {base_url}.\n\n"
                    f"To fix this:\n"
                    f"  1. Open LM Studio and start the local server\n"
                    f"  2. Load the model '{emb_model}' in the Embeddings tab\n"
                    f"  3. Verify it's running: curl {base_url}/v1/models\n"
                    f"  4. If LM Studio is on a different URL, update it in: rag.py settings -> Models tab"
                ),
            }

    return {
        'backend': emb_backend, 'model': emb_model, 'api_url': api_url,
        'storage_backend': storage, 'warning': warning,
    }

# Compatibility definitions for public indexing shim
def embed_text(texts, model_name=None, device=None):
    """Embed list of texts. Returns list of vectors."""
    return embed_texts(texts, override_model=model_name, device=device or 'cpu')

def get_embedder(model_name=None, device='cpu'):
    return _get_model(model_name, device)

def unload_model():
    _unload_embedding_model()

def resolve_for_index(name):
    return resolve_embedding_for_index(name)
