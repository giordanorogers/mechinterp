import os
import torch
import hashlib
from pathlib import Path
from typing import Optional, List, Union

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_text_hash(text: str, sentences: Optional[List[str]] = None) -> str:
    """
    Generate a unique hash based on text content and optional chunk sentences
    """

    if sentences:
        content = text + "|||" + "|||".join(sentences)
    else:
        content = text

    hash_obj = hashlib.sha256(content.encode("utf-8"))
    return hash_obj.hexdigest()[:16]

def get_cache_path(
    cache_dir: Union[str, Path],
    text_id: str,
    model_name: str,
    layer: Union[int, List[int]],
    head: int,
    suffix: str = "",
) -> str:
    """
    Generate cache file path for a specific attention matrix.
    """
    if isinstance(layer, list):
        layer = "_".join(map(str, layer))
    filename = f"{layer}_{head}{suffix}.npy"

    Path(os.path.join(cache_dir, model_name, text_id, filename)).parent.mkdir(
        parents=True, exist_ok=True,
    )
    return os.path.join(cache_dir, model_name, text_id, filename)
