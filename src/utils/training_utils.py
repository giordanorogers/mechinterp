import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

N_LAYER_DICT = {
    "meta-llama/Llama-3.3-70B-Instruct": 80,
    "Qwen/Qwen2.5-72B-Instruct": 80,
    "meta-llama/Meta-Llama-3-8B-Instruct": 32
}

def get_device_map(model_name, upto_layer: int, n_gpus: Optional[int] = None):
    n_gpus = torch.cuda.device_count() if n_gpus is None else n_gpus
    if model_name not in N_LAYER_DICT or n_gpus < 2:
        msg = ""
        if model_name not in N_LAYER_DICT:
            msg = f" Model {model_name} not supported "
        if n_gpus < 2:
            msg += f" Only {n_gpus} GPU(s) available "
        msg += "using default device map = `auto`."
        logger.warning(msg)
        return "auto"

    n_layers = N_LAYER_DICT[model_name]
    block_names = [f"model.layers.{i}" for i in range(n_layers)]
    other_modules = ["model.embed_tokens", "model.norm", "model.rotary_emb", "lm_head"]

    device_map = {}

    # 1. Assign other_modules to GPU 0
    for mod in other_modules:
        device_map[mod] = n_gpus - 1

    # 2. Assign trainable blocks (0:upto_layer) evenly across GPUs
    trainable_blocks = block_names[:upto_layer]
    for idx, block in enumerate(trainable_blocks):
        device_map[block] = idx % n_gpus

    # 3. Assign remaining blocks (upto_layer:n_layers) evenly across GPUs
    non_trainable_blocks = block_names[upto_layer:]
    for idx, block in enumerate(non_trainable_blocks):
        device_map[block] = idx % n_gpus

    return device_map