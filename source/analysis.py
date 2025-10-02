from collections import defaultdict
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Optional, Literal
from nnsight import LanguageModel
import numpy as np
from tqdm import trange
import logging
import torch
from dataclasses import dataclass

@dataclass
class Prediction:
    token_str: str
    token_id: int
    prob: float
    logit: float

    def __str__(self):
        return f'"{self.token_str}"[{self.token_id}] (p={self.prob}, logit={self.logit})'

logger = logging.getLogger(__name__)

def collect_activations(
    model: LanguageModel,
    prompt: str,
    kind: Optional[Literal["residual", "mlp", "attention"]] = "residual",
    remote: bool = False,
    trace_idx: int | None = None,
    model_family: Optional[Literal["llama", "gpt"]] = "llama",
) -> list:
    """
    Collect per-layer activations for a given component.

    Args:
        model: A NNSight Language Model object.
        prompt: The prompt to run the model on.
        kind: The kind of component to collect the activations of.
        remote: Whether to run the model locally or remotely on NDIF.
        trace_idx: The token position to collect activations at.
            Defaults to all positions.

    Returns:
        np.array
    """
    components = {
        "residual":  lambda layer: layer.output,
        "mlp":       lambda layer: layer.mlp.output,
        "attention": lambda layer: layer.self_attn.output
    }

    indexer = (0, trace_idx) if trace_idx is not None else (0,)

    activations = []
    
    if model_family == "llama":
        layers = model.model.layers
    elif model_family == "gpt":
        layers = model.transformer.h

    with torch.no_grad():
        with model.trace(prompt, remote=remote):
            for layer in layers:
                node = components[kind](layer)
                activations.append(node[indexer].save())
    
    return activations

def activation_patch(
    model: LanguageModel,
    prompt: str,
    activations: list,
    target_token_id: int,
    target_logits: float,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    patch_range: Optional[list]= None,
    log: bool = False
) -> list:
    
    patched_activations = []
    num_layers = model.config.num_hidden_layers
    prompt_token_ids = model.tokenizer.encode(prompt)

    if patch_range is None:
        patch_range = range(len(prompt_token_ids))
    
    # If block for kind
    if kind == "residual":

        # Loop through layers
        for l in trange(num_layers, desc="All Layers"):

            layer_patch = []

            # Loop through prompt tokens
            for t in trange(len(prompt_token_ids), desc=f"Layer {l} Tokens"):
                    
                with model.trace(prompt):

                    # If we're at a token we want to patch
                    if t in patch_range:

                        # Replace the residual stream
                        model.model.layers[l].output[0][:, t, :] = activations[l][:, t, :]

                    # Get the model output logits
                    patched_logits = model.output.logits[0].save()

                    # Get the logits for the target
                    patched_target_logits = patched_logits[-1, target_token_id].item().save()

                    # Calculate the logit difference
                    logit_diff = patched_target_logits - target_logits

                    # Save the logit_diff
                    logit_diff = logit_diff.save()

                # Append the token to our list of logit differences at the current layer
                layer_patch.append(logit_diff)

                if log:
                    # Log the layer, token, and logit difference
                    logger.info(f"L{l}, T{t}, LD={logit_diff}")

            # Append each layer's logit differences list to our overall list of lists
            patched_activations.append(layer_patch)

    return patched_activations

            
def logit_lens(
    model: LanguageModel,
    prompt: str,
    kind: Literal["full", "last"] = "last",
    k: int = 5,
    return_logits: bool = False
):
    """ Warning: Only supports llama models! """
    logit_lens = []
    num_layers = model.config.num_hidden_layers

    if kind == "last":
        logits_probs = []
        with torch.no_grad():
            with model.trace(prompt):
                logits = model.output.logits[0].save()
                probs = logits.softmax(dim=-1).save()
                logits_probs.append((logits, probs))
        logits = logits_probs[0][0][-1].topk(k=k).values
        probs = logits_probs[0][1][-1].topk(k=k).values
        token_ids = logits_probs[0][1][-1].topk(k=k).indices
        tokens = [model.tokenizer.decode(id) for id in token_ids]

        for t, l, p, id in zip(tokens, logits, probs, token_ids):
            prediction = Prediction(
                token_str=t,
                token_id=id,
                prob=p,
                logit=l
            )
            logit_lens.append(prediction)

        return logit_lens


    elif kind == "full":

        layer_logits = []

        with torch.no_grad():
            with model.trace(prompt):
                for layer in model.model.layers:
                    # residual stream at this layer
                    h = layer.output[0]
                    h_ln = model.model.norm(h)
                    logits = model.lm_head(h_ln).save()
                    layer_logits.append(logits)

        results = defaultdict(list)
        for li, logits in enumerate(layer_logits):
            last = logits[-1]
            topk = last.topk(k=k)
            ids = topk.indices
            toks = [model.tokenizer.decode(int(i)) for i in ids]
            probs = last.softmax(dim=-1)[ids]
            for rank, (t, i, p, l) in enumerate(zip(toks, ids, probs, topk.values), start=1):
                results[li].append(f'{rank}. "{t}"[{int(i)}] (p={float(p):.6f}, logit={float(l):.6f})')

        return results