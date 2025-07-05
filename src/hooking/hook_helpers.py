"""
Tiny hook helpers for mechanistic interpretability experiments.

These helpers provide simple interfaces for common activation manipulation tasks
like ablation, patching, and capture. They build on the existing PatchSpec and
NNsight infrastructure in the codebase.
"""

import torch
from typing import Dict, List, Tuple, Optional, Literal, Union
from src.models import ModelandTokenizer
from src.functional import PatchSpec, get_hs, get_all_module_states, get_module_nnsight
from src.utils.typing import TokenizerOutput
from src.tokens import prepare_input
import logging

logger = logging.getLogger(__name__)


def ablate_layer(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_name: str,
    token_indices: Union[int, List[int], slice] = slice(None),
    return_logits: bool = True,
) -> torch.Tensor:
    """
    Zero out activations at specified layer and token positions.
    
    Args:
        mt: Model and tokenizer
        inputs: Input text or tokenized input  
        layer_name: Name of layer to ablate (e.g., "model.layers.37")
        token_indices: Which token positions to ablate (default: all)
        return_logits: Whether to return final logits
        
    Returns:
        Final layer logits after ablation
    """
    if isinstance(inputs, str):
        inputs = prepare_input(prompts=[inputs], tokenizer=mt)
    
    if isinstance(token_indices, int):
        token_indices = [token_indices]
    elif isinstance(token_indices, slice):
        seq_len = inputs.input_ids.shape[1]
        token_indices = list(range(seq_len))[token_indices]
    print(f"{token_indices=}")
    
    # Create zero patches for each token position
    patches = []
    for token_idx in token_indices:
        # Get the activation shape by running once
        sample_activation = get_hs(
            mt=mt, 
            input=inputs, 
            locations=[(layer_name, token_idx)]
        )
        zero_patch = torch.zeros_like(sample_activation)
        
        patches.append(PatchSpec(
            location=(layer_name, token_idx),
            patch=zero_patch,
            strategy="replace"
        ))
    
    # Run with ablation patches
    if return_logits:
        return get_hs(
            mt=mt,
            input=inputs,
            locations=[(mt.lm_head_name, -1)],
            patches=patches,
            return_dict=False
        )
    else:
        # Just apply patches without capturing anything specific
        with mt.trace(inputs, scan=False):
            for patch in patches:
                module_name, index = patch.location
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0]
                )
                current_state[:, index, :] = patch.patch
        return None


def patch_layer(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    stored_activations: Dict[Tuple[str, int], torch.Tensor],
    return_logits: bool = True,
) -> torch.Tensor:
    """
    Patch stored activations into specified layer positions.
    
    Args:
        mt: Model and tokenizer
        inputs: Input text or tokenized input
        stored_activations: Dict mapping (layer_name, token_idx) -> activation tensor
        return_logits: Whether to return final logits
        
    Returns:
        Final layer logits after patching
    """
    if isinstance(inputs, str):
        inputs = prepare_input(prompts=[inputs], tokenizer=mt)
    
    # Convert stored activations to PatchSpec objects
    patches = [
        PatchSpec(
            location=location,
            patch=activation,
            strategy="replace"
        )
        for location, activation in stored_activations.items()
    ]
    
    if return_logits:
        return get_hs(
            mt=mt,
            input=inputs,
            locations=[(mt.lm_head_name, -1)],
            patches=patches,
            return_dict=False
        )
    else:
        # Just apply patches
        with mt.trace(inputs, scan=False):
            for patch in patches:
                module_name, index = patch.location
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0]
                )
                current_state[:, index, :] = patch.patch
        return None


def capture_layer(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_names: Union[str, List[str]] = None,
    token_indices: Union[int, List[int], slice] = slice(None),
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> Dict[Tuple[str, int], torch.Tensor]:
    """
    Capture activations from specified layers and token positions.
    
    Args:
        mt: Model and tokenizer
        inputs: Input text or tokenized input
        layer_names: Layer name(s) to capture from. If None, captures from all layers
        token_indices: Which token positions to capture (default: all)
        kind: Type of activations to capture
        
    Returns:
        Dictionary mapping (layer_name, token_idx) -> activation tensor
    """
    if isinstance(inputs, str):
        inputs = prepare_input(prompts=[inputs], tokenizer=mt)
    
    if layer_names is None:
        # Capture from all layers of the specified kind
        return get_all_module_states(mt=mt, input=inputs, kind=kind)
    
    # Handle single layer name
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    # Handle token indices
    if isinstance(token_indices, int):
        token_indices = [token_indices]
    elif isinstance(token_indices, slice):
        seq_len = inputs.input_ids.shape[1]
        token_indices = list(range(seq_len))[token_indices]
    
    # Build locations list
    locations = [
        (layer_name, token_idx)
        for layer_name in layer_names
        for token_idx in token_indices
    ]
    
    return get_hs(mt=mt, input=inputs, locations=locations, return_dict=True)


# Convenience functions for common use cases
def ablate_residual_stream(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_idx: int,
    token_indices: Union[int, List[int], slice] = slice(None),
) -> torch.Tensor:
    """Ablate residual stream at specific layer."""
    layer_name = mt.layer_name_format.format(layer_idx)
    #print(f"{layer_name=}")
    return ablate_layer(mt, inputs, layer_name, token_indices)


def ablate_mlp(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_idx: int,
    token_indices: Union[int, List[int], slice] = slice(None),
) -> torch.Tensor:
    """Ablate MLP at specific layer."""
    layer_name = mt.mlp_module_name_format.format(layer_idx)
    return ablate_layer(mt, inputs, layer_name, token_indices)


def ablate_attention(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_idx: int,
    token_indices: Union[int, List[int], slice] = slice(None),
) -> torch.Tensor:
    """Ablate attention at specific layer."""
    layer_name = mt.attn_module_name_format.format(layer_idx)
    return ablate_layer(mt, inputs, layer_name, token_indices)


def capture_residual_stream(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_indices: Union[int, List[int]] = None,
    token_indices: Union[int, List[int], slice] = slice(None),
) -> Dict[Tuple[str, int], torch.Tensor]:
    """Capture residual stream activations."""
    if layer_indices is None:
        return capture_layer(mt, inputs, kind="residual", token_indices=token_indices)
    
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    
    layer_names = [mt.layer_name_format.format(idx) for idx in layer_indices]
    return capture_layer(mt, inputs, layer_names, token_indices)


# Example usage functions for your odd-one-out experiment
def run_necessity_sweep(
    mt: ModelandTokenizer,
    clean_prompt: str,
    target_token_ids: List[int],
    layer_range: Tuple[int, int] = (0, None),
    name_token_positions: List[int] = None,
) -> Dict[int, float]:
    """
    Run necessity sweep: ablate each layer and measure logit difference.
    
    Args:
        mt: Model and tokenizer
        clean_prompt: The clean prompt 
        target_token_ids: List of token IDs to measure logits for
        layer_range: (start_layer, end_layer) to sweep over
        name_token_positions: Token positions of the entity names
        
    Returns:
        Dictionary mapping layer_idx -> logit margin
    """
    start_layer, end_layer = layer_range
    if end_layer is None:
        end_layer = mt.n_layer
    
    # Get baseline logits
    baseline_logits = get_hs(
        mt=mt,
        input=clean_prompt,
        locations=[(mt.lm_head_name, -1)],
        return_dict=False
    )
    
    results = {}
    for layer_idx in range(start_layer, end_layer):
        # Ablate this layer at name positions
        ablated_logits = ablate_residual_stream(
            mt=mt,
            inputs=clean_prompt,
            layer_idx=layer_idx,
            token_indices=name_token_positions if name_token_positions else slice(None)
        )
        
        # Compute logit margin (correct - max_incorrect)
        correct_logit = ablated_logits[target_token_ids[0]].item()
        if len(target_token_ids) > 1:
            incorrect_logits = [ablated_logits[tid].item() for tid in target_token_ids[1:]]
            print(f"{correct_logit=}")
            print(f"{incorrect_logits=}")
            max_incorrect = max(incorrect_logits)
            margin = correct_logit - max_incorrect
        else:
            margin = correct_logit
            
        results[layer_idx] = margin
        
        logger.debug(f"Layer {layer_idx}: margin = {margin:.3f}")
    
    return results

def run_sufficiency_sweep(
    mt: ModelandTokenizer,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token_ids: List[int],
    layer_range: Tuple[int, int] = (0, None),
    name_token_positions: Union[List[int], slice, None] = None,
) -> Dict[int, float]:
    """
    Patch in *only* one layer's clean residual at a time into the corrupt run
    and measure how many logit‐points of the lost margin it recovers.
    """

    # 1) tokenize once
    clean_inputs   = prepare_input([clean_prompt],   mt.tokenizer)
    corrupt_inputs = prepare_input([corrupt_prompt], mt.tokenizer)

    # 2) grab *all* clean residuals
    clean_acts = capture_residual_stream(
        mt=mt,
        inputs=clean_inputs,
        token_indices=name_token_positions or slice(None)
    )

    # 3) compute the corrupt baseline margin
    corrupt_base = get_hs(
        mt=mt,
        input=corrupt_inputs,
        locations=[(mt.lm_head_name, -1)],
        return_dict=False
    )
    corr_id, *wrong_ids = target_token_ids
    corrupt_corr  = corrupt_base[corr_id].item()
    corrupt_wrong = [corrupt_base[i].item() for i in wrong_ids] if wrong_ids else [0.0]
    corrupt_margin = corrupt_corr - max(corrupt_wrong)

    # 4) set up sweep bounds
    start, end = layer_range
    end = end if end is not None else mt.n_layer

    results: Dict[int, float] = {}

    for L in range(start, end):
        layer_name = mt.layer_name_format.format(L)

        # 5) build exactly one PatchSpec list for this layer
        patches: List[PatchSpec] = [
            PatchSpec(location=(ln, pos), patch=act, strategy="replace")
            for (ln, pos), act in clean_acts.items()
            if ln == layer_name
               and (name_token_positions is None or pos in name_token_positions)
        ]

        # sanity check: at the first layer, print out a couple values
        if L == start:
            print("→ first layer patch specs:", patches[:3])

        #print(f"Layer {L:02d} — patching {len(patches)} slots:", patches[:3])
        #print("  corrupt tokens:", mt.tokenizer.convert_ids_to_tokens(
        #    corrupt_inputs.input_ids[0, name_token_positions]
        #))

        # 6) run the corrupt prompt *with* just this layer patched
        patched_logits = get_hs(
            mt=mt,
            input=corrupt_inputs,
            locations=[(mt.lm_head_name, -1)],
            patches=patches,
            return_dict=False
        )

        # 7) compute the patched margin
        p_corr  = patched_logits[corr_id].item()
        p_wrong = [patched_logits[i].item() for i in wrong_ids] if wrong_ids else [0.0]
        patched_margin = p_corr - max(p_wrong)

        # 8) sufficiency = how many logit points above the corrupt baseline
        results[L] = patched_margin - corrupt_margin

        logger.debug(
            f"Layer {L:02d} | corrupt_margin={corrupt_margin:.3f}  "
            f"patched_margin={patched_margin:.3f}  sufficiency={results[L]:+.3f}"
        )

    return results