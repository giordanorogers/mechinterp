import torch
from src.models import ModelandTokenizer
from typing import Union, List
from src.tokens import TokenizerOutput
from src.functional import prepare_input, get_hs

def ablate_layer(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_name: str,
    token_indices: Union[int, List[int], slice] = slice(None),
    return_logits: bool = True
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
                        else m odule.output[0]
                    )
                    current_state[:, index, :] = patch.patch
            return None

def ablate_residual_stream(
    mt: ModelandTokenizer,
    inputs: Union[str, TokenizerOutput],
    layer_idx: int,
    token_indices: Union[int, List[int], slice] = slice(None),
) -> torch.Tensor:
    """ Ablate residual stream at specific layer. """
    layer_name = mt.layer_name_format.format(layer_idx)
    return ablate_layer(mt, inputs, layer_name, token_indices)