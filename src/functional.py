import gc
import torch
from dataclasses import dataclass
from typing import Union, Optional
from src.models import ModelandTokenizer, unwrap_tokenizer
from src.tokens import prepare_input
from src.utils.printer import printer
from src.utils.typing import (
    ArrayLike, Tokenizer, TokenizerOutput, PredictedToken
)

def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

# dataclass: Automatic special method creation
# frozen=False: The object is mutable -- can change field values after creation
## default behavior but included for clarity
@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
    clean: Optional[torch.Tensor] = None

def normalize_token_of_interest(token_of_interest, num_inputs):
    """
    Normalize token_of_interest to format:
        [[tokens_for_input1], [tokens_for_input2], ...]

    Handles:
    - Single token: "hello" -> [["hello"]]
    - List of tokens: ["hello", "world"] -> [["hello", "world"]]
    - Already nested: [["hello"], ["world"]] -> [["hello"], ["world"]]
    """
    # Ensure it's a list
    if not isinstance(token_of_interest, list):
        token_of_interest = [token_of_interest]

    # Check if we need to add another level of nesting
    # If first element is not a list/array, we have a flat list that needs wrapping
    if not isinstance(token_of_interest[0], ArrayLike):
        # This is a flat list for a single input
        if num_inputs == 1:
            token_of_interest = [token_of_interest]
        else:
            raise ValueError(
                f"Got flat list of token but {num_inputs} inputs. \
                    Need nested structure."
            )
    
    # Validate length matches inputs
    if len(token_of_interest) != num_inputs:
        raise ValueError(
            f"token_of_interest length ({len(token_of_interest)}) \
                must match number of inputs ({num_inputs})"
        )
    
    return token_of_interest

#InferenceMode is a context manager analogous to :class:`~no_grad`
#to be used when you are certain your operations will have no interactions
#with autograd (e.g., model training).
@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    topK: int = 5,
    interested_tokens: tuple[int] = (),
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    tokenizer = unwrap_tokenizer(tokenizer)
    # squeeze: Removes dimensions of size 1 from the tensor
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=topK).indices.squeeze().tolist()
    if isinstance(top_k_indices, ArrayLike) is False:
        top_k_indices = [top_k_indices]

    candidates = [
        PredictedToken(
            token=tokenizer.decode(t),
            prob=probs[t].item(),
            logit=logits[t].item(),
            token_id=t,
        )
        for t in top_k_indices
    ]

    # TODO: Add interested_tokens logic

    return candidates

#InferenceMode is a context manager analogous to :class:`~no_grad`
#to be used when you are certain your operations will have no interactions
#with autograd (e.g., model training).
@torch.inference_mode()
def predict_next_token(
    mt: ModelandTokenizer,
    # Union type; Union[X, Y] means either X or Y
    # TokenizerOutput is a transformers BatchEncoding tokenizer dict-like object
    inputs: Union[str, list[str]] | TokenizerOutput,
    topK: int = 5, # Top K most probable tokens to consider
    batch_size: int = 8,
    # Set a token of interest as [input, token_num], or a list of [input, token_num]'s
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
):
    """ Predict the next token(s) given the input. """
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            # Offset mappings aren't needed for prediction and can cause issues downstream
            inputs.pop("offset_mapping")
    else:
        # Wraps a single string in a list
        # Leaves lists of strings unchanged
        inputs = [inputs] if isinstance(inputs, str) else inputs

    if token_of_interest is not None:
        if isinstance(inputs, TokenizerOutput):
            num_inputs = len(inputs["input_ids"])
        else:
            num_inputs = len(inputs)
        token_of_interest = normalize_token_of_interest(token_of_interest, num_inputs)
        track_interesting_tokens = []

    is_tokenized = isinstance(inputs, TokenizerOutput)
    total_len = len(inputs["input_ids"]) if is_tokenized else len(inputs)

    # TODO: Add patches normalization here.

    predictions = []
    for i in range(0, total_len, batch_size):
        if is_tokenized is False:
            batch_inputs = prepare_input(
                tokenizer=mt,
                prompts=inputs[i : i + batch_size],
                padding_side="left"
            )
        else:
            batch_inputs = {
                key: val[i : i + batch_size] if isinstance(val, ArrayLike) else val
                for key, val in inputs.items()
            }
    
        # scan: Enables tracing to access/modify module activations and gradients during
        ## forward pass, and allows intervention on model internals across multiple inputs.
        with mt.trace(batch_inputs, scan=False, validate=False) as tr:
            # TODO: Add patching logic
            batch_logits = mt.output.logits.save()

        batch_logits = batch_logits[:, -1, :]
        batch_probs = batch_logits.float().softmax(dim=-1)
        batch_topk = batch_probs.topk(k=topK, dim=-1)

        # TODO: Add token of interest logic.

        for batch_order, (token_ids, token_probs) in enumerate(
            zip(batch_topk.indices, batch_topk.values)
        ):
            top_pred = interpret_logits(
                tokenizer=mt,
                logits=batch_logits[batch_order],
                topK=topK,
                #interested_tokens=(
                #    batch_interested_token_indices[batch_order]
                #    if token_of_interest is not None
                #    else []
                #),
            )

            # TODO: Add token of interest logic

            predictions.append(top_pred)

        free_gpu_cache()

    # TODO: Add toke of interest logic

    return predictions
