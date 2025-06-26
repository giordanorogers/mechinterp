import copy
import string
import gc
import torch
from dataclasses import dataclass
from typing import Union, Optional, Literal, Any
from src.models import ModelandTokenizer, unwrap_tokenizer
from src.tokens import prepare_input
from src.utils.printer import printer
from src.utils.typing import (
    ArrayLike, Tokenizer, TokenizerOutput, PredictedToken
)
from nltk.corpus import stopwords

def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

def get_keywords_from_text(
    text: str,
    tokenizer: Tokenizer | ModelandTokenizer,
    maybe_prepend_space: bool = True,
) -> list[int]:
    tokenizer = unwrap_tokenizer(tokenizer)
    if maybe_prepend_space is True and text.startswith(" ") is False:
        text = f" {text}"
    tokenized = tokenizer(text, add_special_tokens=False).input_ids
    # print([tokenizer.decode(t) for t in tokenized])
    filtered = []
    prev_tok = " "
    for idx, t_idx in enumerate(tokenized):
        tok = tokenizer.decode(t_idx)
        skip = False
        if t_idx in tokenizer.all_special_ids:
            skip = True
        if tok in string.whitespace:
            skip = True
        if tok.strip() in string.punctuation:
            skip = True
        if tok.strip().lower() in stopwords.words("english"):
            # print(tokenizer.decode(tokenized[idx + 1]))
            if idx < len(tokenized) - 1 and tokenizer.decode(
                tokenized[idx + 1]
            ).startswith(" "):
                skip = True
        if (
            prev_tok.endswith(" ") is False and tok.startswith(" ") is False
        ):  # continuation of a word, safe to ignore?
            skip = True

        if skip is False:
            filtered.append(t_idx)

        prev_tok = tok
    return filtered

@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 5,
    interested_tokens: tuple[int] = (),
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
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

    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t
                ),
            )
            for t in interested_tokens
        }

        interested_logits = {
            k: v
            for k, v in sorted(
                interested_logits.items(), key=lambda x: x[1][1].prob, reverse=True
            )
        }
        return candidates, interested_logits
    return candidates

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
    k: int = 5,
    interested_tokens: tuple[int] = (),
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    tokenizer = unwrap_tokenizer(tokenizer)
    # print(type(tokenizer))
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()
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

    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t,
                ),
            )
            for t in interested_tokens
        }
        # print(interested_logits)
        interested_logits = {
            k: v
            for k, v in sorted(
                interested_logits.items(), key=lambda x: x[1][1].prob, reverse=True
            )
        }
        return candidates, interested_logits
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
    k: int = 5, # Top K most probable tokens to consider
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
        batch_topk = batch_probs.topk(k=k, dim=-1)

        # TODO: Add token of interest logic.

        for batch_order, (token_ids, token_probs) in enumerate(
            zip(batch_topk.indices, batch_topk.values)
        ):
            top_pred = interpret_logits(
                tokenizer=mt,
                logits=batch_logits[batch_order],
                k=k,
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

def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer

def generate_with_patch(
    mt: ModelandTokenizer,
    inputs: str | TokenizerOutput,
    n_gen_per_prompt: int = 5,
    max_new_tokens: int = 20,
    patches: Optional[list[PatchSpec]] = None,
    do_sample: bool = True,
    patch_strategy: Literal["replace", "add"] = "replace",
    patch_at_all_generations: bool = False,
    remove_prefix: bool = False,
    **kwargs,
) -> list[str]:
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(
            prompts=[inputs],
            tokenizer=mt,
            n_gen_per_prompt=n_gen_per_prompt,
        )

    with mt.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
        **kwargs,
    ) as gen_trace:
        if patches is not None:
            if patch_at_all_generations:
                mt.all()
            for cur_patch in patches:
                module_name, index = cur_patch.location
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                if patch_strategy == "replace":
                    current_state[:, index, :] = cur_patch.patch
                elif patch_strategy == "add":
                    current_state[:, index, :] += cur_patch.patch
                else:
                    raise ValueError("patch_strategy must be one of 'replace', 'add'")
        gen_out = mt.generator.output.save()

    start = 0
    if remove_prefix:
        start = inputs.input_ids.shape[1]
    return mt.tokenizer.batch_decode(
        gen_out.sequences[:, start:], skip_special_tokens=True
    )

def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)

def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)

def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"

def untuple(object: Any):
    if isinstance(object, tuple) or (
        "LanguageModelProxy" in str(type(object)) and len(object) > 1
    ):
        return object[0]
    return object

@torch.inference_mode()
def get_hs(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    return_dict: bool = False,
) -> torch.Tensor | dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    if isinstance(locations, tuple):
        locations = [locations]
    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False
        
        head_id = module_name.split(".")[-1]
        layer_id = ".".join(module_name.split(".")[:-1])

        return layer_id, int(head_id)
    
    layer_names = [layer_name for layer_name, _ in locations]
    layer_names = list(set(layer_names))
    layer_states = {layer_name: torch.empty(0) for layer_name in layer_names}
    with mt.trace(input, scan=False):
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                if is_an_attn_head(module_name) is True:
                    raise NotImplementedError(
                        "patching not supported yet for attn heads"
                    )
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                current_state[:, index, :] = cur_patch.patch
        
        for layer_name in layer_names:
            if is_an_attn_head(layer_name) is False:
                module = get_module_nnsight(mt, layer_name)
                layer_states[layer_name] = module.output.save()
            else:
                attn_module_name, head_idx = is_an_attn_head(layer_name)
                o_proj_name = attn_module_name + ".o_proj"
                head_dim = mt.n_embd // mt.model.config.num_attention_heads
                o_proj = get_module_nnsight(mt, o_proj_name)
                layer_states[layer_name] = o_proj.input[0][0][
                    :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                ].save()

    hs = {}

    for layer_name, index in locations:
        hs[(layer_name, index)] = untuple(layer_states[layer_name].value)[
            :, index, :
        ].squeeze()

    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs

@torch.inference_mode
def get_all_module_states(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError("kind must be one of 'residual', 'mlp', 'attention'")
    
    layer_and_index = []
    for layer_idx in range(mt.n_layer):
        for token_idx in range(input.input_ids.shape[1]):
            layer_and_index.append((layer_name_format.format(layer_idx), token_idx))

    return get_hs(mt, input, layer_and_index)

# useful for saving with jsons
def detensorize(inp: dict[Any, Any] | list[dict[Any, Any]], to_numpy: bool = False):
    if isinstance(inp, list):
        return [detensorize(i) for i in inp]
    if isinstance(inp, dict) is False:
        try:
            cls = type(inp)
            inp = inp.__dict__
        except Exception:
            return inp
    else:
        cls = None

    inp = copy.deepcopy(inp)
    for k in inp:
        if isinstance(inp[k], torch.Tensor):
            if len(inp[k].shape) == 0:
                inp[k] = inp[k].item()
            else:
                inp[k] = inp[k].tolist() if to_numpy is False else inp[k].cpu().numpy()
        else:
            inp[k] = detensorize(inp[k])

    free_gpu_cache()

    if cls is None:
        return inp
    else:
        if cls != TokenizerOutput:
            return cls(**inp)
        else:
            return cls(data=inp)