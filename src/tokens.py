import torch
from contextlib import contextmanager
from typing import Optional, Literal, Iterator, Any
from src.utils.typing import Tokenizer, TokenizerOutput
from src.utils.printer import printer
from src.models import (
    ModelandTokenizer, determine_device, unwrap_tokenizer
)

def maybe_prefix_bos(tokenizer, prompt: str) -> str:
    """ Prefix prompt with EOS token if model has no special start token. """
    tokenizer = unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "bos_token"):
        prefix = tokenizer.bos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt

@contextmanager
def set_padding_side(
    tokenizer: Tokenizer, padding_side: str = "right"
) -> Iterator[None]:
    """
    Temporarily set padding side for tokenizer.

    Useful for when you want to batch generate with causal LMs like GPT,
    as these require the padding to be on the left side in such settings
    but are much easier to mess around with when the padding, by default,
    is on the right.

    Example usage:
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        with tokenizer_utils.set_padding_side(tokenizer, "left"):
            inputs = mt.tokenizer(...)
        # ... later
        model.generate(**inputs)
    """
    _padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    yield
    tokenizer.padding_side = _padding_side

def prepare_input(
    prompts: str | list[str],
    tokenizer: ModelandTokenizer | Tokenizer,
    n_gen_per_prompt: int = 1,
    device: torch.device = "cpu",
    add_bos_token: bool = False,
    return_offsets_mapping = False,
    padding: str = "longest",
    padding_side: Optional[Literal["left", "right"]] = None,
    **kwargs
) -> TokenizerOutput:
    """ Prepare input for the model. """
    if isinstance(tokenizer, ModelandTokenizer):
        device = determine_device(
            tokenizer
        )

    tokenizer = unwrap_tokenizer(tokenizer)
    prompts = [prompts] if isinstance(prompts, str) else prompts
    if add_bos_token:
        prompts = [maybe_prefix_bos(tokenizer, p) for p in prompts]
    prompts = [p for p in prompts for _ in range(n_gen_per_prompt)]

    padding_side = padding_side or tokenizer.padding_side

    # Temporarily set padding side (e.g., "left" for generation [<pad>, token]),
    # then tokenize promps. Restores original padding settings afterward.
    with set_padding_side(tokenizer, padding_side):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=padding,
            return_offsets_mapping=return_offsets_mapping,
            **kwargs,
        )

    # TODO: Consider accounting for offset calculation.

    inputs = inputs.to(device)
    return inputs

def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[ModelandTokenizer | Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.
    
    The kwargs are forwarded to the tokenizer.
    
    A simple example:
    
        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...
        
        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)
        
    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurrence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')
    
    if occurrence < 0:
        # If occurrence is negative, count from the right
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = prepare_input(
            string, return_offsets_mapping=True, tokenizer=tokenizer, **kwargs
        )
        offset_mapping = tokens.offset_mapping[0]

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        if token_char_start == token_char_end:
            # Skip special tokens
            # TODO: Determine if this is the proper way of doing this.
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    assert (
        token_start is not None
    ), "Are you working with Llama-3? Try passing the ModelandTokenizer object as the tokenizer"
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)

def insert_padding_before_pos(
    inp: TokenizerOutput,
    token_position: int,
    pad_len: int,
    pad_id: int,
    fill_attn_mask: bool = False,
):
    """

    Inserts padding tokens before any position
    use cases:
    * Alignment of token positions in a bunch of sequences.
    * Getting rid of positional embeddings

    TEST:

    for idx, (tok_id, attn_mask) in enumerate(zip(inp.input_ids[0], inp.attention_mask[0])):
        print(f"{idx=} [{attn_mask}] | {mt.tokenizer.decode(tok_id)}")

    """
    input_ids = torch.cat(
        [
            inp.input_ids[:, :token_position],
            torch.full(
                (1, pad_len),
                pad_id,
                dtype=inp.input_ids.dtype,
                device=inp.input_ids.device,
            ),
            inp.input_ids[:, token_position:],
        ],
        dim=1,
    )

    attention_mask = torch.cat(
        [
            inp.attention_mask[:, :token_position],
            torch.full(
                (1, pad_len),
                fill_attn_mask,
                dtype=inp.attention_mask.dtype,
                device=inp.attention_mask.device,
            ),
            inp.attention_mask[:, token_position:],
        ],
        dim=1,
    )
    return TokenizerOutput(
        data={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

def insert_padding_before_subj(
    inp: TokenizerOutput,
    subj_range: tuple[int, int],
    subj_ends: int,
    pad_id: int,
    fill_attn_mask: bool = False,
):
    return insert_padding_before_pos(
        inp=inp,
        token_position=subj_range[0],
        pad_len=subj_ends - subj_range[1],
        pad_id=pad_id,
        fill_attn_mask=fill_attn_mask,
    )

def align_patching_positions(
    mt: ModelandTokenizer,
    prompt_template: str,
    clean_subj: str,
    patched_subj: str,
    clean_input: Optional[TokenizerOutput] = None,
    patched_input: Optional[TokenizerOutput] = None,
    trace_start_marker: Optional[str] = None,
) -> dict:
    if clean_input is None:
        clean_input = prepare_input(
            prompts=prompt_template.format(clean_subj),
            tokenizer=mt,
            return_offsets_mapping=True,
        )
    else:
        assert "offset_mapping" in clean_input
    if patched_input is None:
        patched_input = prepare_input(
            prompts=prompt_template.format(patched_subj),
            tokenizer=mt,
            return_offsets_mapping=True,
        )
    else:
        assert "offset_mapping" in patched_input

    clean_subj_range = find_token_range(
        string=prompt_template.format(clean_subj),
        substring=clean_subj,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=clean_input["offset_mapping"][0],
    )
    print(f"{clean_subj_range=}")
    
    patched_subj_range = find_token_range(
        string=prompt_template.format(patched_subj),
        substring=patched_subj,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=patched_input["offset_mapping"][0],
    )
    print(f"{patched_subj_range}")

    trace_start_idx = None
    if trace_start_marker is not None:
        trace_start_idx = (
            find_token_range(
                string=prompt_template.format(clean_subj),
                substring=trace_start_marker,
                tokenizer=mt.tokenizer,
                occurrence=-1,
                offset_mapping=clean_input["offset_mapping"][0],
            )[1]
            -1
        )
        print(f"{trace_start_idx=}")
        assert trace_start_idx <= min(
            clean_subj_range[0], patched_subj_range[0]
        ), f"{trace_start_idx=} has to be smaller than {min(clean_subj_range[0], patched_subj_range[0])=}"
        
    if clean_subj_range == patched_subj_range:
        subj_start, subj_end = clean_subj_range
    else:
        subj_end = max(clean_subj_range[1], patched_subj_range[1])
        clean_input = insert_padding_before_subj(
            inp=clean_input,
            subj_range=clean_subj_range,
            subj_ends=subj_end,
            pad_id=mt.tokenizer.pad_token_id,
            fill_attn_mask=True,
        )
        patched_input = insert_padding_before_subj(
            inp=patched_input,
            subj_range=patched_subj_range,
            subj_ends=subj_end,
            pad_id=mt.tokenizer.pad_token_id,
            fill_attn_mask=True,
        )

        clean_subj_shift = subj_end - clean_subj_range[1]
        clean_subj_range = (clean_subj_range[0] + clean_subj_shift, subj_end)
        patched_subj_shift = subj_end - patched_subj_range[1]
        patched_subj_range = (patched_subj_range[0] + patched_subj_shift, subj_end)
        subj_start = min(clean_subj_range[0], patched_subj_range[0])

        if trace_start_idx is not None:
            trace_start_idx += clean_subj_shift

    return dict(
        clean_input=clean_input,
        patched_input=patched_input,
        subj_range=(subj_start, subj_end),
        trace_start_idx=trace_start_idx,
    )