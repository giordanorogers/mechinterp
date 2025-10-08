import nnsight
from nnsight import LanguageModel
import torch
from typing import Literal, List, Tuple

@nnsight.trace
def streamer(tokens, model, max_length=80, state=None):
    # Initialize state if not provided
    if state is None:
        state = {'current_line': '', 'current_line_length': 0}

    token = tokens[-1] # only use last token

    # Decode the token
    decoded_token = model.tokenizer.decode(token).encode("unicode_escape").decode()

    if (decoded_token == '\\n') or (decoded_token == '\n'):  # Handle explicit newline tokens
        # Print the current line and reset state
        print('',flush=True)
        state['current_line'] = ''
        state['current_line_length'] = 0
    elif (decoded_token == '\n\n') or (decoded_token == '\\n\\n'):
        # Print the current line and reset state
        print('',flush=True)
        print('',flush=True)
        state['current_line'] = ''
        state['current_line_length'] = 0
    else:
        # Check if adding the token would exceed the max length
        if state['current_line_length'] + len(decoded_token) > max_length:
            print('',flush=True)
            state['current_line'] = decoded_token  # Start a new line with the current token
            state['current_line_length'] = len(decoded_token)
            print(decoded_token, flush=True, end="")  # Print ONLY the new token
        else:
            # Add a space if the line isn't empty and append the token
            if state['current_line']:
                state['current_line'] += decoded_token
            else:
                state['current_line'] = decoded_token
            state['current_line_length'] += len(decoded_token)
            print(decoded_token, flush=True, end="")  # Print ONLY the new token

    return state

def generate_with_streaming(
    model,
    prompt,
    max_new_tokens=1000,
    line_length=160,
    do_sample=False,
    token_return_type: Literal['ids', 'text'] = 'ids',
    output_attns: bool = False,
):
    token_ids = []
    state = {'current_line': '', 'current_line_length': 0}
    with model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        output_attentions=output_attns
    ) as tracer:
        with tracer.all():
            # Access model output
            out = model.model.output.save()
            
            last_hs = out.last_hidden_state
            logits = model.lm_head(last_hs)

            attentions = out.attentions.save() if output_attns else None

            # Apply softmax to obtain probs and save the result
            probs = torch.nn.functional.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)
            tokens = max_probs.indices.cpu().tolist()
            token_ids.append(tokens[0][-1]).save()

            state = streamer(tokens[0], model, max_length=line_length, state=state)

    prompt_token_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    all_token_ids = prompt_token_ids + token_ids

    return {
        "token_list": [model.tokenizer.decode(token) for token in all_token_ids],
        "token_ids": all_token_ids,
        "text": model.tokenizer.decode(all_token_ids),
        "attentions": attentions if output_attns else None
    }
    
def get_raw_tokens(model_name, text: str) -> List[str]:
    """Convert text to raw tokens."""
    model = LanguageModel(model_name)
    tokens_int = model.tokenizer.encode(text)
    tokens_words = model.tokenizer.convert_ids_to_tokens(tokens_int)
    return tokens_words

def get_sentence_token_boundaries(
    text: str, sentences: List[str], model_name: str
) -> List[Tuple[int, int]]:
    """
    Get exact token boundaries for sentences within the full text.
    This accounts for tokenization effects where tokens may be different
    when sentences are tokenized together vs separately.

    Args:
        text: Full text containing all sentences
        sentences: List of sentence strings
        model_name: Model name for tokenizer

    Returns:
        List of (start, end) token positions for each sentence
    """
    if not sentences:
        return None

    import re

    def normalize_spaces(s: str) -> str:
        """Replace various Unicode spaces with regular space."""
        return re.sub(r"[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]", " ", s)

    char_positions = []
    search_start = 0

    text_normalized = normalize_spaces(text)

    for sentence in sentences:
        sentence_normalized = normalize_spaces(sentence)

        norm_pos = text_normalized.find(sentence_normalized, search_start)
        if norm_pos == -1:
            sentence_stripped = sentence_normalized.strip()
            norm_pos = text_normalized.find(sentence_stripped, search_start)
            if norm_pos == -1:
                raise ValueError(f"Sentence not found in text: {sentence}")
            norm_end = norm_pos + len(sentence_stripped)
        else:
            norm_end = norm_pos + len(sentence_normalized)

        original_pos = 0
        normalized_count = 0
        actual_start = -1
        actual_end = -1

        for i, char in enumerate(text):
            if normalized_count == norm_pos and actual_start == -1:
                actual_start = i
            if normalized_count == norm_end:
                actual_end = i
                break
            if normalize_spaces(char) == " " or char == text_normalized[normalized_count]:
                normalized_count += 1

        if actual_end == -1 and normalized_count == norm_end:
            actual_end = len(text)

        char_positions.append((actual_start, actual_end))
        search_start = norm_end

    token_boundaries = []

    for char_start, char_end in char_positions:
        if char_start > 0:
            tokens_to_start = len(get_raw_tokens(model_name, text[:char_start]))
        else:
            tokens_to_start = 0

        tokens_to_end = len(get_raw_tokens(model_name, text[:char_end]))

        token_boundaries.append((tokens_to_start, tokens_to_end))

    return token_boundaries
