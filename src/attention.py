import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from src.functional import PatchSpec
from src.models import ModelandTokenizer
from src.tokens import prepare_input
from src.utils.typing import TokenizerOutput
from src.utils.printer import printer

@dataclass(frozen=False)
class AttentionInformation(DataClassJsonMixin):
    tokenized_prompt: list[str]
    attention_matrices: np.ndarray
    logits: torch.Tensor | None = None

    def __init__(
        self,
        prompt: str,
        tokenized_prompt: list[str],
        attention_matrices: torch.tensor,
        logits: torch.Tensor | None = None
    ):
        assert (
            len(tokenized_prompt) == attention_matrices.shape[-1]
        ), "Tokenized prompt and attention matrices must have the same length."
        assert (
            len(attention_matrices.shape) == 4
        ), "Attention matrices must be of shape (layers, heads, tokens, tokens)"
        assert (
            attention_matrices.shape[-1] == attention_matrices.shape[-2]
        ), "Attention matrices must be square"

        self.prompt = prompt
        self.tokenized_prompt = tokenized_prompt
        self.attention_matrices = attention_matrices
        self.logits = logits

    def get_attn_matrix(self, layer: int, head: int) -> torch.tensor:
        return self.attention_matrices[layer, head]

@torch.inference_mode()
def get_attention_matrices(
    input: str | TokenizerOutput,
    mt: ModelandTokenizer,
    value_weighted: bool = False,
    patches: Optional[PatchSpec | list[PatchSpec]] = None
) -> torch.tensor:
    """
    Parameters:
        prompt: str, input prompt
        mt: ModelandTokenizer, model and tokenzier
        value_weighted: bool
            - False: Returns attention masks for each key-value pair (after softmax).
                This is the attention mask actually produced inside the model.
            - True: Considers the value matrices to give a sense of the actual
                contribution of source tokens to the target token residual.
    Returns:
        attention matrices: torch.tensor of shape (layers, heads, tokens, tokens)
    """
    if isinstance(input, str):
        input = prepare_input(prompts=input, tokenizer=mt)
    else:
        assert isinstance(
            input, TokenizerOutput
        ), "Input must be either a string or a TokenizerOutput object."

    # TODO: add patches logic

    with mt.trace(input, output_attentions=True):
        output = mt.model.output.save()
        logits = mt.output.logits[0][-1].save()

    printer(output.keys())
    printer(logits.shape)

    # TODO: Uncomment this to optimize for NVIDIA GPUs
    #output.attentions = [attn.cuda() for attn in output.attentions]

    attentions = torch.vstack(output.attentions) # (layers, heads, tokens, tokens)

    # TODO: Implement value_weighted logic

    return AttentionInformation(
        prompt=input,
        tokenized_prompt=[mt.tokenizer.decode(tok) for tok in input.input_ids[0]],
        attention_matrices=attentions.detach().cpu().to(torch.float32).numpy(),
        logits=logits.detach().cpu(),
    )