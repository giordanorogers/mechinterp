import numpy
import torch
import pathlib
import transformers
from typing import Optional, Literal, Sequence
from nnsight import LanguageModel
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path

# Throughout this codebase, we use HuggingFace model implementations.
Model = (
    LanguageModel
    | transformers.GPT2LMHeadModel
    | transformers.GPTJForCausalLM
    | transformers.GPTNeoXForCausalLM
    | transformers.LlamaForCausalLM
    | transformers.Gemma2ForCausalLM
    | transformers.GemmaForCausalLM
    | transformers.Qwen2ForCausalLM
    | transformers.Olmo2ForCausalLM
    | transformers.OlmoForCausalLM
    | transformers.Qwen3ForCausalLM
)

# PreTrainedTokenizerFast will be the real tokenizer if 'tokenizers' is installed.
# Otherwise it is a dummy object that raises helpful errors.
Tokenizer = transformers.PreTrainedTokenizerFast

# BatchEncoding holds tokenizer outputs (input_ids, attention_mask, etc.) as a dictionary-like object
# that provides utility methods for mapping between word/character positions and token indices,
# with automatic tensor conversion support.
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings
StrSequence = list[str] | tuple[str, ...]

@dataclass(frozen=False)
class PredictedToken(DataClassJsonMixin):
    """ A predicted token and its probability. """
    token: str
    prob: Optional[float] = None
    logit: Optional[float] = None
    token_id: Optional[int] = None
    metadata: Optional[dict] = None

    def __str__(self) -> str:
        rep = f'"{self.token}"[{self.token_id}]'
        add = []

        if self.prob is not None:
            add.append(f"p={self.prob:.3f}")

        if self.logit is not None:
            add.append(f"logit={self.logit:.3f}")

        if self.metadata is not None:
            for key, val in self.metadata.items():
                if key not in ["token", "token_id"] and val is not None:
                    add.append(f"{key}={val}")

        if len(add) > 0:
            rep += " (" + ", ".join(add) + ")"
        
        return rep