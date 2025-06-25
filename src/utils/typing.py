import numpy
import torch
import transformers
from typing import Optional
from nnsight import LanguageModel
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

ArrayLike = list | tuple | numpy.ndarray | torch.Tensor

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