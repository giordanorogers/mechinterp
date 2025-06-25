"""
File for loading and handling models and tokenizers.
"""
import torch
import logging
from typing import Optional
from nnsight import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.typing import Model

logger = logging.getLogger(__name__)

class ModelandTokenizer(LanguageModel):
    """
    A model and its tokenizer.
    """
    def __init__(
        self,
        model_key: Optional[str] = "gpt2",
        device_map: str = "auto",
        abs_path: bool = False
    ) -> None:
        # Load model and tokenizer first
        if abs_path:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_key,
                    local_files_only=True,
                    device_map=device_map
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_key
                )
            except Exception as e:
                logger.error(
                    f"Error loading model: {e}"
                )
                raise
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_key,
                device_map=device_map
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_key
            )
            self.name = model_key
        
        # Initialize the parent LanguageModel class with the loaded model
        super().__init__(model)
        
        # Store the class variables
        self._model.eval() # Set the model to evaluation mode
        self.tokenizer = tokenizer

        # Padding token configuration
        self.tokenizer.pad_token = self.tokenizer.eos_token # For generation
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if (
            hasattr(self.tokenizer, "bos_token") is False
            or self.tokenizer.bos_token is None
        ):
            self.tokenizer.bos_token = self.tokenizer.eos_token
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self._model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        logger.info(
            f"Loaded {model_key}"
        )

def unwrap_model(
    net: ModelandTokenizer | LanguageModel | torch.nn.Module
) -> torch.nn.Module:
    if isinstance(net, LanguageModel):
        return net._model
    if isinstance(net, torch.nn.Module):
        return net
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")

def unwrap_tokenizer(mt: ModelandTokenizer | AutoTokenizer) -> AutoTokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt

def any_parameter(model: ModelandTokenizer | Model) -> torch.nn.Parameter | None:
    """ Get any example parameter for the model. """
    model = unwrap_model(model)
    return next(iter(model.parameters()), None)

def determine_device(model: ModelandTokenizer | Model) -> torch.device | None:
    """ Determines device model is running on. """
    parameter = any_parameter(model)
    return parameter.device if parameter is not None else None

        

if __name__ == "__main__":
    # Load the model and tokenizer, just pass in the model name
    mt = ModelandTokenizer("gpt2")
    print(mt._model)
    ## Under hood:
        ## Check if path exists locally
            ## If so, load the local model and get the corresponding tokenizer from HF
            ## If not, load the model and tokenizer from HF