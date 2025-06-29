import torch
import logging
from src.tokens import TokenizerOutput
from src.functional import get_module_nnsight
from src.models import ModelandTokenizer

logger = logging.getLogger(__name__)

def module_output_has_extra_dim(mt, module_name):
    return (
        "mlp" not in module_name
        or module_name != mt.embedder_name
        or module_name != mt.lm_head_name
    )

def patch(
    h: torch.Tensor,
    mt: ModelandTokenizer,
    inp_layer: str,
    out_layer: str = "lm_head",
    context: TokenizerOutput | None = None,
    h_idx: int = 0,
    z_idx: int = -1,
) -> torch.Tensor:
    if context is None:
        context = mt.tokenizer(
            mt.tokenizer.bos_token, add_special_tokens=False, return_tensors="pt"
        )
        if h_idx != 0:
            logger.warning(
                "Context not provided. Using BOS token as context. Setting h_idx to 0."
            )
        h_idx = 0

    with mt.trace(context) as tr:
        inp_module = get_module_nnsight(mt, inp_layer)
        inp_state = (
            inp_module.output[0].save()
            if module_output_has_extra_dim(mt, inp_layer)
            else inp_module.output.save()
        )
        inp_state[:, h_idx, :] = h

        out_module = get_module_nnsight(mt, out_layer)
        out_state = (
            out_module.output[0].save()
            if module_output_has_extra_dim(mt, out_layer)
            else out_module.output.save()
        )

    if out_state.ndim == 2:
        out_state = out_state.unsqueeze(0)
    return out_state[:, z_idx].squeeze()