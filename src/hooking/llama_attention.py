import torch
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AttentionEdge:
    q_idx: int
    k_idx: int

#def LlamaAttentionPatcher(
#    block_name: Optional[str] = None,
#    cut_attn_edges: Optional[dict[int, list[AttentionEdge]]] = None,
#    save_attn_for: Optional[list[int]] = None,
#    store_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
#    freeze_attn_matrices: Optional[dict[int, torch.Tensor]] = None,
#    value_weighted: bool = False,
#    store_head_contributions: Optional[dict[int, torch.Tensor]] = None,
#    freeze_attn_contributions: Optional[dict[int, torch.Tensor]] = None,
#) -> callable:
#    """
#    Wraps the forward method of the `LlamaSdpaAttention` class
#    Provides extra arguments for intervention and grabbing attention weights for visualization
#
#    Args:
#        block_name: name of the block (mainly for logging and debugging)
#        cut_attn_edges: [head_idx, [AttentionEdge(q_idx, k_idx)]] to cut off attention edge
#            q_idx --> k_idx via a specific head
#        save_attn_weights: list of head indices to save attention weights for visualization
#        attn_matrices: [head_idx, attn_matrix] to store the attention matrix for a specific head
#    """
#
#    if save_attn_for is not None:
