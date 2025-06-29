import torch
import pandas as pd
from typing import Tuple
from nnsight import LanguageModel
from src.models import ModelandTokenizer

def token_id(mt: ModelandTokenizer, word: str) -> int:
    """Return ID of *first* token of 'word' (good enough for single-token words)."""
    return mt.tokenizer.encode(word, add_special_tokens=False)[0]

@torch.no_grad()
def logit_attribution(
    mt: ModelandTokenizer,
    prompt: str,
    target: str,
    max_layers: int | None = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Run prompt once, capture every attn-head & mlp output.
    Compute each's direct logit contribution to the target token.

    Returns:
        A tuple containing:
        - A pandas dataframe with columns:
            layer, kind ('head'|'mlp'), index (head-idx or None), contribution (float), pct
        - The total logit of the target token
    """
    tgt_id = token_id(mt, target)

    config = mt.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    
    saved_proxies = []

    with mt.trace(prompt) as t:
        # save embedding contribution
        embed_proxy = mt.model.embed_tokens.output.save()
        saved_proxies.append({'layer': -1, 'kind': 'embed', 'proxy': embed_proxy})
        
        # save every attention head & mlp output's contribution
        for layer, block in enumerate(mt.model.layers):
            if max_layers and layer >= max_layers: break

            # to get per-head attribution, we save the input to o_proj 
            o_proj_input_proxy = block.self_attn.o_proj.input.save()
            saved_proxies.append({
                'layer': layer,
                'kind': 'head',
                'proxy': o_proj_input_proxy,
                'block': block
            })

            # save mlp output
            mlp_proxy = block.mlp.output.save()
            saved_proxies.append({
                'layer': layer,
                'kind': 'mlp',
                'proxy': mlp_proxy
            })

        final_logits_proxy = mt.lm_head.output.save()

    # The direct logit attribution of a component is its output vector
    ## projected by the unembedding matrix
    unembed_matrix = mt.lm_head.weight

    final_logits = final_logits_proxy[0, -1, :]
    total_logit = final_logits[tgt_id].item()

    rows = []
    for p in saved_proxies:
        kind = p['kind']
        
        # We're interested in the last token's logit, so we take the state at
        ## sequence position -1. 
        # The proxy value is a tuple, so we access the first element.
        hidden_state = p['proxy'].value[0, -1, :]

        if kind == 'head':
            # This is the concatenated output of all heads.
            # We need to split it and apply the corresponding part of the
            ## o_proj weight matrix.
            # h_per_head has shape [num_heads, head_dim]
            h_per_head = hidden_state.view(num_heads, head_dim)

            # W_O_heads is a list of (hidden_size, head_dim) tensors
            W_O = p['block'].self_attn.o_proj.weight
            W_O_heads = W_O.chunk(num_heads, dim=1)

            for i in range(num_heads):
                # head_contribution is shape [hidden_size]
                head_contribution = h_per_head[i] @ W_O_heads[i].T
                head_contribution = head_contribution.to(unembed_matrix.device)
                logit_contribution = head_contribution @ unembed_matrix.T
                contribution = logit_contribution[tgt_id].item()
                pct_contribution = (contribution / total_logit) * 100
                rows.append({
                    'layer': p['layer'],
                    'kind': 'head',
                    'index': i,
                    'contribution': logit_contribution[tgt_id].item(),
                    'pct': f"{pct_contribution:.6f}%"
                })

        else: #mlp or embed
            hidden_state = hidden_state.to(unembed_matrix.device)
            logit_contribution = hidden_state @ unembed_matrix.T
            contribution = logit_contribution[tgt_id].item()
            pct_contribution = (contribution / total_logit) * 100
            rows.append({
                'layer': p['layer'],
                'kind': kind,
                'index': None,
                'contribution': contribution,
                'pct': f"{pct_contribution:.6f}%"
            })

    df = pd.DataFrame(rows)
    return df, total_logit