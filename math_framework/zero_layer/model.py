"""
Zero-layer Transformer.

token -> embed -> unembed -> logits
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class ZLTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 0
    n_embd: int = 768
    dropout: float = 0.0

class ZeroLayerTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(
                config.vocab_size, config.n_embd
            )
        ))
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size,
            bias=False
        )

        # Init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.2
            )

    def forward(self, idx, targets=None):
        """
        When forward gets used:
            - Training step: Call the model with `idx` (input tokens) and usually
            `targets` (the same sequence shifted left by one). `forward` returns
            logits and optionally a loss. The call `model(idx, targets)` triggers
            `forward`.
            - Generation/inference: Call the model with the current context `idx`
            (often cropped to the last `block_size` tokens) to get logits for the
            next token, sample/argmax it, append, and repeat. The call `model(idx)`
            triggers `forward`.

        idx: token indices: LongTensor of shape (B, T) where:
            - B: batch size
            - T: sequence length (context length)
            - Each entry is an integer token ID [0, vocab_size],
            produced by the tokenizer. It's the context you feed the model.
        """
        b, t = idx.size()
        assert t <= self.config.block_size

        # token embeddings: (b, t, n_embed)
        tok_emb = self.transformer.wte(idx)

        # project to vocab: (b, t, vocab_size)
        logits = self.lm_head(tok_emb)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(b*t, -1),
                targets.view(b*t)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditionaing sequence of indices idx (LongTensor shape (b,t))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time.
        Most likely you'll want to make sure to be in model.eval()
        mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must
            # crop it at block_size
            if idx.size(1) <= self.config.block_size:
                idx_cond = idx
            else:
                idx_cond = idx[:, -self.config.block_size:]

            # forward the model to get the logits for
            # the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            if temperature <= 0:
                temperature = 1e6
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(
                    logits, min(top_k, logits.size(-1))
                )
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # apend sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
            
    
if __name__ == "__main__":
    cfg = ZLTConfig()
    zlt = ZeroLayerTransformer(cfg)
    print(zlt)
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    # shape check
    # targets should be shifted in real training
    logits, loss = zlt(x, x)
    print(logits.shape, loss)