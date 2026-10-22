"""Whole-universe contrastive denominators, evaluated in query chunks."""
from __future__ import annotations


def residual_contrastive_loss(queries, keys, temperature, chunk_size, *, squared=False, checkpointing=False):
    """Aligned diagonal positives; temporal squared logits retain positive direction.

    Every query sees ALL keys, including its positive. Chunking does not create
    independent negative pools. Returns the mean over query stocks.
    """
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    if queries.shape != keys.shape or len(queries) < 2:
        raise ValueError("contrastive representations must align and contain at least two stocks")
    queries = functional.normalize(queries, dim=-1, eps=1e-8)
    keys = functional.normalize(keys, dim=-1, eps=1e-8)

    def block(q, k, positive_rows):
        cosine = q @ k.T
        positive = cosine.gather(1, positive_rows[:, None]).squeeze(1)
        logits = cosine.square() if squared else cosine
        numerator = positive.sign() * positive.square() if squared else positive
        return (torch.logsumexp(logits / temperature, dim=1) - numerator / temperature).sum()

    total = queries.new_zeros(())
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        positive_rows = torch.arange(start, stop, device=queries.device)
        if checkpointing and torch.is_grad_enabled():
            loss = checkpoint(block, queries[start:stop], keys, positive_rows)
        else:
            loss = block(queries[start:stop], keys, positive_rows)
        total = total + loss
    return total / len(queries)
