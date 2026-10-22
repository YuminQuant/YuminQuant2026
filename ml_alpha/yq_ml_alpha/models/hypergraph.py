"""Unit-edge-weight, symmetric hypergraph propagation without an N x N matrix."""
from __future__ import annotations


def propagate_hypergraph(x, incidence):
    """D_v^-1/2 H D_e^-1 H.T D_v^-1/2 X; tuple incidence stores binary edges."""
    if isinstance(incidence, tuple):
        stocks, edges, count = incidence
        if count == 0 or stocks.numel() == 0:
            return x * 0
        ones = x.new_ones(stocks.numel())
        node_degree = x.new_zeros(len(x)).index_add(0, stocks, ones)
        edge_degree = x.new_zeros(count).index_add(0, edges, ones)
        inv_node = node_degree.clamp_min(1).rsqrt()
        messages = x.new_zeros((count, x.shape[1])).index_add(
            0, edges, (x * inv_node[:, None])[stocks])
        messages = messages / edge_degree.clamp_min(1)[:, None]
        return x.new_zeros(x.shape).index_add(0, stocks, messages[edges]) * inv_node[:, None]
    node_degree = incidence.sum(dim=1).clamp_min(1e-8)
    edge_degree = incidence.sum(dim=0).clamp_min(1e-8)
    inv_node = node_degree.rsqrt()
    messages = incidence.T @ (x * inv_node[:, None])
    return (incidence @ (messages / edge_degree[:, None])) * inv_node[:, None]


def subset_binary_incidence(incidence, selected, node_count):
    """Take original incidence rows in selected order; do not infer new relations."""
    stocks, edges, count = incidence
    mapping = stocks.new_full((node_count,), -1)
    import torch

    mapping[selected] = torch.arange(len(selected), device=selected.device)
    local = mapping[stocks]
    keep = local >= 0
    return local[keep], edges[keep], count
