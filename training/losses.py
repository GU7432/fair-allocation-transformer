from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from eval_pipeline.utils.calculations import (
    calculate_agent_bundle_values_batch,
    nash_welfare_batch,
)
from eval_pipeline.utils.ef1_repair import ef1_quick_repair_batch
from fftransformer.helpers import get_nash_welfare


def sample_repair_best_nll_loss(
    valuations: torch.Tensor,
    assignment_distribution: torch.Tensor,
    num_samples: int = 8,
    ef1_repair_passes: int = 10,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Train toward the sampled allocation with the best repaired Nash welfare.

    Args:
        valuations: Valuation matrix batch of shape (B, n, m).
        assignment_distribution: Per-item agent probabilities of shape (B, m, n).
        num_samples: Number of candidate allocations sampled per valuation matrix.
        ef1_repair_passes: Maximum EF1 repair passes for each sampled allocation.
        eps: Minimum probability used for stable log-probabilities.

    Returns:
        A scalar NLL loss and logging/debug metrics.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if valuations.ndim != 3 or assignment_distribution.ndim != 3:
        raise ValueError("valuations and assignment_distribution must be rank-3 tensors")

    batch_size, n_agents, n_items = valuations.shape
    dist_batch, dist_items, dist_agents = assignment_distribution.shape
    if (dist_batch, dist_items, dist_agents) != (batch_size, n_items, n_agents):
        raise ValueError(
            "assignment_distribution must have shape (B, m, n) matching valuations (B, n, m)"
        )

    distribution = Categorical(probs=assignment_distribution)
    sampled_agents = distribution.sample((num_samples,)).permute(1, 0, 2).contiguous()

    candidate_allocations = F.one_hot(
        sampled_agents, num_classes=n_agents
    ).permute(0, 1, 3, 2).reshape(
        batch_size * num_samples, n_agents, n_items
    )

    with torch.no_grad():
        valuations_np = valuations.detach().cpu().numpy()
        allocations_np = candidate_allocations.detach().cpu().numpy().astype(np.int64)
        repeated_valuations_np = np.repeat(valuations_np, num_samples, axis=0)

        repaired_allocations = ef1_quick_repair_batch(
            allocations_np,
            repeated_valuations_np,
            max_passes=ef1_repair_passes,
        )
        bundle_values = calculate_agent_bundle_values_batch(
            repeated_valuations_np,
            repaired_allocations,
        )
        repaired_scores = nash_welfare_batch(bundle_values).reshape(
            batch_size, num_samples
        )
        best_sample_indices_np = np.argmax(repaired_scores, axis=1)

    best_sample_indices = torch.as_tensor(
        best_sample_indices_np,
        device=assignment_distribution.device,
        dtype=torch.long,
    )
    log_probs = torch.log(assignment_distribution.clamp_min(eps))
    sampled_log_probs = log_probs.unsqueeze(1).expand(
        batch_size, num_samples, n_items, n_agents
    ).gather(3, sampled_agents.unsqueeze(-1)).squeeze(-1)

    batch_indices = torch.arange(batch_size, device=assignment_distribution.device)
    best_log_probs = sampled_log_probs[batch_indices, best_sample_indices]
    loss = -(best_log_probs.sum(dim=1) / n_items).mean()

    with torch.no_grad():
        soft_nash_welfare = get_nash_welfare(
            valuations, assignment_distribution, reduction="mean"
        )
    best_repaired_scores = repaired_scores[
        np.arange(batch_size), best_sample_indices_np
    ]
    metrics = {
        "best_repaired_nash_welfare": float(best_repaired_scores.mean()),
        "mean_repaired_nash_welfare": float(repaired_scores.mean()),
        "soft_nash_welfare": float(soft_nash_welfare.detach().cpu().item()),
        "sampled_allocations_shape": tuple(candidate_allocations.shape),
    }

    return loss, metrics
