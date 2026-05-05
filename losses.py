"""
Loss computation utilities for contrastive learning models.

This module provides shared loss computation methods for document embedding models,
reducing code duplication across UniDEC, SupConDR, and LLM implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLossMixin:
    """
    Mixin class providing contrastive loss computation methods.

    This mixin can be inherited by any nn.Module that needs contrastive loss
    computation. The inheriting class should have a `bce_loss` attribute if
    using the 'bce' loss criterion.
    """

    bce_loss: Optional[nn.BCEWithLogitsLoss] = None

    def compute_loss(
        self,
        sim: torch.Tensor,
        target: torch.Tensor,
        loss_crit: str
    ) -> torch.Tensor:
        """
        Compute loss based on similarity scores and targets.

        Args:
            sim: Similarity scores tensor of shape (batch_size, num_labels)
            target: Target labels tensor of shape (batch_size, num_labels)
            loss_crit: Loss criterion - one of 'balanced-supcon', 'decoupled-supcon',
                      'supcon', or 'bce'

        Returns:
            Computed loss value as a scalar tensor

        Raises:
            ValueError: If loss criterion is not implemented
        """
        num_pos = target.sum(-1)

        if loss_crit == 'balanced-supcon':
            return self._balanced_supcon_loss(sim, target, num_pos)
        elif loss_crit == 'decoupled-supcon':
            return self._decoupled_supcon_loss(sim, target, num_pos)
        elif loss_crit == 'supcon':
            return self._supcon_loss(sim, target, num_pos)
        elif loss_crit == 'bce':
            if self.bce_loss is None:
                raise ValueError("bce_loss not initialized")
            return self.bce_loss(sim, target)
        else:
            raise ValueError(f"{loss_crit} loss not implemented")

    def _balanced_supcon_loss(
        self,
        sim: torch.Tensor,
        target: torch.Tensor,
        num_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute balanced supervised contrastive loss."""
        target_mask_pos = target / num_pos[..., None]
        denom_mask = torch.where(target_mask_pos == 0., 1., target_mask_pos)

        denom_val = (torch.exp(sim) * denom_mask).sum(-1)
        log_probs = sim - torch.log(denom_val)[..., None]

        soft_logits = (target * log_probs).sum(dim=-1)
        return -(soft_logits / num_pos).mean()

    def _decoupled_supcon_loss(
        self,
        sim: torch.Tensor,
        target: torch.Tensor,
        num_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute decoupled supervised contrastive loss."""
        pos_rows, pos_cols = torch.nonzero(target, as_tuple=True)
        sim_pos = sim[pos_rows, pos_cols]

        denom_sim = sim.clone()
        denom_sim[pos_rows, pos_cols] = -100.0
        log_denom = denom_sim.logsumexp(1)
        log_denom = torch.logaddexp(log_denom[pos_rows], sim_pos)

        log_prob = sim_pos - log_denom
        return -(log_prob / num_pos[pos_rows]).sum() / sim.shape[0]

    def _supcon_loss(
        self,
        sim: torch.Tensor,
        target: torch.Tensor,
        num_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard supervised contrastive loss."""
        exp_sim = F.log_softmax(sim, dim=-1)
        soft_logits = (target * exp_sim).sum(dim=-1)
        return -(soft_logits / num_pos).mean()
