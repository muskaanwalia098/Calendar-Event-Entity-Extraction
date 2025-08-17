from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn


class AuxiliaryStructuredLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.0, gamma: float = 0.0):
        super().__init__()
        self.alpha = alpha  # not used here; main CE is handled by Trainer
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, aux: Optional[Dict] = None) -> torch.Tensor:
        # labels contain -100 masked prompt tokens; compute a light penalty on shape only
        loss = torch.tensor(0.0, device=logits.device)
        if self.beta > 0.0 and aux is not None:
            # entity_accuracy_loss proxy: encourage lower perplexity on non-null tokens
            mask = (labels != -100).float()
            probs = torch.softmax(logits, dim=-1)
            # take max prob as confidence proxy
            conf = probs.max(dim=-1).values
            loss = loss + self.beta * (1.0 - (conf * mask).sum() / (mask.sum() + 1e-8))
        if self.gamma > 0.0 and aux is not None:
            # format penalty proxy (weak): penalize sequences that do not end with '}'
            ends_with_brace = aux.get("ends_with_brace", 1.0)
            loss = loss + self.gamma * (1.0 - float(ends_with_brace))
        return loss
