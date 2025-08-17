from __future__ import annotations

from typing import Optional, Dict
import json
import torch
import torch.nn as nn


class EntityExtractionLoss(nn.Module):
    """
    Custom loss function for entity extraction task.
    Designed specifically for calendar event entity extraction.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.1, delta: float = 0.05):
        super().__init__()
        self.alpha = alpha  # Base CE loss weight (handled by Trainer)
        self.beta = beta    # JSON format compliance penalty
        self.gamma = gamma  # Entity field accuracy penalty
        self.delta = delta  # Confidence regularization
        
        # Expected entity fields for validation
        self.expected_fields = {"action", "date", "time", "attendees", "location", "duration", "recurrence", "notes"}
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, aux: Optional[Dict] = None) -> torch.Tensor:
        """
        Custom loss for entity extraction:
        1. JSON format compliance - ensure valid JSON structure
        2. Entity field completeness - penalize missing required fields
        3. Confidence regularization - encourage decisive predictions
        """
        device = logits.device
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Only compute loss on unmasked tokens (completion part)
        unmasked_positions = (labels != -100)
        
        if not unmasked_positions.any():
            return loss
        
        # 1. JSON Format Compliance Penalty (β)
        if self.beta > 0.0 and aux is not None:
            json_validity = aux.get("json_validity", 1.0)  # 1.0 = valid, 0.0 = invalid
            json_penalty = self.beta * (1.0 - float(json_validity))
            loss = loss + json_penalty
        
        # 2. Entity Field Accuracy Penalty (γ)
        if self.gamma > 0.0 and aux is not None:
            field_accuracy = aux.get("field_accuracy", 1.0)  # Ratio of correct fields
            field_penalty = self.gamma * (1.0 - float(field_accuracy))
            loss = loss + field_penalty
        
        # 3. Confidence Regularization (δ)
        if self.delta > 0.0:
            # Encourage confident predictions on completion tokens
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            
            # Apply only to unmasked positions
            masked_confidence = max_probs * unmasked_positions.float()
            avg_confidence = masked_confidence.sum() / (unmasked_positions.sum().float() + 1e-8)
            
            # Penalty for low confidence
            confidence_penalty = self.delta * (1.0 - avg_confidence)
            loss = loss + confidence_penalty
        
        return loss


class EntityMetrics:
    """Helper class to compute entity extraction metrics."""
    
    @staticmethod
    def compute_field_accuracy(predicted: Dict, target: Dict) -> float:
        """Compute field-level accuracy between predicted and target entities."""
        if not isinstance(predicted, dict) or not isinstance(target, dict):
            return 0.0
        
        expected_fields = {"action", "date", "time", "attendees", "location", "duration", "recurrence", "notes"}
        correct_fields = 0
        total_fields = len(expected_fields)
        
        for field in expected_fields:
            pred_val = predicted.get(field)
            target_val = target.get(field)
            
            # Exact match for field values
            if pred_val == target_val:
                correct_fields += 1
        
        return correct_fields / total_fields
    
    @staticmethod
    def is_valid_json(text: str) -> bool:
        """Check if text contains valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict]:
        """Extract JSON object from text."""
        try:
            # Find first { and matching }
            start = text.find('{')
            if start == -1:
                return None
            
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = text[start:i+1]
                        return json.loads(json_str)
            return None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
