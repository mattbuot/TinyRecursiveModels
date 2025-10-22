import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).contiguous().view(-1, logits.shape[-1]), labels.to(torch.long).contiguous().view(-1), ignore_index=ignore_index, reduction="none").contiguous().view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, differential_loss: bool):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.differential_loss = differential_loss
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, list[torch.Tensor], Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        previous_logits = model_kwargs["carry"].inner_carry.output_logits
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        logits = outputs["logits"][-1]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(logits, dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "exact_accuracy_flatten": (valid_metrics & seq_is_correct),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        internal_lm_losses = [self.loss_fn(outputs["logits"][i+1] - outputs["logits"][i].detach(), labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) for i in range(len(outputs["logits"])-1)]
        loss_input = logits - previous_logits if self.differential_loss else logits
        lm_loss = self.loss_fn(loss_input, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)# - self.loss_fn(previous_logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        
        #lm_loss = self.loss_fn(logits - previous_logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)# - self.loss_fn(previous_logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
        #print(f"LM Loss shape: {lm_loss.shape}")
        lm_loss = (lm_loss / loss_divisor)

        #print(f"Q Halt Loss shape: {F.binary_cross_entropy_with_logits(outputs['q_halt_logits'], seq_is_correct.to(outputs['q_halt_logits'].dtype), reduction='none').shape}")
        
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="none")
        metrics.update({
            "lm_loss": lm_loss.sum().detach(),
            "q_halt_loss": q_halt_loss.sum().detach(),
        })

        losses = [lm_loss, 0.5 * q_halt_loss, internal_lm_losses]

        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:

            #print(f"Q Continue Loss shape: {F.binary_cross_entropy_with_logits(outputs['q_continue_logits'], outputs['target_q_continue'], reduction='none').shape}")
            
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="none")

            metrics["q_continue_loss"] = q_continue_loss.sum().detach()
            

            losses.append(0.5 * q_continue_loss)
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, losses, metrics, detached_outputs, new_carry.halted.all()
