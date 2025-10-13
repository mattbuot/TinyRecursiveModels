import torch
from enum import Enum
from einops import rearrange

class AggregationStrategy(Enum):
    IWRM_WITH_Q_HALT_IN = "iwrm_with_q_halt_in"
    IWRM_WITH_Q_HALT_OUT = "iwrm_with_q_halt_out"
    PIXELWISE_Q_HALT_IN = "pixelwise_q_halt_in"
    PIXELWISE_Q_HALT_OUT = "pixelwise_q_halt_out"
    IWRM_PIXELWISE_Q_HALT_IN = "iwrm_pixelwise_q_halt_in"
    IWRM_PIXELWISE_Q_HALT_OUT = "iwrm_pixelwise_q_halt_out"
    LM_LOSS_VS_Q_HALT_LOSS = "lm_loss_vs_q_halt_loss"
    SUM = "sum"


def aggregate_losses(
    lm_loss: torch.Tensor,
    q_halt_loss: torch.Tensor,
    aggregation_strategy: AggregationStrategy,
    n_groups: int | None = None,
) -> list[torch.Tensor]:
    
    match aggregation_strategy:
        case AggregationStrategy.IWRM_WITH_Q_HALT_IN:
            lm_loss = lm_loss.sum(dim=1)
            losses = lm_loss + q_halt_loss

        case AggregationStrategy.IWRM_WITH_Q_HALT_OUT:
            lm_loss = lm_loss.sum(dim=1)
            losses = torch.cat([lm_loss, q_halt_loss], dim=0)

        case AggregationStrategy.PIXELWISE_Q_HALT_IN:
            lm_loss = lm_loss.sum(dim=0)
            q_halt_loss = q_halt_loss.sum() / lm_loss.shape[0]
            losses = lm_loss + q_halt_loss
            
        case AggregationStrategy.PIXELWISE_Q_HALT_OUT:
            lm_loss = lm_loss.sum(dim=0)
            losses = torch.cat([lm_loss, q_halt_loss], dim=0)

        case AggregationStrategy.IWRM_PIXELWISE_Q_HALT_IN:
            lm_loss = rearrange(lm_loss, "b p -> (b p)")
            q_halt_loss = q_halt_loss.sum() / lm_loss.shape[0]
            losses = lm_loss + q_halt_loss

        case AggregationStrategy.IWRM_PIXELWISE_Q_HALT_OUT:
            lm_loss = rearrange(lm_loss, "b p -> (b p)")
            losses = torch.cat([lm_loss, q_halt_loss], dim=0)

        case AggregationStrategy.LM_LOSS_VS_Q_HALT_LOSS:
            losses = torch.stack([lm_loss.sum(), q_halt_loss.sum()])
        case AggregationStrategy.SUM:
            losses = torch.stack([lm_loss.sum() + q_halt_loss.sum()])
        case _:
            raise ValueError(f"Invalid aggregation strategy: {aggregation_strategy}")

    if n_groups is not None:
        # Randomly group losses into n groups
        n_losses = losses.shape[0]
        assert n_losses % n_groups == 0, f"Number of losses must be divisible by n_groups: {n_losses} % {n_groups} != 0"

        losses = losses[torch.randperm(n_losses)]
        losses = rearrange(losses, "(k n_groups) -> n_groups k", n_groups=n_groups).sum(dim=1)
        
    return list(losses.unbind(dim=0))
