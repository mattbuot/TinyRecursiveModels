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
    STACK_INTERNAL_LOSSES = "stack_internal_losses"
    STACK_INTERNAL_LOSSES_ONLY = "stack_internal_losses_only"
    STACK_SUPERVISIONS = "stack_supervisions"
    STACK_SUPERVISIONS_AND_SUM = "stack_supervisions_and_sum"
    STACK_SUPERVISIONS_AND_IWRM = "stack_supervisions_and_iwrm"
    STACK_SUPERVISIONS_AND_PIXELWISE = "stack_supervisions_and_pixelwise"
    STACK_SUPERVISIONS_AND_IWRM_PIXELWISE = "stack_supervisions_and_iwrm_pixelwise"
    STACK_INTERNAL_AND_SUM = "stack_internal_and_sum"

    def is_stack_supervisions(self) -> bool:
        return self in (AggregationStrategy.STACK_SUPERVISIONS,
                        AggregationStrategy.STACK_SUPERVISIONS_AND_IWRM,
                        AggregationStrategy.STACK_SUPERVISIONS_AND_SUM,
                        AggregationStrategy.STACK_SUPERVISIONS_AND_PIXELWISE,
                        AggregationStrategy.STACK_SUPERVISIONS_AND_IWRM_PIXELWISE)


def aggregate_losses(
    lm_loss: torch.Tensor | list[torch.Tensor],
    q_halt_loss: torch.Tensor | None,
    internal_lm_losses: list[torch.Tensor] | None,
    aggregation_strategy: AggregationStrategy,
    intermediate_loss_weight: float,
    n_groups: int | None = None,
) -> list[torch.Tensor]:

    if isinstance(lm_loss, list) and not aggregation_strategy.is_stack_supervisions():
        # should not happen
        raise ValueError(f"lm_loss is a list but aggregation strategy is not stack_supervisions: {aggregation_strategy}")
        # lm_loss = torch.stack(lm_loss).sum()
    
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

        case AggregationStrategy.STACK_INTERNAL_LOSSES:
            losses = torch.stack([lm_loss] + [intermediate_loss_weight * loss for loss in internal_lm_losses])
            lm_loss = losses.sum(dim=(1, 2))
            q_halt_loss = q_halt_loss.sum() / lm_loss.shape[0]
            losses = lm_loss + q_halt_loss

        case AggregationStrategy.STACK_INTERNAL_LOSSES_ONLY:
            losses = torch.stack([intermediate_loss_weight * loss for loss in internal_lm_losses])
            q_halt_loss = q_halt_loss.sum() / losses.shape[0]
            losses = losses.sum(dim=(1, 2)) + q_halt_loss

        case AggregationStrategy.SUM:
            losses = torch.stack([lm_loss.sum() + q_halt_loss.sum()])

        case AggregationStrategy.STACK_INTERNAL_AND_SUM:
            losses = torch.stack([lm_loss.sum() + q_halt_loss.sum() + intermediate_loss_weight * sum(internal_lm_losses)])

        case AggregationStrategy.STACK_SUPERVISIONS:
            assert isinstance(lm_loss, list), "lm_loss must be a list"
            losses = torch.stack([l.sum() for l in lm_loss])

        case AggregationStrategy.STACK_SUPERVISIONS_AND_IWRM:
            assert isinstance(lm_loss, list), "lm_loss must be a list"
            losses = torch.stack([l.sum(dim=1) for l in lm_loss]).flatten()

        case AggregationStrategy.STACK_SUPERVISIONS_AND_SUM:
            assert isinstance(lm_loss, list), "lm_loss must be a list"
            losses = torch.stack([sum([l.sum() for l in lm_loss])])

        case AggregationStrategy.STACK_SUPERVISIONS_AND_PIXELWISE:
            assert isinstance(lm_loss, list), "lm_loss must be a list"
            losses = torch.stack([l.sum(dim=0) for l in lm_loss]).flatten()

        case AggregationStrategy.STACK_SUPERVISIONS_AND_IWRM_PIXELWISE:
            assert isinstance(lm_loss, list), "lm_loss must be a list"
            losses = torch.stack(lm_loss).flatten()

        case _:
            raise ValueError(f"Invalid aggregation strategy: {aggregation_strategy}")

    if n_groups is not None:
        # Randomly group losses into n groups
        n_losses = losses.shape[0]
        assert n_losses % n_groups == 0, f"Number of losses must be divisible by n_groups: {n_losses} % {n_groups} != 0"

        losses = losses[torch.randperm(n_losses)]
        losses = rearrange(losses, "(k n_groups) -> n_groups k", n_groups=n_groups).sum(dim=1)
        
    return list(losses.unbind(dim=0))
