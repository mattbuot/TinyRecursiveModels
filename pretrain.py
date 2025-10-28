import copy
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import coolname
import hydra
import numpy as np
import pydantic
import torch
import torch.distributed as dist
import tqdm
import yaml
from adam_atan2 import AdamATan2
from omegaconf import DictConfig
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from torchjd import backward
from torchjd.aggregation import UPGrad

import wandb
from models.ema import EMAHelper
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import get_model_source_path, load_model_class
from utils.torchjd_utils import AggregationStrategy, aggregate_losses

global_step = 0


def get_gpu_info():
    """Get GPU count and GPU type information."""
    if not torch.cuda.is_available():
        return {"gpu_count": 0, "gpu_type": "none"}
    
    gpu_count = torch.cuda.device_count()
    gpu_type = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
    
    return {"gpu_count": gpu_count, "gpu_type": gpu_type}


def print_grammian(_, inputs, __):
    if not dist.is_initialized() or dist.get_rank() == 0:
        #print(inputs[0])
        wandb.log({"grammian_min": inputs[0].min(), "grammian_mean": inputs[0].mean(), "grammian_median": inputs[0].median()},
        step=global_step)

def log_gd_similarity(_, inputs: tuple[torch.Tensor, ...], aggregation: torch.Tensor) -> None:
    """Prints the cosine similarity between the aggregation and the average gradient."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        matrix = inputs[0]
        gd_output = matrix.mean(dim=0)
        similarity = cosine_similarity(aggregation, gd_output, dim=0)
        wandb.log({"gd_similarity": similarity.item()}, step=global_step)


aggregator = UPGrad()
aggregator.weighting.weighting.register_forward_hook(print_grammian)
aggregator.register_forward_hook(log_gd_similarity)

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None #"checkpoints/downloaded/Sanjin2024_TinyRecursiveModels-ARC-AGI-2/step_217602"
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    run_evaluator_only_at_end: bool = False
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # TorchJD
    grad_aggregation: AggregationStrategy = AggregationStrategy.SUM
    grad_n_groups: int | None = None
    differential_loss: bool = False
    intermediate_loss_weight: float = 1
    
    # Dropout
    dropout: float = 0.1
    
    # Wandb
    no_wandb: bool = False
    in_sweep: bool = False

    # Custom sampling
    custom_sampling: bool = True
    
    # GPU info (added dynamically)
    gpu_count: int = 0
    gpu_type: str = "unknown"

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, puzzle_weights: dict[int, float] | None = None, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split, puzzle_weights=puzzle_weights)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,  # Non-autoregressive
        dropout=config.dropout
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Handle _orig_mod prefix mismatch between checkpoint and compiled model
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Check if model expects _orig_mod prefix but checkpoint doesn't have it
        if any(key.startswith("_orig_mod.") for key in model_keys) and not any(key.startswith("_orig_mod.") for key in checkpoint_keys):
            print("Adding _orig_mod prefix to state_dict keys for compiled model")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"_orig_mod.{key}"
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        # Check if checkpoint has _orig_mod prefix but model doesn't expect it
        elif any(key.startswith("_orig_mod.") for key in checkpoint_keys) and not any(key.startswith("_orig_mod.") for key in model_keys):
            print("Removing _orig_mod prefix from state_dict keys")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("_orig_mod."):
                    new_key = key[len("_orig_mod."):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "model.inner.puzzle_emb.weights"
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )

        output_logits_key = "model.inner.output_logits_init"
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            output_logits_key = "_orig_mod.model.inner.output_logits_init"
            
        if output_logits_key not in state_dict:
            state_dict[output_logits_key] = model.model.inner.init_output_logits()
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int) -> tuple[dict[str, float] | None, dict[int, tuple[int, int]] | None]:
    global global_step
    
    train_state.step += 1
    global_step = train_state.step
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, losses, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    if hasattr(train_state.model.model.config, "no_act") and train_state.model.model.config.no_act:

        assert config.grad_aggregation == AggregationStrategy.STACK_SUPERVISIONS, "Only STACK_SUPERVISIONS is supported for no_act"
        all_finish = False
        losses = []
        iterations = 0
        iteration_metrics = {}
        while not all_finish:
            train_state.carry, loss_list, metrics, preds, all_finish = train_state.model(
                carry=train_state.carry, batch=batch, return_keys=[]
            )
            # losses are the concatenation of lm_loss only
            iteration_metrics[f"lm_loss_{iterations}"] = loss_list[0].sum().detach()
            losses.append(loss_list[0].sum())
            iterations += 1
            
            assert iterations <= config.arch.halt_max_steps, "Max iterations reached"
        metrics.update(iteration_metrics)
    else:
        # Forward
        train_state.carry, losses, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

        lm_loss, q_halt_loss, internal_lm_losses = losses
        losses = aggregate_losses(lm_loss, q_halt_loss, internal_lm_losses, config.grad_aggregation, n_groups=config.grad_n_groups, intermediate_loss_weight=config.intermediate_loss_weight)

    if len(losses) == 1:
        loss = losses[0]
        ((1 / global_batch_size) * loss).backward()
    else:
        backward([(1 / global_batch_size) * loss for loss in losses], aggregator, parallel_chunk_size=1)

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Compute and store gradient norms
    if rank == 0 and not config.no_wandb:
        for name, param in train_state.model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                wandb.log({f"train/grad_norm_{name}": grad_norm.item()}, step=global_step)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = [m for m in sorted(metrics.keys()) if not m.endswith("_flatten")]  # Sort keys to guarantee all processes use the same order.
        metric_keys_flatten = [m for m in sorted(metrics.keys()) if m.endswith("_flatten")]
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        metric_values_flatten = torch.stack([metrics[k] for k in metric_keys_flatten])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
            dist.reduce(metric_values_flatten, dst=0)
        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            if config.custom_sampling:
                metric_values_flatten = metric_values_flatten.cpu()
                reduced_metrics_flatten = {k: metric_values_flatten[i] for i, k in enumerate(metric_keys_flatten)}
                exact_accuracy = reduced_metrics_flatten["exact_accuracy_flatten"]
                puzzle_ids = batch["puzzle_identifiers"].cpu()
                puzzle_id_to_counts = compute_exact_accuracy_per_puzzle(puzzle_ids, exact_accuracy)
            else:
                puzzle_id_to_counts = None

            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics, puzzle_id_to_counts

    return None, None

def compute_exact_accuracy_per_puzzle(puzzle_ids: torch.Tensor, exact_accuracy: torch.Tensor) -> dict[int, tuple[int, int]]:
    unique_ids, positions = torch.unique(puzzle_ids, sorted=True, return_inverse=True)

    puzzle_counts = torch.bincount(positions)
    valid_puzzle_counts = torch.bincount(positions, weights=exact_accuracy)

    result = dict(zip(unique_ids.tolist(), zip(puzzle_counts.tolist(), valid_puzzle_counts.tolist())))
    return result


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, losses, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, losses, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len([k for k in metrics.keys() if not k.endswith("_flatten")])), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys if not k.endswith("_flatten")])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate([k for k in metric_keys if not k.endswith("_flatten")])
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or (not config.no_wandb and wandb.run is None):
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code only if wandb is enabled
    if not config.no_wandb and wandb.run is not None:
        wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Add GPU information to config
        gpu_info = get_gpu_info()
        config.gpu_count = gpu_info["gpu_count"]
        config.gpu_type = gpu_info["gpu_type"]

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)


        if config.in_sweep:
            wandb.init(settings=wandb.Settings(_disable_stats=True))  # type: ignore
        
            config.data_paths = [f"data/{wandb.config.data}"]
            config.arch.H_cycles = wandb.config.H_cycles
            config.arch.L_cycles = wandb.config.L_cycles
            config.arch.halt_max_steps = wandb.config.halt_max_steps
            config.lr = wandb.config.lr

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)    

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        print(f"No evaluator found: {e}")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        if not config.no_wandb:
            if not config.in_sweep:
                wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
            wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    if config.custom_sampling:
        puzzle_id_counts = defaultdict(lambda: (0, 0))
    else:
        puzzle_id_counts = None

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")

           
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics, new_puzzle_id_counts = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                if not config.no_wandb:
                    wandb.log(metrics, step=train_state.step)

                if config.custom_sampling and new_puzzle_id_counts is not None:

                    for puzzle_id, (count, valid_count) in new_puzzle_id_counts.items():
                        puzzle_id_counts[puzzle_id] = (puzzle_id_counts[puzzle_id][0] + count, puzzle_id_counts[puzzle_id][1] + valid_count)

                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if RANK == 0 and config.custom_sampling and puzzle_id_counts:
            puzzle_weights = {}
            for puzzle_id, (total_count, correct_count) in puzzle_id_counts.items():
                accuracy = correct_count / (total_count + 1)
                weight = 1 - accuracy
                puzzle_weights[puzzle_id] = weight

            train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE, puzzle_weights=puzzle_weights)


        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            if not config.run_evaluator_only_at_end or _iter_id == total_iters - 1:
                train_state_eval.model.eval()
                metrics = evaluate(config, 
                    train_state_eval, 
                    eval_loader, 
                    eval_metadata, 
                    evaluators,
                    rank=RANK, 
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP)

                if RANK == 0 and metrics is not None:
                    if not config.no_wandb:
                        wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                print(f"SAVE CHECKPOINT to {config.checkpoint_path}")
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if not config.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    launch()
