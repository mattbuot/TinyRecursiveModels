import os
import subprocess
import sys
from typing import Dict, Any

import wandb


def train() -> None:
    """Training function for wandb sweep that reads config and launches training."""
    
    nproc_per_node = 2
    
    # Construct torchrun command for distributed training
    cmd = [
        "torchrun",
        "--nproc-per-node", str(nproc_per_node),
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        "--nnodes=1",
        "pretrain.py",
        "arch=trm",
        "epochs=4000",
        "eval_interval=200",
        '+load_checkpoint="checkpoints/downloaded/Sanjin2024_TinyRecursiveModels-ARC-AGI-2/step_217602"',
        '+in_sweep=True',
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, timeout=60*60)
        print("Training completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        raise


def main() -> None:
    """Main function to define and run the sweep."""
    
    # Define sweep configuration
    sweep_config: Dict[str, Any] = {
        'method': 'bayes',
        'metric': {
            'name': 'test/accuracy',  # Will be logged by evaluators
            'goal': 'maximize'
        },
        'parameters': {
            'H_cycles': {
                'values': [3, 4]
            },
            'L_cycles': {
                'values': [3, 4]
            },
            'data': {
                'values': ['arc2eval-aug-200', 'arc2eval-aug-400', 'arc2eval-aug-800']
            },
            'lr': {
                'values': [1e-4, 2e-4]
            },
            'halt_max_steps': {
                "values": [8, 16, 24]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 2,
            's': 2
        }
    }
    
    # Set project name
    project_name = "TinyRecursiveModels-Sweep"
    
    print("Creating wandb sweep...")
    print(f"Sweep config: {sweep_config}")
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name
    )
    
    print(f"Sweep created with ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Run sweep agent
    print("Starting sweep agent...")
    wandb.agent(
        sweep_id=sweep_id,
        function=train,
        count=20  # Run 20 trials
    )


if __name__ == "__main__":
    main()
