import io
import json
import os
from typing import Dict, Optional, Sequence

from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from numba import njit
from PIL import Image

import wandb
from dataset.build_arc_dataset import arc_grid_to_np, grid_hash, inverse_aug
from dataset.common import PuzzleDatasetMetadata

# Standard ARC color palette (RGB values for digits 0-9)
ARC_COLORS = {
    0: (0, 0, 0),        # Black
    1: (0, 0, 255),      # Blue
    2: (255, 0, 0),      # Red
    3: (0, 255, 0),      # Green
    4: (255, 255, 0),    # Yellow
    5: (128, 128, 128),  # Gray
    6: (255, 0, 255),    # Magenta
    7: (255, 165, 0),    # Orange
    8: (173, 216, 230),  # Light Blue
    9: (165, 42, 42)     # Brown
}

def create_arc_colormap():
    """Create a matplotlib colormap for ARC grids."""
    colors = [ARC_COLORS[i] for i in range(10)]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]  # Normalize to [0,1]
    return mcolors.ListedColormap(colors)

def visualize_arc_grid(grid: np.ndarray, title: str = ""):
    """Visualize a single ARC grid with proper colors."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    create_arc_grid_ax(grid, ax, title)
    plt.tight_layout()
    return fig


def create_arc_grid_ax(grid: np.ndarray, ax: Axes, title: str):
    cmap = create_arc_colormap()
    
    # Convert to int32 to allow negative values for empty cells
    grid_plot = grid.astype(np.int32)
    # Use -1 for empty cells (value 0)
    grid_plot[grid == 0] = -1
    
    im = ax.imshow(grid_plot, cmap=cmap, vmin=-1, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)


def create_composite_visualization(input_grid: np.ndarray, expected_grid: np.ndarray, 
                                 pred1_grid: np.ndarray, pred2_grid: np.ndarray,
                                 puzzle_name: str, test_idx: int):
    """Create a 4-panel composite visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    cmap = create_arc_colormap()
    
    grids = [input_grid, expected_grid, pred1_grid, pred2_grid]
    titles = ["Input", "Expected", "Prediction 1", "Prediction 2"]
    
    for i, (grid, title) in enumerate(zip(grids, titles)):
        ax = axes[i // 2, i % 2]
        
        # Convert to int32 to allow negative values for empty cells
        grid_plot = grid.astype(np.int32)
        # Use -1 for empty cells (value 0)
        grid_plot[grid == 0] = -1
        
        im = ax.imshow(grid_plot, cmap=cmap, vmin=-1, vmax=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines
        for row in range(grid.shape[0] + 1):
            ax.axhline(row - 0.5, color='black', linewidth=0.5)
        for col in range(grid.shape[1] + 1):
            ax.axvline(col - 0.5, color='black', linewidth=0.5)
    
    fig.suptitle(f"ARC Puzzle: {puzzle_name} (Test {test_idx})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)
    
    return pil_image

@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC:
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
    
    def __init__(self, data_path: str, 
        eval_metadata: PuzzleDatasetMetadata, 
        submission_K: int = 2, 
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000), 
        aggregated_voting: bool = True):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        # Load identifiers and test puzzles
        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)
            
        # States
        self._local_hmap = {}
        self._local_preds = {}
        
    def begin_eval(self):
        if not self.aggregated_voting:
            # Clear previous predictions
            self._local_hmap = {}
            self._local_preds = {}
    
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Collect required outputs to CPU
        outputs = {}
        q_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()
                        
        assert q_values is not None # TODO: Matthieu fix for no_act

        # Remove padding from outputs
        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        # Get predictions
        for identifier, input, pred, q in zip(outputs["puzzle_identifiers"].numpy(), outputs["inputs"].numpy(), outputs["preds"].numpy(), q_values.numpy()):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            input_hash = grid_hash(_inverse_fn(_crop(input)))
            
            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            # Store into local state
            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred
            
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))
    
    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        # Gather predictions to rank 0 for voting
        if group is not None:
            # Multi-process mode: gather from all processes
            global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object((self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group)
        else:
            # Single-process mode: use local predictions directly
            global_hmap_preds = [(self._local_hmap, self._local_preds)] if rank == 0 else None
        
        # Rank 0 logic
        if rank != 0:
            return

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]
        visualization_images = {}  # Store images for wandb logging
        puzzle_count = 0  # Track number of puzzles processed

        for name, puzzle in self.test_puzzles.items():

            # Process test examples in this puzzle
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            for test_idx, pair in enumerate(puzzle["test"]):
                input_hash = grid_hash(arc_grid_to_np(pair["input"]))
                label_hash = grid_hash(arc_grid_to_np(pair["output"]))
                
                p_map = {}
                for hmap, preds in global_hmap_preds:  # type: ignore
                    for h, q in preds.get(name, {}).get(input_hash, {}):
                        p_map.setdefault(h, [0, 0])
                        p_map[h][0] += 1
                        p_map[h][1] += q
                        
                if not len(p_map):
                    print (f"Puzzle {name} has no predictions.")
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]
                    
                p_map = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                # vote for different Ks
                for i, k in enumerate(self.pass_Ks):
                    ok = False
                    for h, stats in p_map[:k]:
                        ok |= h == label_hash
                        
                    if ok:
                        print(f"Puzzle {name} is correct for pass@{k}")
                    num_test_correct[i] += ok
                    
                # Query grids
                pred_grids = []
                for h, stats in p_map[:self.submission_K]:
                    for hmap, preds in global_hmap_preds:  # type: ignore
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break
                        
                # Pad to K
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0])
                
                submission[name].append({f"attempt_{i + 1}": grid.tolist() for i, grid in enumerate(pred_grids)})
                
                # Create visualization for this test case
                if len(pred_grids) >= 2 and puzzle_count < 32:  # Ensure we have at least 2 predictions
                    input_grid = arc_grid_to_np(pair["input"])
                    expected_grid = arc_grid_to_np(pair["output"])
                    pred1_grid = pred_grids[0]
                    pred2_grid = pred_grids[1]
                    
                    # Create composite visualization
                    viz_image = create_composite_visualization(
                        input_grid, expected_grid, pred1_grid, pred2_grid,
                        name, test_idx
                    )
                    
                    # Store for wandb logging
                    viz_key = f"ARC/viz/{name}_{test_idx}"
                    visualization_images[viz_key] = wandb.Image(viz_image)

            # Total correctness
            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / len(puzzle["test"])
            
            puzzle_count += 1

        # Save submission
        if save_path is not None:
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        # Final result
        all_results = {f"ARC/pass@{k}": correct[i] / len(self.test_puzzles) for i, k in enumerate(self.pass_Ks)}
        
        # Add visualization images to results
        all_results.update(visualization_images)
        
        # Debug print
        if len(visualization_images) > 0:
            print(f"Generated {len(visualization_images)} visualization images for wandb logging")
            for key in list(visualization_images.keys())[:3]:  # Show first 3 keys
                print(f"  - {key}")
        else:
            print("No visualization images generated - check if puzzles have predictions")

        return all_results