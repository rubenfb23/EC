#!/usr/bin/env python3
"""
HOPE Distributed Training Script for AI-Generated Text Detection.

Run with torchrun for multi-GPU training:
    torchrun --nproc_per_node=4 hope_train_distributed.py

Or with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 hope_train_distributed.py

References:
- Nested Learning: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- Titans: https://arxiv.org/abs/2501.00663
"""

import os
import warnings

# Suppress warnings before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NCCL.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

import random
import math
import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
import argparse

import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/distributed environments
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoModel, AutoTokenizer, set_seed, logging as transformers_logging
transformers_logging.set_verbosity_error()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HOPEConfig:
    """Configuration for HOPE model."""

    # Base encoder
    base_model: str = "microsoft/deberta-v3-base"
    max_length: int = 512

    # Neural Memory Module
    memory_dim: int = 256
    memory_hidden_dim: int = 512
    memory_layers: int = 2

    # Continuum Memory System (CMS)
    num_memory_levels: int = 3
    update_frequencies: Tuple[int, ...] = (1, 4, 16)

    # Surprise mechanism
    surprise_momentum: float = 0.9
    surprise_scale: float = 0.1
    forget_rate: float = 0.01

    # Training
    learning_rate: float = 2e-5
    memory_lr: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 5
    batch_size: int = 28  # Per GPU batch size
    gradient_accumulation: int = 2
    warmup_ratio: float = 0.1

    # CV
    n_splits: int = 5

    # Classifier
    num_classes: int = 2
    dropout: float = 0.1

    # Nested optimization
    inner_steps: int = 1
    use_self_referential: bool = True

    # Distributed
    sync_cms_steps: bool = False  # Disable CMS sync to prevent NCCL hangs


# =============================================================================
# Model Components
# =============================================================================


class NeuralMemoryModule(nn.Module):
    """
    Neural Memory Module from Titans architecture.
    Uses an MLP to store key-value associations with surprise-based updates.
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        surprise_momentum: float = 0.9,
        surprise_scale: float = 0.1,
        forget_rate: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim

        self.eta = surprise_momentum
        self.theta = surprise_scale
        self.alpha = forget_rate

        # Key-Value projections
        self.key_proj = nn.Linear(input_dim, memory_dim)
        self.value_proj = nn.Linear(input_dim, memory_dim)
        self.query_proj = nn.Linear(input_dim, memory_dim)

        # Memory MLP
        layers = []
        dims = [memory_dim] + [hidden_dim] * (num_layers - 1) + [memory_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.memory_mlp = nn.Sequential(*layers)

        # Learnable gates
        self.eta_gate = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.theta_gate = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
        self.alpha_gate = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

        self.register_buffer("surprise_momentum_buffer", None)
        self.output_proj = nn.Linear(memory_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Project to keys, values, queries
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)

        keys = F.normalize(keys, p=2, dim=-1)
        queries = F.normalize(queries, p=2, dim=-1)

        # Compute gated parameters (preserved for architecture)
        x_pooled = x.mean(dim=1)
        eta_t = self.eta * self.eta_gate(x_pooled).squeeze(-1)
        theta_t = self.theta * self.theta_gate(x_pooled).squeeze(-1)
        alpha_t = self.alpha * self.alpha_gate(x_pooled).squeeze(-1)

        # Store gate statistics for monitoring (detached to avoid affecting computation graph)
        self.last_stats = {
            "eta_mean": eta_t.mean().detach().cpu().item(),
            "theta_mean": theta_t.mean().detach().cpu().item(),
            "alpha_mean": alpha_t.mean().detach().cpu().item(),
        }

        # MLP pass 1: Predict values from keys (for association loss)
        predicted_values = self.memory_mlp(keys)
        assoc_loss = F.mse_loss(predicted_values, values)

        # MLP pass 2: Retrieve from queries (for output)
        retrieved = self.memory_mlp(queries)
        output = self.output_proj(retrieved)

        if output.size(1) == 1:
            output = output.squeeze(1)

        if return_loss:
            return output, assoc_loss

        return output


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System (CMS) from HOPE.
    Multiple memory modules updating at different frequencies.
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        hidden_dim: int,
        num_levels: int = 3,
        update_frequencies: Tuple[int, ...] = (1, 4, 16),
        surprise_momentum: float = 0.9,
        surprise_scale: float = 0.1,
        forget_rate: float = 0.01,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.update_frequencies = update_frequencies[:num_levels]

        self.memory_levels = nn.ModuleList(
            [
                NeuralMemoryModule(
                    input_dim=input_dim,
                    memory_dim=memory_dim,
                    hidden_dim=hidden_dim,
                    surprise_momentum=surprise_momentum + 0.05 * i,
                    surprise_scale=surprise_scale / (i + 1),
                    forget_rate=forget_rate / (i + 1),
                )
                for i in range(num_levels)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(input_dim * num_levels, input_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
        )

        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        self.register_buffer("step_counter", torch.tensor(0))
        self.slowest_frequency = max(self.update_frequencies)
        self.register_buffer("last_sync_step", torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        return_level_outputs: bool = False,
        return_memory_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        level_outputs = []
        level_losses = []

        for i, (memory, freq) in enumerate(
            zip(self.memory_levels, self.update_frequencies)
        ):
            output, assoc_loss = memory(x, return_loss=True)
            level_outputs.append(output)
            level_losses.append(assoc_loss)

        # Note: step_counter is incremented in the training loop once per optimizer step,
        # not here per micro-batch, to correctly align with gradient accumulation.

        weights = F.softmax(self.level_weights, dim=0)

        if level_outputs[0].dim() == 2:
            stacked = torch.stack(level_outputs, dim=-1)
            weighted = (stacked * weights.view(1, 1, -1)).sum(dim=-1)
            concat = torch.cat(level_outputs, dim=-1)
            fused = self.fusion(concat)
            output = weighted + fused
        else:
            stacked = torch.stack(level_outputs, dim=-1)
            weighted = (stacked * weights.view(1, 1, 1, -1)).sum(dim=-1)
            concat = torch.cat(level_outputs, dim=-1)
            fused = self.fusion(concat)
            output = weighted + fused

        if return_level_outputs:
            if return_memory_loss:
                total_memory_loss = self._compute_weighted_loss(level_losses)
                return output, level_outputs, level_losses, total_memory_loss
            return output, level_outputs, level_losses

        if return_memory_loss:
            total_memory_loss = self._compute_weighted_loss(level_losses)
            return output, total_memory_loss

        return output

    def _compute_weighted_loss(self, level_losses: List[torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=level_losses[0].device)
        for loss, freq in zip(level_losses, self.update_frequencies):
            weight = 1.0 / freq
            total_loss = total_loss + weight * loss
        return total_loss

    def reset_step_counter(self):
        self.step_counter.zero_()
        self.last_sync_step.zero_()

    @torch.no_grad()
    def sync_step_counter(self):
        """
        Periodically synchronize CMS state without per-step all-reduce.

        WARNING: This method can cause NCCL deadlocks if ranks have different
        step counts. Only call at epoch boundaries or keep sync_cms_steps=False.
        """
        if not dist.is_initialized():
            return

        steps_since_sync = int((self.step_counter - self.last_sync_step).item())
        if steps_since_sync < self.slowest_frequency:
            return

        dist.broadcast(self.step_counter, src=0)
        self.last_sync_step.copy_(self.step_counter)


class SelfReferentialModule(nn.Module):
    """Self-Referential Module - HOPE's key innovation over Titans."""

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim

        self.meta_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )

        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.persistent_memory = nn.Parameter(torch.randn(1, dim) * 0.02)

    def forward(self, x: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, memory_output], dim=-1)
        update = self.meta_net(combined)
        gate = self.gate(memory_output)
        output = memory_output + gate * update
        persistent = self.persistent_memory.expand(x.size(0), -1)
        output = output + 0.1 * persistent
        return output


class HOPEClassifier(nn.Module):
    """HOPE Classifier for Text Classification."""

    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(config.base_model)
        self.encoder_dim = self.encoder.config.hidden_size

        self.cms = ContinuumMemorySystem(
            input_dim=self.encoder_dim,
            memory_dim=config.memory_dim,
            hidden_dim=config.memory_hidden_dim,
            num_levels=config.num_memory_levels,
            update_frequencies=config.update_frequencies,
            surprise_momentum=config.surprise_momentum,
            surprise_scale=config.surprise_scale,
            forget_rate=config.forget_rate,
        )

        self.self_ref = (
            SelfReferentialModule(
                dim=self.encoder_dim,
                hidden_dim=config.memory_hidden_dim,
            )
            if config.use_self_referential
            else None
        )

        self.attention_pool = nn.Sequential(nn.Linear(self.encoder_dim, 1))

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_dim * 2, self.encoder_dim),
            nn.SiLU(),
            nn.LayerNorm(self.encoder_dim),
            nn.Dropout(config.dropout),
            nn.Linear(self.encoder_dim, config.num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_memory_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = encoder_output.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        mean_pooled = sum_hidden / mask_expanded.sum(dim=1).clamp(min=1e-9)

        attn_weights = self.attention_pool(hidden_states).squeeze(-1)
        attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_pooled = (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)

        if return_memory_loss:
            memory_output, memory_loss = self.cms(mean_pooled, return_memory_loss=True)
        else:
            memory_output = self.cms(mean_pooled)

        if self.self_ref is not None:
            memory_output = self.self_ref(mean_pooled, memory_output)

        combined = torch.cat([attn_pooled, memory_output], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        if return_memory_loss:
            output["memory_loss"] = memory_loss

        return output

    def get_memory_parameters(self):
        params = list(self.cms.parameters())
        if self.self_ref is not None:
            params.extend(list(self.self_ref.parameters()))
        return params

    def get_encoder_parameters(self):
        params = list(self.encoder.parameters())
        params.extend(list(self.classifier.parameters()))
        params.extend(list(self.attention_pool.parameters()))
        return params


# =============================================================================
# Dataset
# =============================================================================


class TextDataset(Dataset):
    """Dataset for AI vs Human text classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str = "text",
        label_col: Optional[str] = "label",
        max_length: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        text = str(self.df.loc[idx, self.text_col])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.label_col and self.label_col in self.df.columns:
            item["labels"] = torch.tensor(int(self.df.loc[idx, self.label_col]))

        return item


# =============================================================================
# Distributed Training Utilities
# =============================================================================


def setup_distributed():
    """Initialize distributed training."""
    # Increase NCCL timeout to 30 minutes (default is 10 minutes)
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    # Enable async error handling for better debugging
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(msg):
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


def get_world_size():
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=op)
    return rt


# =============================================================================
# Training Monitor for Real-Time Visualization
# =============================================================================


class TrainingMonitor:
    """
    Real-time monitoring and visualization for HOPE training.

    Only runs on Rank 0 to avoid file conflicts in distributed training.
    Logs metrics to CSV and generates plots at configurable intervals.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "training_metrics",
        plot_interval: int = 100,
        num_memory_levels: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.plot_interval = plot_interval
        self.num_memory_levels = num_memory_levels

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        self.csv_path = self.output_dir / "metrics_log.csv"

        # Historical data storage
        self.history: Dict[str, List[float]] = {
            "step": [],
            "epoch": [],
            "cls_loss": [],
            "mem_loss": [],
            "outer_lr": [],
            "inner_lr": [],
        }
        # Add gate metrics for each memory level
        for level in range(num_memory_levels):
            self.history[f"eta_mean_L{level}"] = []
            self.history[f"theta_mean_L{level}"] = []
            self.history[f"alpha_mean_L{level}"] = []

        # Initialize CSV with headers
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(self.history.keys()))

    def log(
        self,
        step: int,
        epoch: int,
        cls_loss: float,
        mem_loss: float,
        outer_optimizer: Optimizer,
        inner_optimizer: Optimizer,
        model: nn.Module,
    ):
        """
        Log metrics for the current step.

        Args:
            step: Current global step (optimizer steps, not micro-batches)
            epoch: Current epoch
            cls_loss: Classification loss value
            mem_loss: Memory loss value
            outer_optimizer: Encoder optimizer (for LR extraction)
            inner_optimizer: Memory optimizer (for LR extraction)
            model: The FSDP-wrapped model
        """
        # Extract learning rates
        outer_lr = outer_optimizer.param_groups[0]["lr"]
        inner_lr = inner_optimizer.param_groups[0]["lr"]

        # Extract gate statistics from memory levels
        # Access underlying module through FSDP wrapper
        try:
            base_module = model.module if hasattr(model, "module") else model
            memory_levels = base_module.cms.memory_levels

            gate_stats = []
            for level_idx, memory in enumerate(memory_levels):
                if hasattr(memory, "last_stats"):
                    gate_stats.append(memory.last_stats)
                else:
                    # Fallback if last_stats not yet computed
                    gate_stats.append({
                        "eta_mean": 0.0,
                        "theta_mean": 0.0,
                        "alpha_mean": 0.0,
                    })
        except Exception:
            # Fallback for edge cases
            gate_stats = [
                {"eta_mean": 0.0, "theta_mean": 0.0, "alpha_mean": 0.0}
                for _ in range(self.num_memory_levels)
            ]

        # Append to history
        self.history["step"].append(step)
        self.history["epoch"].append(epoch)
        self.history["cls_loss"].append(cls_loss)
        self.history["mem_loss"].append(mem_loss)
        self.history["outer_lr"].append(outer_lr)
        self.history["inner_lr"].append(inner_lr)

        for level_idx, stats in enumerate(gate_stats):
            self.history[f"eta_mean_L{level_idx}"].append(stats["eta_mean"])
            self.history[f"theta_mean_L{level_idx}"].append(stats["theta_mean"])
            self.history[f"alpha_mean_L{level_idx}"].append(stats["alpha_mean"])

        # Append row to CSV
        row = [
            step, epoch, cls_loss, mem_loss, outer_lr, inner_lr,
        ]
        for level_idx in range(self.num_memory_levels):
            row.extend([
                gate_stats[level_idx]["eta_mean"],
                gate_stats[level_idx]["theta_mean"],
                gate_stats[level_idx]["alpha_mean"],
            ])

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Generate plots at intervals
        if step > 0 and step % self.plot_interval == 0:
            self._generate_plots()

    def _generate_plots(self):
        """Generate and save monitoring plots."""
        steps = self.history["step"]
        if len(steps) < 2:
            return

        # Plot 1: Losses
        self._plot_losses(steps)

        # Plot 2: Gate evolution
        self._plot_gates(steps)

        # Plot 3: Learning rates
        self._plot_lrs(steps)

        plt.close('all')  # Free memory

    def _plot_losses(self, steps: List[int]):
        """Plot classification and memory losses."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_cls = 'tab:blue'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Classification Loss', color=color_cls)
        ax1.plot(steps, self.history["cls_loss"], color=color_cls, label='CLS Loss', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_cls)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color_mem = 'tab:orange'
        ax2.set_ylabel('Memory Loss', color=color_mem)
        ax2.plot(steps, self.history["mem_loss"], color=color_mem, label='Mem Loss', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color_mem)

        fig.suptitle('Training Losses Over Time', fontsize=14, fontweight='bold')
        fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
        fig.tight_layout()
        fig.savefig(self.output_dir / "plot_losses.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_gates(self, steps: List[int]):
        """Plot gate evolution across memory levels."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        gate_names = ["eta", "theta", "alpha"]
        gate_labels = [r"$\eta$ (Input Gate)", r"$\theta$ (Surprise Gate)", r"$\alpha$ (Forget Gate)"]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, self.num_memory_levels))

        for ax_idx, (gate_name, gate_label) in enumerate(zip(gate_names, gate_labels)):
            ax = axes[ax_idx]
            for level_idx in range(self.num_memory_levels):
                key = f"{gate_name}_mean_L{level_idx}"
                ax.plot(
                    steps,
                    self.history[key],
                    color=colors[level_idx],
                    label=f"Level {level_idx}",
                    alpha=0.8,
                    linewidth=1.5,
                )
            ax.set_xlabel('Step')
            ax.set_ylabel(f'{gate_label} Mean')
            ax.set_title(gate_label)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Memory Gate Evolution (Saturation Monitor)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(self.output_dir / "plot_gates.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_lrs(self, steps: List[int]):
        """Plot learning rate schedules."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(steps, self.history["outer_lr"], label='Encoder LR (Outer)', color='tab:blue', linewidth=2)
        ax.plot(steps, self.history["inner_lr"], label='Memory LR (Inner)', color='tab:red', linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedules', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for LRs

        fig.tight_layout()
        fig.savefig(self.output_dir / "plot_lrs.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def finalize(self):
        """Generate final plots at the end of training."""
        if len(self.history["step"]) > 0:
            self._generate_plots()
            print_rank0(f"Training metrics saved to: {self.output_dir}")


# =============================================================================
# Optimizer: M3 (Multi-scale Momentum Muon)
# =============================================================================


class M3(Optimizer):
    """
    Multi-scale Momentum Muon optimizer.

    - Hierarchical momenta: each level integrates the level below (not the raw grad),
      stabilizing nested loops.
    - Newton-Schulz orthogonalization on the update covariance to condition steps.
    - Decoupled weight decay (AdamW style).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-5,
        betas: Tuple[float, ...] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        ns_iters: int = 5,
        ns_eps: float = 1e-4,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if any(b <= 0.0 or b >= 1.0 for b in betas):
            raise ValueError("Invalid beta parameter(s): {}".format(betas))

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            ns_iters=ns_iters,
            ns_eps=ns_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _newton_schulz(self, mat: torch.Tensor, num_iters: int, eps: float):
        """Matrix inverse square root via Newton-Schulz iteration."""
        # Scale for numerical stability
        norm = mat.norm() + eps
        Y = mat / norm
        I = torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype)
        Z = I.clone()

        for _ in range(num_iters):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z

        return Z / norm.sqrt()

    @torch.no_grad()
    def _orthogonalize_update(
        self, update: torch.Tensor, num_iters: int, eps: float
    ) -> torch.Tensor:
        if update.ndim < 2:
            return update

        # Flatten into (rows, cols)
        flat = update.reshape(update.shape[0], -1)
        mat = flat if flat.dtype in (torch.float32, torch.float64) else flat.float()

        cov = mat @ mat.transpose(0, 1)
        cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        inv_sqrt = self._newton_schulz(cov, num_iters=num_iters, eps=eps)
        conditioned = inv_sqrt @ mat
        conditioned = conditioned.to(update.dtype)
        return conditioned.reshape_as(update)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            ns_iters = group["ns_iters"]
            ns_eps = group["ns_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("M3 does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentums"] = [torch.zeros_like(p) for _ in betas]

                state["step"] += 1

                # Decoupled weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Hierarchical momentum: each level follows the previous level
                prev = grad
                momentums = state["momentums"]
                for i, beta in enumerate(betas):
                    m = momentums[i]
                    m.mul_(beta).add_(prev, alpha=1.0 - beta)
                    momentums[i] = m
                    prev = m

                update = momentums[-1]
                update = self._orthogonalize_update(update, ns_iters, ns_eps + eps)

                p.add_(update, alpha=-lr)

        return loss


def build_fsdp_auto_wrap_policy():
    """
    Size-based auto wrap that keeps the ContinuumMemorySystem intact.

    We wrap CMS as a single FSDP unit so the memory hierarchy is not fragmented,
    and skip wrapping its inner NeuralMemoryModule blocks. Everything else
    follows a 10M parameter threshold.
    """

    base_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=10_000_000)

    def _policy(module, recurse, nonwrapped_numel, **kwargs):
        if isinstance(module, ContinuumMemorySystem):
            return True
        if isinstance(module, NeuralMemoryModule):
            return False
        return base_wrap_policy(module, recurse, nonwrapped_numel)

    return _policy


# =============================================================================
# Training
# =============================================================================


def train_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_df: pd.DataFrame,
    config: HOPEConfig,
    local_rank: int,
    text_col: str = "text",
    label_col: str = "label",
    model_dir: Path = None,
) -> Tuple[np.ndarray, float, str]:
    """Train a single fold with distributed training."""

    world_size = get_world_size()
    print_rank0(f"\n{'='*50}")
    print_rank0(f"Training Fold {fold + 1}/{config.n_splits}")
    print_rank0(f"World size: {world_size} GPUs")
    print_rank0(f"{'='*50}")

    # Prepare data
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)

    train_dataset = TextDataset(
        train_data, tokenizer, text_col, label_col, config.max_length
    )
    val_dataset = TextDataset(
        val_data, tokenizer, text_col, label_col, config.max_length
    )

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model with FSDP
    device = torch.device(f"cuda:{local_rank}")
    base_model = HOPEClassifier(config).to(device)

    fsdp_policy = build_fsdp_auto_wrap_policy()
    fsdp_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    model = FSDP(
        base_model,
        auto_wrap_policy=fsdp_policy,
        cpu_offload=None,  # Disable CPU offload to prevent NCCL timeouts
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
    )

    # Access the underlying module for parameter groups
    base_model = model.module

    # Nested optimization: two optimizers (M3 for stability in nested loops)
    outer_optimizer = M3(
        base_model.get_encoder_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    inner_optimizer = M3(
        base_model.get_memory_parameters(),
        lr=config.memory_lr,
        weight_decay=config.weight_decay * 0.1,
        betas=(0.9, 0.98),
    )

    # Schedulers
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation
    outer_scheduler = CosineAnnealingLR(outer_optimizer, T_max=total_steps)
    inner_scheduler = CosineAnnealingLR(inner_optimizer, T_max=total_steps)

    # Training loop
    best_auc = 0
    best_model_state = None
    fold_dir = model_dir / f"fold_{fold}"
    if is_main_process():
        fold_dir.mkdir(exist_ok=True)

    # bf16 autocast does not require loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    for epoch in range(config.num_epochs):
        model.train()
        base_model.cms.reset_step_counter()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling

        epoch_loss = torch.tensor(0.0, device=device)
        epoch_memory_loss = torch.tensor(0.0, device=device)
        num_batches = torch.tensor(0, device=device)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            disable=not is_main_process(),
        )

        outer_optimizer.zero_grad()
        inner_optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_memory_loss=True,
                )

                cls_loss = outputs["loss"] / config.gradient_accumulation
                mem_loss = outputs["memory_loss"] / config.gradient_accumulation
                total_loss = cls_loss + 0.1 * mem_loss

            scaler.scale(total_loss).backward()

            epoch_loss += cls_loss.detach() * config.gradient_accumulation
            epoch_memory_loss += mem_loss.detach() * config.gradient_accumulation
            num_batches += 1

            if (step + 1) % config.gradient_accumulation == 0:
                scaler.unscale_(outer_optimizer)
                scaler.unscale_(inner_optimizer)
                model.clip_grad_norm_(1.0)

                # Gradient masking for CMS frequency-based updates:
                # Zero out gradients for memory levels that shouldn't update this step
                cms = base_model.cms
                current_step = int(cms.step_counter.item())
                for level_idx, (memory_level, freq) in enumerate(
                    zip(cms.memory_levels, cms.update_frequencies)
                ):
                    if current_step % freq != 0:
                        # This level should not update - zero its gradients
                        for param in memory_level.parameters():
                            if param.grad is not None:
                                param.grad = None

                # Single optimizer step for each optimizer (no inner loop)
                scaler.step(inner_optimizer)
                scaler.step(outer_optimizer)
                scaler.update()

                outer_optimizer.zero_grad()
                inner_optimizer.zero_grad()

                # Increment CMS step counter once per optimizer step
                cms.step_counter += 1

                # Let CMS perform distributed sync only when needed
                if config.sync_cms_steps:
                    cms.sync_step_counter()

                outer_scheduler.step()
                inner_scheduler.step()

            if is_main_process():
                pbar.set_postfix(
                    {
                        "loss": f"{epoch_loss.item() / max(num_batches.item(), 1):.4f}",
                        "mem_loss": f"{epoch_memory_loss.item() / max(num_batches.item(), 1):.4f}",
                    }
                )

        # Reduce losses across GPUs
        epoch_loss = reduce_tensor(epoch_loss) / world_size
        num_batches = reduce_tensor(num_batches)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc="Validating", disable=not is_main_process()
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = F.softmax(outputs["logits"], dim=-1)[:, 1]
                val_preds.append(probs)
                val_labels.append(batch["labels"].to(device))

        # Gather predictions from all GPUs
        val_preds = torch.cat(val_preds).float()
        val_labels = torch.cat(val_labels)

        # Gather all predictions to rank 0
        all_preds = [torch.zeros_like(val_preds) for _ in range(world_size)]
        all_labels = [torch.zeros_like(val_labels) for _ in range(world_size)]
        dist.all_gather(all_preds, val_preds)
        dist.all_gather(all_labels, val_labels)

        if is_main_process():
            all_preds = torch.cat(all_preds).float().cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()

            # Remove duplicates from distributed sampler padding
            n_val = len(val_data)
            all_preds = all_preds[:n_val]
            all_labels = all_labels[:n_val]

            val_auc = roc_auc_score(all_labels, all_preds)
            val_acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

            print(
                f"Epoch {epoch + 1}: Loss={epoch_loss.item()/num_batches.item():.4f}, "
                f"Val AUC={val_auc:.5f}, Val Acc={val_acc:.5f}"
            )

            is_new_best = val_auc > best_auc
            if is_new_best:
                best_auc = val_auc
                print(f"  -> New best model! AUC: {best_auc:.5f}")
        else:
            is_new_best = False

        # Broadcast save decision from rank 0 so ALL ranks participate in state_dict
        should_save_tensor = torch.tensor(1 if is_new_best else 0, device=device)
        dist.broadcast(should_save_tensor, src=0)

        # ALL ranks must call state_dict() together - FSDP requires collective participation
        if should_save_tensor.item():
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, fsdp_state_cfg
            ):
                state_dict = model.state_dict()  # All ranks call this (collective op)
            if is_main_process():
                best_model_state = state_dict  # Only rank 0 keeps it

        # Broadcast best_auc to all ranks for consistent state
        best_auc_tensor = torch.tensor(best_auc, device=device)
        dist.broadcast(best_auc_tensor, src=0)
        best_auc = best_auc_tensor.item()

        # Synchronize before next epoch
        dist.barrier()

    # Save best model (rank 0 only)
    model_path = fold_dir / "best_model.pt"
    if is_main_process() and best_model_state is not None:
        torch.save(best_model_state, model_path)
        print(f"Saved best model to {model_path}")

    dist.barrier()

    # Get OOF predictions using best model
    has_best_state = best_model_state is not None or model_path.exists()
    if has_best_state:
        if is_main_process() and best_model_state is None:
            best_model_state = torch.load(model_path, map_location="cpu")

        load_state = best_model_state if is_main_process() else {}
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_state_cfg):
            model.load_state_dict(load_state)

    dist.barrier()
    model.eval()
    oof_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = F.softmax(outputs["logits"], dim=-1)[:, 1]
            oof_preds.append(probs)

    oof_preds = torch.cat(oof_preds).float()
    all_oof = [torch.zeros_like(oof_preds) for _ in range(world_size)]
    dist.all_gather(all_oof, oof_preds)

    if is_main_process():
        all_oof = torch.cat(all_oof).float().cpu().numpy()[: len(val_data)]
    else:
        all_oof = None

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return all_oof, best_auc, str(model_path)


def main():
    parser = argparse.ArgumentParser(description="HOPE Distributed Training")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=30, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup distributed
    local_rank = setup_distributed()

    # Set seeds
    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Config
    config = HOPEConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_splits=args.n_splits,
    )

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        cwd = Path.cwd()
        if (cwd / "src/ai_vs_human").exists():
            data_dir = cwd / "src/ai_vs_human"
        elif cwd.name == "ai_vs_human":
            data_dir = cwd
        else:
            data_dir = cwd

    train_file = data_dir / "merged_ai_human_multisocial_features_cleaned_train.csv"
    test_file = data_dir / "merged_ai_human_multisocial_features_cleaned_test.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")

    model_dir = data_dir / "models" / "hope"
    oof_dir = data_dir / "oof"

    if is_main_process():
        model_dir.mkdir(parents=True, exist_ok=True)
        oof_dir.mkdir(exist_ok=True)

    dist.barrier()

    print_rank0(f"Training data: {train_file}")
    print_rank0(f"Test data: {test_file}")
    print_rank0(f"Model directory: {model_dir}")
    print_rank0(f"Config: {config}")

    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file) if test_file.exists() else None

    text_col = "text"
    label_col = "label"

    if text_col not in train_df.columns:
        for alt in ["text_content", "content"]:
            if alt in train_df.columns:
                train_df = train_df.rename(columns={alt: text_col})
                if test_df is not None:
                    test_df = test_df.rename(columns={alt: text_col})
                break

    print_rank0(f"Train shape: {train_df.shape}")
    print_rank0(f"Label distribution:\n{train_df[label_col].value_counts()}")

    y = train_df[label_col].values

    # Setup CV
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=args.seed)
    folds = list(cv.split(train_df, y))

    # Train all folds
    oof_predictions = np.zeros(len(train_df))
    fold_aucs = []
    model_paths = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        oof_preds, best_auc, model_path = train_fold(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            train_df=train_df,
            config=config,
            local_rank=local_rank,
            text_col=text_col,
            label_col=label_col,
            model_dir=model_dir,
        )

        if is_main_process():
            oof_predictions[val_idx] = oof_preds
            fold_aucs.append(best_auc)
            model_paths.append(model_path)

        dist.barrier()

    # Final results (rank 0 only)
    if is_main_process():
        overall_auc = roc_auc_score(y, oof_predictions)
        print(f"\n{'='*50}")
        print(f"HOPE OOF Results (Distributed)")
        print(f"{'='*50}")
        print(f"Fold AUCs: {[f'{auc:.5f}' for auc in fold_aucs]}")
        print(f"Mean Fold AUC: {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})")
        print(f"Overall OOF AUC: {overall_auc:.5f}")

        # Save OOF predictions
        pd.DataFrame({"oof_hope": oof_predictions}).to_csv(
            oof_dir / "oof_hope.csv", index=False
        )
        print(f"\nSaved OOF predictions to {oof_dir / 'oof_hope.csv'}")

        # Test inference
        if test_df is not None:
            print("\nRunning inference on test set...")
            # Load all fold models and average predictions
            tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
            test_dataset = TextDataset(
                test_df, tokenizer, text_col, None, config.max_length
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size * 2,
                shuffle=False,
                num_workers=4,
            )

            all_preds = []
            device = torch.device(f"cuda:{local_rank}")

            for model_path in model_paths:
                model = HOPEClassifier(config).to(device)
                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )
                model.eval()

                fold_preds = []
                with torch.no_grad():
                    for batch in tqdm(
                        test_loader,
                        desc=f"Predicting with {Path(model_path).parent.name}",
                    ):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            outputs = model(
                                input_ids=input_ids, attention_mask=attention_mask
                            )

                        probs = F.softmax(outputs["logits"], dim=-1)[:, 1]
                        fold_preds.extend(probs.float().cpu().numpy())

                all_preds.append(fold_preds)
                del model
                torch.cuda.empty_cache()

            test_preds = np.mean(all_preds, axis=0)

            submission = pd.DataFrame(
                {
                    "id": (
                        test_df.index if "id" not in test_df.columns else test_df["id"]
                    ),
                    "prediction": test_preds,
                }
            )
            submission.to_csv(data_dir / "submission_hope.csv", index=False)
            print(f"Saved submission to {data_dir / 'submission_hope.csv'}")

            if label_col in test_df.columns:
                test_auc = roc_auc_score(test_df[label_col], test_preds)
                test_acc = accuracy_score(
                    test_df[label_col], (test_preds > 0.5).astype(int)
                )
                print(f"Test AUC: {test_auc:.5f}")
                print(f"Test Accuracy: {test_acc:.5f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
