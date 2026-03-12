#!/usr/bin/env python3
"""
Neural Architecture Generation with MAP-Elites Quality-Diversity (NAG-ME-QD)
Part 2: MAP-Elites, Zero-Shot Metrics, Evaluation Methods
============================================================================
"""

import os
import sys
import json
import copy
import math
import time
import random
import hashlib
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable
    def trange(*args, **kwargs):
        return range(*args)

# Import from main module
from nag_mapelites_qd import (
    CellType, OperationType, DatasetType,
    OperationSpec, EdgeSpec, ArchitectureSpec,
    NASNetwork, TextToSpecEncoder,
    DatasetManager, ExperimentLogger,
    set_seed, cleanup_memory, CONFIG,
    create_operation
)


# =============================================================================
# ZERO-SHOT METRICS
# =============================================================================

class ZeroShotMetrics:
    """
    Zero-shot NAS metrics for quick architecture evaluation without training.
    Implements: SynFlow, Jacob_cov, NASWOT, Zen-Score, Gradient Norm
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def compute_synflow(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...] = (1, 3, 32, 32)
    ) -> float:
        """
        Compute SynFlow score (parameter saliency).
        Higher is better - indicates information flow through the network.
        """
        model = model.to(self.device)
        model.eval()
        
        # Set all parameters to require grad
        for p in model.parameters():
            p.requires_grad_(True)
            
        # Create dummy input with ones
        x = torch.ones(input_shape, device=self.device, requires_grad=True)
        
        try:
            # Forward pass
            output = model(x)
            
            # Sum all outputs
            if output.dim() > 1:
                output = output.sum()
            
            # Backward pass
            output.backward()
            
            # Compute SynFlow score
            score = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    score += (p * p.grad).abs().sum().item()
                    
            return np.log(score + 1e-10)
            
        except Exception as e:
            return -float('inf')
        finally:
            model.zero_grad()
            for p in model.parameters():
                p.requires_grad_(False)
                
    def compute_jacob_cov(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_samples: int = 10
    ) -> float:
        """
        Compute Jacobian covariance score.
        Measures how well gradients cover the input space.
        """
        model = model.to(self.device)
        model.eval()
        
        jacobians = []
        
        try:
            for i, (x, _) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                x = x[:1].to(self.device)
                x.requires_grad_(True)
                
                output = model(x)
                
                # Compute Jacobian for each output dimension
                jac = []
                for j in range(output.shape[1]):
                    model.zero_grad()
                    if x.grad is not None:
                        x.grad.zero_()
                    output[0, j].backward(retain_graph=True)
                    if x.grad is not None:
                        jac.append(x.grad.flatten().cpu().numpy())
                        
                if jac:
                    jacobians.append(np.stack(jac))
                    
            if not jacobians:
                return -float('inf')
                
            # Concatenate all Jacobians
            J = np.concatenate(jacobians, axis=0)
            
            # Compute covariance
            cov = np.cov(J)
            
            # Return log determinant (or trace for numerical stability)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 1e-10]
            
            if len(eigvals) == 0:
                return -float('inf')
                
            return np.sum(np.log(eigvals))
            
        except Exception as e:
            return -float('inf')
            
    def compute_naswot(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_samples: int = 128
    ) -> float:
        """
        Compute NASWOT (NAS Without Training) score.
        Based on the correlation between activations at initialization.
        """
        model = model.to(self.device)
        model.eval()
        
        # Collect activations
        activations = []
        hooks = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach().cpu())
                
        # Register hooks on ReLU layers
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_fn))
                
        try:
            collected = 0
            for x, _ in dataloader:
                if collected >= num_samples:
                    break
                    
                x = x.to(self.device)
                activations.clear()
                
                with torch.no_grad():
                    model(x)
                    
                collected += x.shape[0]
                
            # Remove hooks
            for hook in hooks:
                hook.remove()
                
            if not activations:
                return -float('inf')
                
            # Compute kernel matrix
            K = 0
            for act in activations:
                act = act.flatten(1)
                K += (act > 0).float() @ (act > 0).float().T
                
            K = K.numpy()
            
            # Compute log determinant
            _, logdet = np.linalg.slogdet(K + np.eye(K.shape[0]) * 1e-6)
            return logdet
            
        except Exception as e:
            for hook in hooks:
                hook.remove()
            return -float('inf')
            
    def compute_zen_score(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (32, 3, 32, 32),
        num_samples: int = 10
    ) -> float:
        """
        Compute Zen-Score (expressivity metric).
        Measures how much the network transforms Gaussian inputs.
        """
        model = model.to(self.device)
        model.eval()
        
        try:
            scores = []
            
            for _ in range(num_samples):
                # Random Gaussian input
                x = torch.randn(input_shape, device=self.device)
                
                with torch.no_grad():
                    output = model(x)
                    
                # Compute variance of output
                var = output.var().item()
                scores.append(np.log(var + 1e-10))
                
            return np.mean(scores)
            
        except Exception as e:
            return -float('inf')
            
    def compute_grad_norm(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_samples: int = 10
    ) -> float:
        """
        Compute average gradient norm at initialization.
        """
        model = model.to(self.device)
        model.train()
        
        criterion = nn.CrossEntropyLoss()
        grad_norms = []
        
        try:
            for i, (x, y) in enumerate(dataloader):
                if i >= num_samples:
                    break
                    
                x, y = x.to(self.device), y.to(self.device)
                
                model.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                
                # Compute gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm(2).item() ** 2
                grad_norms.append(np.sqrt(total_norm))
                
            if not grad_norms:
                return -float('inf')
                
            return np.mean(grad_norms)
            
        except Exception as e:
            return -float('inf')
            
    def compute_all_metrics(
        self,
        model: nn.Module,
        dataloader: Optional[DataLoader] = None,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32)
    ) -> Dict[str, float]:
        """Compute all zero-shot metrics."""
        metrics = {}
        
        metrics['synflow'] = self.compute_synflow(model, input_shape)
        metrics['zen_score'] = self.compute_zen_score(model, (32,) + input_shape[1:])
        
        if dataloader is not None:
            metrics['jacob_cov'] = self.compute_jacob_cov(model, dataloader)
            metrics['naswot'] = self.compute_naswot(model, dataloader)
            metrics['grad_norm'] = self.compute_grad_norm(model, dataloader)
        else:
            metrics['jacob_cov'] = 0.0
            metrics['naswot'] = 0.0
            metrics['grad_norm'] = 0.0
            
        # Composite score (weighted average)
        valid_scores = [v for v in metrics.values() if v > -float('inf')]
        metrics['composite'] = np.mean(valid_scores) if valid_scores else -float('inf')
        
        return metrics


# =============================================================================
# ARCHITECTURE EVALUATOR
# =============================================================================

class ArchitectureEvaluator:
    """
    Evaluates architectures using different methods:
    1. Minimal training (2-5 epochs)
    2. Zero-shot metrics
    3. Full training with early stopping
    """
    
    def __init__(
        self,
        dataset_manager: DatasetManager,
        device: str = 'cuda',
        grad_clip: float = 1.0,
        logger: Optional[ExperimentLogger] = None
    ):
        self.dataset_manager = dataset_manager
        self.device = device
        self.grad_clip = grad_clip
        self.logger = logger
        self.zero_shot = ZeroShotMetrics(device)
        
    def evaluate_minimal(
        self,
        arch_spec: ArchitectureSpec,
        epochs: int = 3,
        subset_size: int = 5000
    ) -> Dict[str, float]:
        """
        Minimal training evaluation (2-5 epochs on subset).
        Fast but gives rough estimate of architecture quality.
        """
        try:
            # Build model
            model = NASNetwork(arch_spec).to(self.device)
            
            # Get data
            train_loader = self.dataset_manager.get_dataset(
                DatasetType.CIFAR10, train=True, subset_size=subset_size
            )
            val_loader = self.dataset_manager.get_dataset(
                DatasetType.CIFAR10, train=False, subset_size=1000
            )
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                running_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    loss = criterion(output, y)
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
                    
            accuracy = correct / total
            
            # Cleanup
            del model
            cleanup_memory()
            
            return {
                'accuracy': accuracy,
                'val_loss': val_loss / len(val_loader),
                'num_params': NASNetwork(arch_spec).get_num_params(),
                'eval_method': 'minimal',
                'epochs': epochs
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Minimal eval failed: {str(e)}")
            cleanup_memory()
            return {
                'accuracy': 0.0,
                'val_loss': float('inf'),
                'num_params': 0,
                'eval_method': 'minimal',
                'error': str(e)
            }
            
    def evaluate_zero_shot(
        self,
        arch_spec: ArchitectureSpec,
        subset_size: int = 256
    ) -> Dict[str, float]:
        """
        Zero-shot evaluation using proxy metrics.
        Very fast, no training required.
        """
        try:
            # Build model
            model = NASNetwork(arch_spec).to(self.device)
            
            # Get small data batch for metrics that need it
            train_loader = self.dataset_manager.get_dataset(
                DatasetType.CIFAR10, train=True, subset_size=subset_size
            )
            
            # Compute metrics
            metrics = self.zero_shot.compute_all_metrics(
                model, train_loader, (1, 3, 32, 32)
            )
            
            # Add metadata
            metrics['eval_method'] = 'zero_shot'
            metrics['num_params'] = model.get_num_params()
            
            # Convert composite to "accuracy-like" score for comparison
            # Normalize to roughly 0-1 range
            composite = metrics['composite']
            if composite > -float('inf'):
                metrics['accuracy'] = 1 / (1 + np.exp(-composite / 10))
            else:
                metrics['accuracy'] = 0.0
                
            # Cleanup
            del model
            cleanup_memory()
            
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Zero-shot eval failed: {str(e)}")
            cleanup_memory()
            return {
                'accuracy': 0.0,
                'composite': -float('inf'),
                'eval_method': 'zero_shot',
                'error': str(e)
            }
            
    def evaluate_full(
        self,
        arch_spec: ArchitectureSpec,
        epochs: int = 25,
        patience: int = 5,
        subset_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Full training with early stopping.
        Most accurate but slowest evaluation method.
        """
        try:
            # Build model
            model = NASNetwork(arch_spec).to(self.device)
            
            # Get data
            train_loader = self.dataset_manager.get_dataset(
                DatasetType.CIFAR10, train=True, subset_size=subset_size
            )
            val_loader = self.dataset_manager.get_dataset(
                DatasetType.CIFAR10, train=False
            )
            
            # Training setup with cosine annealing
            optimizer = optim.SGD(
                model.parameters(), 
                lr=0.025, 
                momentum=0.9, 
                weight_decay=3e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            criterion = nn.CrossEntropyLoss()
            
            # Early stopping
            best_val_acc = 0.0
            best_epoch = 0
            patience_counter = 0
            history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
            
            # Training loop with progress bar
            epoch_iter = trange(epochs, desc="Training") if TQDM_AVAILABLE else range(epochs)
            
            for epoch in epoch_iter:
                # Training phase
                model.train()
                running_loss = 0.0
                
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                avg_train_loss = running_loss / len(train_loader)
                history['train_loss'].append(avg_train_loss)
                
                # Validation phase
                model.eval()
                correct = 0
                total = 0
                val_loss = 0.0
                
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        output = model(x)
                        loss = criterion(output, y)
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        total += y.size(0)
                        correct += predicted.eq(y).sum().item()
                        
                val_acc = correct / total
                avg_val_loss = val_loss / len(val_loader)
                history['val_acc'].append(val_acc)
                history['val_loss'].append(avg_val_loss)
                
                # Early stopping check
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if self.logger:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
                scheduler.step()
                
                if TQDM_AVAILABLE:
                    epoch_iter.set_postfix({
                        'val_acc': f'{val_acc:.4f}',
                        'best': f'{best_val_acc:.4f}'
                    })
                    
            # Cleanup
            del model
            cleanup_memory()
            
            return {
                'accuracy': best_val_acc,
                'best_epoch': best_epoch,
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
                'num_params': NASNetwork(arch_spec).get_num_params(),
                'eval_method': 'full',
                'epochs_trained': len(history['train_loss']),
                'history': history
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Full eval failed: {str(e)}")
            cleanup_memory()
            return {
                'accuracy': 0.0,
                'eval_method': 'full',
                'error': str(e)
            }


# =============================================================================
# SMART MUTATION WITH CURRICULUM LEARNING
# =============================================================================

class SmartMutator:
    """
    Intelligent mutation operator with curriculum learning.
    Adapts mutation probability and type based on search progress.
    """
    
    def __init__(
        self,
        max_nodes: int = 8,
        max_edges: int = 16,
        initial_mutation_rate: float = 0.3,
        min_mutation_rate: float = 0.1,
        curriculum_decay: float = 0.995
    ):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.mutation_rate = initial_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.curriculum_decay = curriculum_decay
        
        # Mutation statistics
        self.mutation_counts = defaultdict(int)
        self.mutation_successes = defaultdict(int)
        
        # Available operations
        self.operations = [
            OperationType.CONV,
            OperationType.TRANSFORMER,
            OperationType.RECURRENT,
            OperationType.IDENTITY,
            OperationType.POOLING
        ]
        
        self.cell_types = [CellType.DAG, CellType.DARTS, CellType.HYBRID]
        
    def mutate(self, arch_spec: ArchitectureSpec) -> ArchitectureSpec:
        """
        Apply smart mutation to architecture.
        Returns new mutated architecture.
        """
        # Deep copy
        new_spec = copy.deepcopy(arch_spec)
        
        # Select mutation type based on success rates
        mutation_type = self._select_mutation_type()
        
        try:
            if mutation_type == 'add_edge':
                new_spec = self._mutate_add_edge(new_spec)
            elif mutation_type == 'remove_edge':
                new_spec = self._mutate_remove_edge(new_spec)
            elif mutation_type == 'change_operation':
                new_spec = self._mutate_change_operation(new_spec)
            elif mutation_type == 'change_cell_type':
                new_spec = self._mutate_change_cell_type(new_spec)
            elif mutation_type == 'add_node':
                new_spec = self._mutate_add_node(new_spec)
            elif mutation_type == 'modify_weight':
                new_spec = self._mutate_modify_weight(new_spec)
            elif mutation_type == 'cross_paradigm':
                new_spec = self._mutate_cross_paradigm(new_spec)
                
            self.mutation_counts[mutation_type] += 1
            
        except Exception as e:
            # Mutation failed - return original
            return arch_spec
            
        # Update ID
        new_spec.arch_id = new_spec._generate_id()
        return new_spec
        
    def _select_mutation_type(self) -> str:
        """Select mutation type based on historical success rates."""
        mutation_types = [
            'add_edge', 'remove_edge', 'change_operation',
            'change_cell_type', 'add_node', 'modify_weight',
            'cross_paradigm'
        ]
        
        # Compute weights based on success rates
        weights = []
        for mt in mutation_types:
            count = self.mutation_counts[mt]
            successes = self.mutation_successes[mt]
            if count > 0:
                # Thompson sampling inspired
                weight = (successes + 1) / (count + 2)
            else:
                weight = 0.5  # Prior
            weights.append(weight)
            
        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return np.random.choice(mutation_types, p=weights)
        
    def _mutate_add_edge(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Add a new edge to the architecture."""
        if len(spec.edges) >= self.max_edges:
            return spec
            
        # Find valid edge positions
        existing = {(e.src_node, e.dst_node) for e in spec.edges}
        valid_positions = []
        
        for src in range(spec.num_nodes):
            for dst in range(src + 1, spec.num_nodes):
                if (src, dst) not in existing:
                    valid_positions.append((src, dst))
                    
        if not valid_positions:
            return spec
            
        # Random position and operation
        src, dst = random.choice(valid_positions)
        op_type = random.choice(self.operations)
        
        new_edge = EdgeSpec(
            src_node=src,
            dst_node=dst,
            operation=OperationSpec(op_type),
            weight=random.uniform(0.5, 1.0)
        )
        
        spec.edges.append(new_edge)
        return spec
        
    def _mutate_remove_edge(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Remove an edge from the architecture."""
        if len(spec.edges) <= 1:
            return spec
            
        idx = random.randint(0, len(spec.edges) - 1)
        spec.edges.pop(idx)
        return spec
        
    def _mutate_change_operation(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Change the operation type of an edge."""
        if not spec.edges:
            return spec
            
        idx = random.randint(0, len(spec.edges) - 1)
        new_op_type = random.choice(self.operations)
        
        # Preserve some parameters, change operation type
        old_spec = spec.edges[idx].operation
        spec.edges[idx].operation = OperationSpec(
            op_type=new_op_type,
            kernel_size=old_spec.kernel_size if random.random() > 0.5 else random.choice([3, 5, 7]),
            num_heads=old_spec.num_heads if random.random() > 0.5 else random.choice([1, 2, 4, 8]),
            dropout=old_spec.dropout
        )
        
        return spec
        
    def _mutate_change_cell_type(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Change the cell type."""
        current_idx = self.cell_types.index(spec.cell_type)
        new_idx = (current_idx + random.choice([1, 2])) % len(self.cell_types)
        spec.cell_type = self.cell_types[new_idx]
        return spec
        
    def _mutate_add_node(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Add a new node to the architecture."""
        if spec.num_nodes >= self.max_nodes:
            return spec
            
        new_node = spec.num_nodes
        spec.num_nodes += 1
        
        # Connect to existing nodes
        if spec.edges:
            # Connect from random existing node
            src = random.randint(0, new_node - 1)
            new_edge = EdgeSpec(
                src_node=src,
                dst_node=new_node,
                operation=OperationSpec(random.choice(self.operations)),
                weight=1.0
            )
            spec.edges.append(new_edge)
            
        return spec
        
    def _mutate_modify_weight(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """Modify edge weight (for continuous relaxation)."""
        if not spec.edges:
            return spec
            
        idx = random.randint(0, len(spec.edges) - 1)
        # Gaussian perturbation
        new_weight = spec.edges[idx].weight + random.gauss(0, 0.1)
        spec.edges[idx].weight = max(0.0, min(1.0, new_weight))
        return spec
        
    def _mutate_cross_paradigm(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """
        Cross-paradigm mutation: convert operation types to create
        architectures that span multiple paradigms.
        """
        if not spec.edges:
            return spec
            
        # Get current operation distribution
        op_counts = spec.get_operation_counts()
        
        # Find underrepresented paradigm
        paradigms = {
            'conv': [OperationType.CONV, OperationType.POOLING],
            'transformer': [OperationType.TRANSFORMER],
            'recurrent': [OperationType.RECURRENT]
        }
        
        paradigm_counts = {
            p: sum(op_counts.get(op, 0) for op in ops)
            for p, ops in paradigms.items()
        }
        
        # Add operation from underrepresented paradigm
        min_paradigm = min(paradigm_counts, key=paradigm_counts.get)
        target_ops = paradigms[min_paradigm]
        
        # Change a random edge to this paradigm
        idx = random.randint(0, len(spec.edges) - 1)
        spec.edges[idx].operation.op_type = random.choice(target_ops)
        
        return spec
        
    def record_success(self, mutation_type: str, improved: bool):
        """Record whether a mutation was successful."""
        if improved:
            self.mutation_successes[mutation_type] += 1
            
    def step_curriculum(self):
        """Step the curriculum (decay mutation rate)."""
        self.mutation_rate = max(
            self.min_mutation_rate,
            self.mutation_rate * self.curriculum_decay
        )


# =============================================================================
# MAP-ELITES IMPLEMENTATION
# =============================================================================

class MAPElitesGrid:
    """
    MAP-Elites grid for quality-diversity optimization.
    Stores elite architectures indexed by behavior descriptors.
    """
    
    def __init__(
        self,
        behavior_dims: int = 5,
        resolution: int = 10,
        behavior_bounds: Optional[List[Tuple[float, float]]] = None
    ):
        self.behavior_dims = behavior_dims
        self.resolution = resolution
        
        # Default bounds for each behavior dimension
        if behavior_bounds is None:
            behavior_bounds = [(0.0, 1.0)] * behavior_dims
        self.behavior_bounds = behavior_bounds
        
        # Grid storage: cell_idx -> (fitness, architecture)
        self.grid: Dict[Tuple[int, ...], Tuple[float, ArchitectureSpec]] = {}
        
        # Statistics
        self.total_evaluations = 0
        self.improvements = 0
        
    def _get_cell_index(self, behavior: Tuple[float, ...]) -> Tuple[int, ...]:
        """Convert behavior descriptor to grid cell index."""
        indices = []
        for i, (b, (low, high)) in enumerate(zip(behavior, self.behavior_bounds)):
            # Normalize to [0, 1]
            normalized = (b - low) / (high - low + 1e-10)
            normalized = max(0.0, min(1.0, normalized))
            # Convert to grid index
            idx = int(normalized * (self.resolution - 1))
            indices.append(idx)
        return tuple(indices)
        
    def add(
        self,
        arch_spec: ArchitectureSpec,
        fitness: float,
        behavior: Optional[Tuple[float, ...]] = None
    ) -> bool:
        """
        Add architecture to grid.
        Returns True if it's a new elite (improvement or new cell).
        """
        if behavior is None:
            behavior = arch_spec.get_behavior_descriptor()
            
        cell_idx = self._get_cell_index(behavior)
        self.total_evaluations += 1
        
        # Check if cell is empty or new fitness is better
        if cell_idx not in self.grid or fitness > self.grid[cell_idx][0]:
            self.grid[cell_idx] = (fitness, arch_spec)
            self.improvements += 1
            return True
            
        return False
        
    def sample_elite(self) -> Optional[ArchitectureSpec]:
        """Sample a random elite from the grid."""
        if not self.grid:
            return None
        cell_idx = random.choice(list(self.grid.keys()))
        return self.grid[cell_idx][1]
        
    def sample_elites(self, n: int) -> List[ArchitectureSpec]:
        """Sample n elites (with replacement if necessary)."""
        if not self.grid:
            return []
        indices = random.choices(list(self.grid.keys()), k=n)
        return [self.grid[idx][1] for idx in indices]
        
    def get_top_k(self, k: int) -> List[Tuple[float, ArchitectureSpec]]:
        """Get top k architectures by fitness."""
        sorted_elites = sorted(
            self.grid.values(),
            key=lambda x: x[0],
            reverse=True
        )
        return sorted_elites[:k]
        
    def get_coverage(self) -> float:
        """Get fraction of grid cells that are filled."""
        total_cells = self.resolution ** self.behavior_dims
        return len(self.grid) / total_cells
        
    def get_qd_score(self) -> float:
        """Get QD-score (sum of all fitnesses)."""
        return sum(fitness for fitness, _ in self.grid.values())
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get grid statistics."""
        if not self.grid:
            return {
                'num_elites': 0,
                'coverage': 0.0,
                'qd_score': 0.0,
                'max_fitness': 0.0,
                'mean_fitness': 0.0
            }
            
        fitnesses = [f for f, _ in self.grid.values()]
        return {
            'num_elites': len(self.grid),
            'coverage': self.get_coverage(),
            'qd_score': self.get_qd_score(),
            'max_fitness': max(fitnesses),
            'mean_fitness': np.mean(fitnesses),
            'total_evaluations': self.total_evaluations,
            'improvements': self.improvements
        }
        
    def save(self, filepath: str):
        """Save grid to file."""
        data = {
            'behavior_dims': self.behavior_dims,
            'resolution': self.resolution,
            'behavior_bounds': self.behavior_bounds,
            'grid': {
                str(k): (v[0], v[1].to_dict())
                for k, v in self.grid.items()
            },
            'total_evaluations': self.total_evaluations,
            'improvements': self.improvements
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'MAPElitesGrid':
        """Load grid from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        grid = cls(
            behavior_dims=data['behavior_dims'],
            resolution=data['resolution'],
            behavior_bounds=data['behavior_bounds']
        )
        
        grid.total_evaluations = data['total_evaluations']
        grid.improvements = data['improvements']
        
        for k, (fitness, arch_dict) in data['grid'].items():
            cell_idx = tuple(map(int, k.strip('()').split(', ')))
            arch_spec = ArchitectureSpec.from_dict(arch_dict)
            grid.grid[cell_idx] = (fitness, arch_spec)
            
        return grid


class MAPElitesOptimizer:
    """
    Main MAP-Elites optimizer integrating all components.
    """
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        mutator: SmartMutator,
        encoder: Optional[TextToSpecEncoder] = None,
        grid_resolution: int = 10,
        behavior_dims: int = 5,
        logger: Optional[ExperimentLogger] = None,
        device: str = 'cuda'
    ):
        self.evaluator = evaluator
        self.mutator = mutator
        self.encoder = encoder
        self.logger = logger
        self.device = device
        
        # Initialize grid
        self.grid = MAPElitesGrid(
            behavior_dims=behavior_dims,
            resolution=grid_resolution
        )
        
        # Evaluation mode
        self.eval_mode = 'minimal'  # 'minimal', 'zero_shot', 'full'
        
    def set_eval_mode(self, mode: str):
        """Set evaluation mode: 'minimal', 'zero_shot', or 'full'."""
        assert mode in ['minimal', 'zero_shot', 'full']
        self.eval_mode = mode
        
    def _evaluate(self, arch_spec: ArchitectureSpec) -> Dict[str, float]:
        """Evaluate architecture using current mode."""
        if self.eval_mode == 'minimal':
            return self.evaluator.evaluate_minimal(arch_spec)
        elif self.eval_mode == 'zero_shot':
            return self.evaluator.evaluate_zero_shot(arch_spec)
        else:
            return self.evaluator.evaluate_full(arch_spec)
            
    def generate_random_architecture(
        self,
        input_channels: int = 3,
        num_classes: int = 10
    ) -> ArchitectureSpec:
        """Generate a random architecture."""
        cell_type = random.choice(list(CellType))
        num_nodes = random.randint(3, CONFIG.max_nodes)
        
        # Generate random edges
        edges = []
        num_edges = random.randint(2, min(CONFIG.max_edges, num_nodes * 2))
        
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 2)
            dst = random.randint(src + 1, num_nodes - 1)
            
            op_type = random.choice(list(OperationType))
            op_spec = OperationSpec(
                op_type=op_type,
                kernel_size=random.choice([3, 5, 7]),
                num_heads=random.choice([1, 2, 4]),
                dropout=random.uniform(0.0, 0.2)
            )
            
            edge = EdgeSpec(
                src_node=src,
                dst_node=dst,
                operation=op_spec,
                weight=random.uniform(0.5, 1.0)
            )
            edges.append(edge)
            
        return ArchitectureSpec(
            cell_type=cell_type,
            num_nodes=num_nodes,
            edges=edges,
            input_channels=input_channels,
            num_classes=num_classes
        )
        
    def generate_from_text(self, text: str) -> ArchitectureSpec:
        """Generate architecture from text description."""
        if self.encoder is None:
            # Fallback to random
            return self.generate_random_architecture()
        return self.encoder.generate_architecture(text)
        
    def initialize_population(
        self,
        size: int = 50,
        text_descriptions: Optional[List[str]] = None
    ):
        """Initialize population with random and/or text-generated architectures."""
        if self.logger:
            self.logger.info(f"Initializing population with {size} architectures")
            
        init_iter = trange(size, desc="Initializing") if TQDM_AVAILABLE else range(size)
        
        for i in init_iter:
            try:
                # Generate architecture
                if text_descriptions and i < len(text_descriptions):
                    arch = self.generate_from_text(text_descriptions[i])
                else:
                    arch = self.generate_random_architecture()
                    
                # Evaluate
                results = self._evaluate(arch)
                fitness = results.get('accuracy', 0.0)
                
                # Add to grid
                behavior = arch.get_behavior_descriptor()
                improved = self.grid.add(arch, fitness, behavior)
                
                if self.logger and improved:
                    self.logger.log_architecture(
                        arch.arch_id, arch.to_dict(), fitness, behavior
                    )
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Init failed for arch {i}: {str(e)}")
                continue
                
        if self.logger:
            stats = self.grid.get_statistics()
            self.logger.info(f"Initialization complete: {stats}")
            
    def run_iteration(self) -> Dict[str, Any]:
        """Run single MAP-Elites iteration."""
        # Sample parent
        parent = self.grid.sample_elite()
        if parent is None:
            parent = self.generate_random_architecture()
            
        # Mutate
        child = self.mutator.mutate(parent)
        
        # Evaluate
        try:
            results = self._evaluate(child)
            fitness = results.get('accuracy', 0.0)
            
            # Add to grid
            behavior = child.get_behavior_descriptor()
            improved = self.grid.add(child, fitness, behavior)
            
            # Update mutation statistics
            if improved:
                self.mutator.record_success('mutation', True)
                
            # Step curriculum
            self.mutator.step_curriculum()
            
            return {
                'fitness': fitness,
                'improved': improved,
                'behavior': behavior,
                'arch_id': child.arch_id
            }
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Iteration failed: {str(e)}")
            return {'fitness': 0.0, 'improved': False, 'error': str(e)}
            
    def run(
        self,
        num_iterations: int = 100,
        checkpoint_interval: int = 20
    ) -> Dict[str, Any]:
        """Run full MAP-Elites optimization."""
        if self.logger:
            self.logger.info(f"Starting MAP-Elites with {num_iterations} iterations")
            self.logger.info(f"Evaluation mode: {self.eval_mode}")
            
        iter_range = trange(num_iterations, desc="MAP-Elites") if TQDM_AVAILABLE else range(num_iterations)
        
        for iteration in iter_range:
            result = self.run_iteration()
            
            # Log metrics periodically
            if iteration % 10 == 0:
                stats = self.grid.get_statistics()
                if self.logger:
                    self.logger.log_metric(iteration, stats)
                    
                if TQDM_AVAILABLE:
                    iter_range.set_postfix({
                        'elites': stats['num_elites'],
                        'max_fit': f"{stats['max_fitness']:.4f}",
                        'qd': f"{stats['qd_score']:.2f}"
                    })
                    
            # Checkpoint
            if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                self.save_checkpoint(iteration)
                
        # Final statistics
        final_stats = self.grid.get_statistics()
        if self.logger:
            self.logger.info(f"Optimization complete: {final_stats}")
            
        return final_stats
        
    def save_checkpoint(self, iteration: int):
        """Save checkpoint locally and to Google Drive with full state."""
        import shutil
        
        # Save grid
        grid_path = os.path.join(
            CONFIG.checkpoint_dir,
            f"mapelites_iter{iteration}.json"
        )
        self.grid.save(grid_path)
        
        # Save full optimizer state for resumability
        state = {
            'iteration': iteration,
            'eval_mode': self.eval_mode,
            'grid_resolution': self.grid.resolution,
            'mutator_stats': self.mutator.get_stats() if hasattr(self.mutator, 'get_stats') else {},
            'timestamp': datetime.now().isoformat(),
        }
        
        state_path = os.path.join(
            CONFIG.checkpoint_dir,
            f"optimizer_state_iter{iteration}.json"
        )
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save "latest" symlink/copy for easy resumption
        latest_grid = os.path.join(CONFIG.checkpoint_dir, "mapelites_latest.json")
        latest_state = os.path.join(CONFIG.checkpoint_dir, "optimizer_state_latest.json")
        shutil.copy(grid_path, latest_grid)
        shutil.copy(state_path, latest_state)
        
        if self.logger:
            self.logger.log_checkpoint(grid_path)
        
        # Backup to Google Drive if enabled
        if CONFIG.gdrive_backup and CONFIG.gdrive_path:
            try:
                gdrive_grid = os.path.join(CONFIG.gdrive_path, f"mapelites_iter{iteration}.json")
                gdrive_state = os.path.join(CONFIG.gdrive_path, f"optimizer_state_iter{iteration}.json")
                gdrive_latest_grid = os.path.join(CONFIG.gdrive_path, "mapelites_latest.json")
                gdrive_latest_state = os.path.join(CONFIG.gdrive_path, "optimizer_state_latest.json")
                
                shutil.copy(grid_path, gdrive_grid)
                shutil.copy(state_path, gdrive_state)
                shutil.copy(grid_path, gdrive_latest_grid)
                shutil.copy(state_path, gdrive_latest_state)
                
                if self.logger:
                    self.logger.info(f"☁️ Backed up to Drive: iter {iteration}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Drive backup failed: {e}")
            
    def load_checkpoint(self, filepath: str) -> int:
        """Load from checkpoint and return the iteration number."""
        self.grid = MAPElitesGrid.load(filepath)
        
        # Try to load optimizer state
        state_path = filepath.replace("mapelites_", "optimizer_state_")
        start_iteration = 0
        
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                start_iteration = state.get('iteration', 0)
                if 'eval_mode' in state:
                    self.eval_mode = state['eval_mode']
                if self.logger:
                    self.logger.info(f"Restored optimizer state from iter {start_iteration}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not load optimizer state: {e}")
        
        if self.logger:
            self.logger.info(f"Loaded checkpoint: {filepath}")
            
        return start_iteration
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str = None, gdrive_path: str = None) -> Optional[str]:
        """Find the latest checkpoint file for resumption."""
        search_dirs = []
        
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            search_dirs.append(checkpoint_dir)
        if gdrive_path and os.path.exists(gdrive_path):
            search_dirs.append(gdrive_path)
        if not search_dirs:
            search_dirs = [CONFIG.checkpoint_dir]
            if CONFIG.gdrive_path:
                search_dirs.append(CONFIG.gdrive_path)
        
        latest_file = None
        latest_iter = -1
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            # First check for "latest" file
            latest_path = os.path.join(search_dir, "mapelites_latest.json")
            if os.path.exists(latest_path):
                return latest_path
            
            # Otherwise find highest iteration
            import glob
            pattern = os.path.join(search_dir, "mapelites_iter*.json")
            for f in glob.glob(pattern):
                try:
                    # Extract iteration number
                    basename = os.path.basename(f)
                    iter_num = int(basename.replace("mapelites_iter", "").replace(".json", ""))
                    if iter_num > latest_iter:
                        latest_iter = iter_num
                        latest_file = f
                except ValueError:
                    continue
        
        return latest_file
    
    def run_with_resume(
        self,
        num_iterations: int = 100,
        checkpoint_interval: int = 20,
        resume: bool = True,
        checkpoint_path: str = None
    ) -> Dict[str, Any]:
        """Run MAP-Elites with automatic checkpoint resumption."""
        start_iteration = 0
        
        # Try to resume from checkpoint
        if resume:
            if checkpoint_path and os.path.exists(checkpoint_path):
                start_iteration = self.load_checkpoint(checkpoint_path)
            else:
                latest = self.find_latest_checkpoint()
                if latest:
                    start_iteration = self.load_checkpoint(latest)
                    if self.logger:
                        self.logger.info(f"🔄 Auto-resuming from iteration {start_iteration}")
        
        if start_iteration > 0:
            if self.logger:
                self.logger.info(f"Resuming from iteration {start_iteration}")
                stats = self.grid.get_statistics()
                self.logger.info(f"Archive state: {stats['num_elites']} elites, QD={stats['qd_score']:.2f}")
        else:
            if self.logger:
                self.logger.info(f"Starting fresh MAP-Elites with {num_iterations} iterations")
        
        # Setup graceful interrupt handling
        import signal
        interrupted = False
        
        def handle_interrupt(signum, frame):
            nonlocal interrupted
            if self.logger:
                self.logger.warning("⚠️ Interrupt received - saving checkpoint before exit...")
            interrupted = True
        
        old_handler = signal.signal(signal.SIGINT, handle_interrupt)
        
        try:
            if self.logger:
                self.logger.info(f"Evaluation mode: {self.eval_mode}")
                
            remaining = num_iterations - start_iteration
            iter_range = trange(remaining, desc="MAP-Elites") if TQDM_AVAILABLE else range(remaining)
            
            for i, _ in enumerate(iter_range):
                iteration = start_iteration + i
                
                if interrupted:
                    self.save_checkpoint(iteration)
                    if self.logger:
                        self.logger.info(f"💾 Checkpoint saved at iteration {iteration}. Exiting gracefully.")
                    break
                
                result = self.run_iteration()
                
                # Log metrics periodically
                if iteration % 10 == 0:
                    stats = self.grid.get_statistics()
                    if self.logger:
                        self.logger.log_metric(iteration, stats)
                        
                    if TQDM_AVAILABLE:
                        iter_range.set_postfix({
                            'elites': stats['num_elites'],
                            'max_fit': f"{stats['max_fitness']:.4f}",
                            'qd': f"{stats['qd_score']:.2f}"
                        })
                        
                # Checkpoint
                if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                    self.save_checkpoint(iteration)
                    
        finally:
            signal.signal(signal.SIGINT, old_handler)
                    
        # Final save
        self.save_checkpoint(num_iterations)
        
        # Final statistics
        final_stats = self.grid.get_statistics()
        if self.logger:
            self.logger.info(f"Optimization complete: {final_stats}")
            
        return final_stats
            
    def get_top_architectures(self, k: int = 4) -> List[ArchitectureSpec]:
        """Get top k architectures by fitness."""
        top_k = self.grid.get_top_k(k)
        return [arch for _, arch in top_k]


# =============================================================================
# PARALLEL EVALUATION
# =============================================================================

class ParallelEvaluator:
    """Parallel architecture evaluation for speedup."""
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        num_workers: int = 4
    ):
        self.evaluator = evaluator
        self.num_workers = num_workers
        
    def evaluate_batch(
        self,
        architectures: List[ArchitectureSpec],
        mode: str = 'minimal'
    ) -> List[Dict[str, float]]:
        """Evaluate batch of architectures in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for GPU operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if mode == 'minimal':
                futures = [
                    executor.submit(self.evaluator.evaluate_minimal, arch)
                    for arch in architectures
                ]
            elif mode == 'zero_shot':
                futures = [
                    executor.submit(self.evaluator.evaluate_zero_shot, arch)
                    for arch in architectures
                ]
            else:
                futures = [
                    executor.submit(self.evaluator.evaluate_full, arch)
                    for arch in architectures
                ]
                
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'accuracy': 0.0, 'error': str(e)})
                    
        return results


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_architecture(
    arch_spec: ArchitectureSpec,
    model: Optional[nn.Module] = None,
    output_dir: str = './output',
    prefix: str = 'arch'
):
    """
    Save architecture to JSON and model weights to PTH.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save architecture specification
    json_path = os.path.join(output_dir, f"{prefix}_{arch_spec.arch_id}.json")
    with open(json_path, 'w') as f:
        json.dump(arch_spec.to_dict(), f, indent=2)
        
    # Save model weights if provided
    if model is not None:
        pth_path = os.path.join(output_dir, f"{prefix}_{arch_spec.arch_id}.pth")
        torch.save({
            'state_dict': model.state_dict(),
            'arch_spec': arch_spec.to_dict(),
            'num_params': model.get_num_params()
        }, pth_path)
        
    return json_path


def load_architecture(json_path: str) -> Tuple[ArchitectureSpec, Optional[nn.Module]]:
    """
    Load architecture from JSON and optionally model weights.
    """
    with open(json_path, 'r') as f:
        arch_dict = json.load(f)
    arch_spec = ArchitectureSpec.from_dict(arch_dict)
    
    # Try to load model weights
    pth_path = json_path.replace('.json', '.pth')
    model = None
    
    if os.path.exists(pth_path):
        model = NASNetwork(arch_spec)
        checkpoint = torch.load(pth_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        
    return arch_spec, model


if __name__ == "__main__":
    print("NAG-ME-QD Module 2 loaded successfully")
    print("Components: MAPElitesGrid, MAPElitesOptimizer, ZeroShotMetrics")
