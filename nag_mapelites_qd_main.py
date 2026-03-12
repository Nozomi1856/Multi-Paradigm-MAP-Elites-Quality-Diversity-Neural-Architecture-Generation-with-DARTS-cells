#!/usr/bin/env python3
"""
Neural Architecture Generation with MAP-Elites Quality-Diversity (NAG-ME-QD)
Part 3: Main Execution - Three Versions Pipeline
============================================================================

This script runs the complete NAG-ME-QD pipeline with three evaluation versions:
1. Minimal Training (2-5 epochs)
2. Zero-Shot Evaluation
3. Full Training with Early Stopping

Finally, it trains the top 4 architectures for 25 epochs.
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# Import from modules
from nag_mapelites_qd import (
    CellType, OperationType, DatasetType,
    OperationSpec, EdgeSpec, ArchitectureSpec,
    NASNetwork, TextToSpecEncoder, TextToSpecEncoderTrainer,
    DatasetManager, ExperimentLogger,
    set_seed, cleanup_memory, CONFIG, GlobalConfig
)

from nag_mapelites_qd_part2 import (
    ZeroShotMetrics, ArchitectureEvaluator,
    SmartMutator, MAPElitesGrid, MAPElitesOptimizer,
    ParallelEvaluator, save_architecture, load_architecture
)


# =============================================================================
# EXPERIMENTAL BASELINES
# =============================================================================

class RandomSearchBaseline:
    """Random search baseline for comparison."""
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.evaluator = evaluator
        self.logger = logger
        self.best_arch = None
        self.best_fitness = 0.0
        self.history = []
        
    def generate_random_architecture(self) -> ArchitectureSpec:
        """Generate random architecture."""
        cell_type = random.choice(list(CellType))
        num_nodes = random.randint(3, CONFIG.max_nodes)
        
        edges = []
        num_edges = random.randint(2, min(CONFIG.max_edges, num_nodes * 2))
        
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 2)
            dst = random.randint(src + 1, num_nodes - 1)
            
            op_type = random.choice(list(OperationType))
            op_spec = OperationSpec(
                op_type=op_type,
                kernel_size=random.choice([3, 5, 7]),
                num_heads=random.choice([1, 2, 4])
            )
            
            edges.append(EdgeSpec(src_node=src, dst_node=dst, operation=op_spec))
            
        return ArchitectureSpec(
            cell_type=cell_type,
            num_nodes=num_nodes,
            edges=edges
        )
        
    def run(self, num_iterations: int, eval_mode: str = 'minimal') -> Dict:
        """Run random search."""
        if self.logger:
            self.logger.info(f"Running Random Search baseline ({num_iterations} iterations)")
            
        iter_range = trange(num_iterations, desc="Random Search") if TQDM_AVAILABLE else range(num_iterations)
        
        for i in iter_range:
            arch = self.generate_random_architecture()
            
            try:
                if eval_mode == 'minimal':
                    results = self.evaluator.evaluate_minimal(arch)
                elif eval_mode == 'zero_shot':
                    results = self.evaluator.evaluate_zero_shot(arch)
                else:
                    results = self.evaluator.evaluate_full(arch)
                    
                fitness = results.get('accuracy', 0.0)
                self.history.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_arch = arch
                    
                if TQDM_AVAILABLE:
                    iter_range.set_postfix({'best': f'{self.best_fitness:.4f}'})
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Random search iteration {i} failed: {e}")
                    
        return {
            'best_fitness': self.best_fitness,
            'best_arch': self.best_arch,
            'mean_fitness': np.mean(self.history) if self.history else 0.0,
            'num_evaluated': len(self.history)
        }


class RegularizedEvolutionBaseline:
    """Regularized evolution baseline (simplified)."""
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        population_size: int = 20,
        tournament_size: int = 5,
        logger: Optional[ExperimentLogger] = None
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.logger = logger
        self.mutator = SmartMutator()
        
        self.population: List[Tuple[float, ArchitectureSpec]] = []
        self.best_arch = None
        self.best_fitness = 0.0
        
    def _generate_random(self) -> ArchitectureSpec:
        """Generate random architecture."""
        cell_type = random.choice(list(CellType))
        num_nodes = random.randint(3, CONFIG.max_nodes)
        
        edges = []
        for _ in range(random.randint(2, CONFIG.max_edges)):
            src = random.randint(0, num_nodes - 2)
            dst = random.randint(src + 1, num_nodes - 1)
            edges.append(EdgeSpec(
                src_node=src, dst_node=dst,
                operation=OperationSpec(random.choice(list(OperationType)))
            ))
            
        return ArchitectureSpec(cell_type=cell_type, num_nodes=num_nodes, edges=edges)
        
    def run(self, num_iterations: int, eval_mode: str = 'minimal') -> Dict:
        """Run regularized evolution."""
        if self.logger:
            self.logger.info(f"Running Regularized Evolution ({num_iterations} iterations)")
            
        # Initialize population
        for _ in trange(self.population_size, desc="Init Population") if TQDM_AVAILABLE else range(self.population_size):
            arch = self._generate_random()
            try:
                if eval_mode == 'minimal':
                    results = self.evaluator.evaluate_minimal(arch)
                elif eval_mode == 'zero_shot':
                    results = self.evaluator.evaluate_zero_shot(arch)
                else:
                    results = self.evaluator.evaluate_full(arch)
                fitness = results.get('accuracy', 0.0)
                self.population.append((fitness, arch))
            except:
                pass
                
        # Evolution loop
        iter_range = trange(num_iterations, desc="Evolution") if TQDM_AVAILABLE else range(num_iterations)
        
        for i in iter_range:
            # Tournament selection
            tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
            parent = max(tournament, key=lambda x: x[0])[1]
            
            # Mutation
            child = self.mutator.mutate(parent)
            
            try:
                if eval_mode == 'minimal':
                    results = self.evaluator.evaluate_minimal(child)
                elif eval_mode == 'zero_shot':
                    results = self.evaluator.evaluate_zero_shot(child)
                else:
                    results = self.evaluator.evaluate_full(child)
                fitness = results.get('accuracy', 0.0)
                
                # Update best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_arch = child
                    
                # Add to population, remove oldest
                self.population.append((fitness, child))
                if len(self.population) > self.population_size:
                    self.population.pop(0)
                    
                if TQDM_AVAILABLE:
                    iter_range.set_postfix({'best': f'{self.best_fitness:.4f}'})
                    
            except Exception as e:
                pass
                
        return {
            'best_fitness': self.best_fitness,
            'best_arch': self.best_arch,
            'population_size': len(self.population)
        }


class GridSearchBaseline:
    """
    Grid Search Baseline - Classic hyperparameter search over architecture space.
    
    Systematically explores predefined combinations of:
    - Cell types (DAG, DARTS, HYBRID)
    - Number of nodes (3-6)
    - Operation types (CONV, TRANSFORMER, RECURRENT)
    - Kernel sizes (3, 5, 7)
    """
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.evaluator = evaluator
        self.logger = logger
        self.best_arch = None
        self.best_fitness = 0.0
        self.all_results: List[Dict] = []
        
        # Define search grid
        self.cell_types = list(CellType)
        self.num_nodes_options = [3, 4, 5, 6]
        self.op_types = [OperationType.CONV, OperationType.TRANSFORMER, OperationType.RECURRENT]
        self.kernel_sizes = [3, 5, 7]
        self.num_edges_options = [3, 5, 7]
        
    def _generate_grid_architecture(
        self,
        cell_type: CellType,
        num_nodes: int,
        dominant_op: OperationType,
        kernel_size: int,
        num_edges: int
    ) -> ArchitectureSpec:
        """Generate architecture from grid parameters."""
        edges = []
        
        for i in range(num_edges):
            src = i % (num_nodes - 1)
            dst = min(src + 1 + (i // (num_nodes - 1)), num_nodes - 1)
            
            # Mix dominant op with others
            if i < num_edges * 0.7:  # 70% dominant operation
                op_type = dominant_op
            else:
                op_type = random.choice(self.op_types)
                
            op_spec = OperationSpec(
                op_type=op_type,
                kernel_size=kernel_size if op_type == OperationType.CONV else 3,
                num_heads=4 if op_type == OperationType.TRANSFORMER else 1
            )
            
            edges.append(EdgeSpec(src_node=src, dst_node=dst, operation=op_spec))
            
        return ArchitectureSpec(
            cell_type=cell_type,
            num_nodes=num_nodes,
            edges=edges
        )
        
    def run(self, num_iterations: int = None, eval_mode: str = 'minimal') -> Dict:
        """
        Run grid search. num_iterations limits total evaluations if provided.
        """
        if self.logger:
            self.logger.info("Running Grid Search baseline")
            
        # Generate all grid combinations
        from itertools import product
        
        grid_combinations = list(product(
            self.cell_types,
            self.num_nodes_options,
            self.op_types,
            self.kernel_sizes,
            self.num_edges_options
        ))
        
        # Limit if num_iterations specified
        if num_iterations and num_iterations < len(grid_combinations):
            grid_combinations = random.sample(grid_combinations, num_iterations)
            
        if self.logger:
            self.logger.info(f"Grid search: {len(grid_combinations)} configurations")
            
        iter_range = tqdm(grid_combinations, desc="Grid Search") if TQDM_AVAILABLE else grid_combinations
        
        for cell_type, num_nodes, dominant_op, kernel_size, num_edges in iter_range:
            arch = self._generate_grid_architecture(
                cell_type, num_nodes, dominant_op, kernel_size, num_edges
            )
            
            try:
                if eval_mode == 'minimal':
                    results = self.evaluator.evaluate_minimal(arch)
                elif eval_mode == 'zero_shot':
                    results = self.evaluator.evaluate_zero_shot(arch)
                else:
                    results = self.evaluator.evaluate_full(arch)
                    
                fitness = results.get('accuracy', 0.0)
                
                self.all_results.append({
                    'cell_type': cell_type.name,
                    'num_nodes': num_nodes,
                    'dominant_op': dominant_op.name,
                    'kernel_size': kernel_size,
                    'num_edges': num_edges,
                    'fitness': fitness
                })
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_arch = arch
                    
                if TQDM_AVAILABLE:
                    iter_range.set_postfix({'best': f'{self.best_fitness:.4f}'})
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Grid search config failed: {e}")
                    
        return {
            'best_fitness': self.best_fitness,
            'best_arch': self.best_arch,
            'num_evaluated': len(self.all_results),
            'all_results': self.all_results
        }


class DARTSGradientBaseline:
    """
    DARTS-like Gradient-Based One-Shot NAS Baseline.
    
    Creates a supernet with architecture weights (alpha) and optimizes:
    - Network weights via standard backprop on training data
    - Architecture weights via gradient descent on validation data
    
    This is a simplified version of DARTS (Differentiable Architecture Search).
    """
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        num_nodes: int = 5,
        logger: Optional[ExperimentLogger] = None
    ):
        self.evaluator = evaluator
        self.num_nodes = num_nodes
        self.logger = logger
        self.best_arch = None
        self.best_fitness = 0.0
        self.device = CONFIG.device
        
        # Operations to choose from
        self.ops_list = [
            OperationType.CONV,
            OperationType.TRANSFORMER,
            OperationType.RECURRENT,
            OperationType.IDENTITY,
            OperationType.POOLING
        ]
        self.num_ops = len(self.ops_list)
        
    def _create_supernet(self) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create supernet with architecture parameters.
        Returns (supernet, alpha_params)
        """
        # Create a DARTS cell architecture with all edges
        edges = []
        edge_idx = 0
        
        for dst in range(2, self.num_nodes):
            for src in range(dst):
                # Add edge with mixed operation (will be weighted by alpha)
                edges.append(EdgeSpec(
                    src_node=src,
                    dst_node=dst,
                    operation=OperationSpec(OperationType.CONV),  # Placeholder
                    weight=1.0 / (dst)  # Initial uniform weight
                ))
                edge_idx += 1
                
        arch_spec = ArchitectureSpec(
            cell_type=CellType.DARTS,
            num_nodes=self.num_nodes,
            edges=edges,
            num_cells=2  # Smaller supernet
        )
        
        # Number of edges
        num_edges = len(edges)
        
        # Architecture parameters: alpha[edge][op]
        alpha = nn.Parameter(torch.zeros(num_edges, self.num_ops))
        nn.init.normal_(alpha, 0, 0.001)
        
        # Build supernet
        supernet = NASNetwork(arch_spec).to(self.device)
        
        return supernet, alpha, arch_spec, num_edges
        
    def _derive_architecture(self, alpha: nn.Parameter, base_arch: ArchitectureSpec) -> ArchitectureSpec:
        """Derive discrete architecture from continuous alpha."""
        # Softmax over operations for each edge
        weights = torch.softmax(alpha, dim=-1)
        
        # Select best operation per edge
        best_ops = torch.argmax(weights, dim=-1)
        
        new_edges = []
        for i, edge in enumerate(base_arch.edges):
            op_idx = best_ops[i].item()
            op_type = self.ops_list[op_idx]
            
            # Only keep edges with non-trivial operations
            if op_type not in [OperationType.ZERO]:
                new_edges.append(EdgeSpec(
                    src_node=edge.src_node,
                    dst_node=edge.dst_node,
                    operation=OperationSpec(
                        op_type=op_type,
                        kernel_size=3 if op_type == OperationType.CONV else 1,
                        num_heads=4 if op_type == OperationType.TRANSFORMER else 1
                    ),
                    weight=float(weights[i, op_idx].item())
                ))
                
        return ArchitectureSpec(
            cell_type=CellType.DARTS,
            num_nodes=base_arch.num_nodes,
            edges=new_edges if new_edges else base_arch.edges[:3]  # Fallback
        )
        
    def run(self, num_iterations: int = 50, eval_mode: str = 'minimal') -> Dict:
        """
        Run DARTS-like search.
        
        Args:
            num_iterations: Number of alternating optimization steps
            eval_mode: Evaluation mode for final architecture
        """
        if self.logger:
            self.logger.info(f"Running DARTS Gradient-Based baseline ({num_iterations} iterations)")
            
        # Create supernet
        supernet, alpha, base_arch, num_edges = self._create_supernet()
        
        # Get data
        dataset_manager = DatasetManager(batch_size=32)
        train_loader = dataset_manager.get_dataset(DatasetType.CIFAR10, train=True, subset_size=5000)
        val_loader = dataset_manager.get_dataset(DatasetType.CIFAR10, train=False, subset_size=1000)
        
        # Optimizers
        weight_optimizer = optim.SGD(supernet.parameters(), lr=0.01, momentum=0.9, weight_decay=3e-4)
        alpha_optimizer = optim.Adam([alpha], lr=3e-4, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_loss': [], 'alpha_entropy': []}
        
        iter_range = trange(num_iterations, desc="DARTS Search") if TQDM_AVAILABLE else range(num_iterations)
        
        for iteration in iter_range:
            supernet.train()
            
            # Step 1: Update network weights on training data
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 10:  # Limit batches per iteration
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                weight_optimizer.zero_grad()
                output = supernet(data)
                loss = criterion(output, target)
                loss.backward()
                clip_grad_norm_(supernet.parameters(), 1.0)
                weight_optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
            # Step 2: Update architecture weights on validation data
            val_loss = 0.0
            val_batches = 0
            
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 5:  # Limit batches
                    break
                    
                data, target = data.to(self.device), target.to(self.device)
                
                alpha_optimizer.zero_grad()
                
                # Forward with soft architecture weights
                output = supernet(data)
                loss = criterion(output, target)
                
                # Add entropy regularization to encourage discrete choices
                probs = torch.softmax(alpha, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                loss = loss - 0.01 * entropy  # Minimize entropy
                
                loss.backward()
                alpha_optimizer.step()
                
                val_loss += loss.item()
                val_batches += 1
                
            # Record history
            history['train_loss'].append(train_loss / max(num_batches, 1))
            history['val_loss'].append(val_loss / max(val_batches, 1))
            
            # Compute alpha entropy
            probs = torch.softmax(alpha, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
            history['alpha_entropy'].append(entropy)
            
            if TQDM_AVAILABLE:
                iter_range.set_postfix({
                    'train_loss': f'{history["train_loss"][-1]:.4f}',
                    'entropy': f'{entropy:.2f}'
                })
                
        # Derive final architecture
        derived_arch = self._derive_architecture(alpha, base_arch)
        
        # Evaluate derived architecture
        try:
            if eval_mode == 'minimal':
                results = self.evaluator.evaluate_minimal(derived_arch)
            elif eval_mode == 'zero_shot':
                results = self.evaluator.evaluate_zero_shot(derived_arch)
            else:
                results = self.evaluator.evaluate_full(derived_arch)
                
            self.best_fitness = results.get('accuracy', 0.0)
            self.best_arch = derived_arch
        except Exception as e:
            if self.logger:
                self.logger.error(f"DARTS final evaluation failed: {e}")
                
        # Cleanup
        del supernet
        cleanup_memory()
        
        return {
            'best_fitness': self.best_fitness,
            'best_arch': self.best_arch,
            'history': history,
            'final_alpha': alpha.detach().cpu().numpy().tolist()
        }


class LocalSearchBaseline:
    """
    Local Search (Hill-Climbing) Baseline.
    
    Starts from a random architecture and iteratively explores neighbors,
    accepting improvements. Includes restarts to escape local optima.
    """
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        num_restarts: int = 5,
        logger: Optional[ExperimentLogger] = None
    ):
        self.evaluator = evaluator
        self.num_restarts = num_restarts
        self.logger = logger
        self.best_arch = None
        self.best_fitness = 0.0
        self.search_history: List[Dict] = []
        
    def _generate_random_arch(self) -> ArchitectureSpec:
        """Generate random starting architecture."""
        cell_type = random.choice(list(CellType))
        num_nodes = random.randint(3, CONFIG.max_nodes)
        
        edges = []
        for _ in range(random.randint(3, 6)):
            src = random.randint(0, num_nodes - 2)
            dst = random.randint(src + 1, num_nodes - 1)
            edges.append(EdgeSpec(
                src_node=src, dst_node=dst,
                operation=OperationSpec(random.choice(list(OperationType)))
            ))
            
        return ArchitectureSpec(cell_type=cell_type, num_nodes=num_nodes, edges=edges)
        
    def _get_neighbors(self, arch: ArchitectureSpec, num_neighbors: int = 5) -> List[ArchitectureSpec]:
        """Generate neighboring architectures via small modifications."""
        neighbors = []
        
        for _ in range(num_neighbors):
            # Deep copy architecture
            new_edges = [EdgeSpec(
                src_node=e.src_node,
                dst_node=e.dst_node,
                operation=OperationSpec(
                    op_type=e.operation.op_type,
                    kernel_size=e.operation.kernel_size,
                    num_heads=e.operation.num_heads
                ),
                weight=e.weight
            ) for e in arch.edges]
            
            # Apply one of several neighbor operations
            neighbor_type = random.choice([
                'change_op', 'change_kernel', 'add_edge', 'remove_edge', 'change_cell'
            ])
            
            if neighbor_type == 'change_op' and new_edges:
                # Change operation type of random edge
                idx = random.randint(0, len(new_edges) - 1)
                new_op = random.choice(list(OperationType))
                new_edges[idx] = EdgeSpec(
                    src_node=new_edges[idx].src_node,
                    dst_node=new_edges[idx].dst_node,
                    operation=OperationSpec(op_type=new_op),
                    weight=new_edges[idx].weight
                )
                
            elif neighbor_type == 'change_kernel' and new_edges:
                # Change kernel size
                idx = random.randint(0, len(new_edges) - 1)
                new_kernel = random.choice([3, 5, 7])
                new_edges[idx] = EdgeSpec(
                    src_node=new_edges[idx].src_node,
                    dst_node=new_edges[idx].dst_node,
                    operation=OperationSpec(
                        op_type=new_edges[idx].operation.op_type,
                        kernel_size=new_kernel,
                        num_heads=new_edges[idx].operation.num_heads
                    ),
                    weight=new_edges[idx].weight
                )
                
            elif neighbor_type == 'add_edge' and len(new_edges) < CONFIG.max_edges:
                # Add new edge
                src = random.randint(0, arch.num_nodes - 2)
                dst = random.randint(src + 1, arch.num_nodes - 1)
                new_edges.append(EdgeSpec(
                    src_node=src, dst_node=dst,
                    operation=OperationSpec(random.choice(list(OperationType)))
                ))
                
            elif neighbor_type == 'remove_edge' and len(new_edges) > 2:
                # Remove random edge
                idx = random.randint(0, len(new_edges) - 1)
                new_edges.pop(idx)
                
            elif neighbor_type == 'change_cell':
                # Change cell type
                new_cell = random.choice(list(CellType))
                neighbors.append(ArchitectureSpec(
                    cell_type=new_cell,
                    num_nodes=arch.num_nodes,
                    edges=new_edges
                ))
                continue
                
            neighbors.append(ArchitectureSpec(
                cell_type=arch.cell_type,
                num_nodes=arch.num_nodes,
                edges=new_edges
            ))
            
        return neighbors
        
    def _evaluate(self, arch: ArchitectureSpec, eval_mode: str) -> float:
        """Evaluate architecture and return fitness."""
        try:
            if eval_mode == 'minimal':
                results = self.evaluator.evaluate_minimal(arch)
            elif eval_mode == 'zero_shot':
                results = self.evaluator.evaluate_zero_shot(arch)
            else:
                results = self.evaluator.evaluate_full(arch)
            return results.get('accuracy', 0.0)
        except:
            return 0.0
            
    def run(self, num_iterations: int = 100, eval_mode: str = 'minimal') -> Dict:
        """
        Run local search with restarts.
        
        Args:
            num_iterations: Total iterations across all restarts
            eval_mode: Evaluation mode
        """
        if self.logger:
            self.logger.info(f"Running Local Search baseline ({num_iterations} iterations, {self.num_restarts} restarts)")
            
        iterations_per_restart = num_iterations // self.num_restarts
        
        for restart in range(self.num_restarts):
            if self.logger:
                self.logger.info(f"  Restart {restart + 1}/{self.num_restarts}")
                
            # Start from random architecture
            current = self._generate_random_arch()
            current_fitness = self._evaluate(current, eval_mode)
            
            restart_best = current_fitness
            restart_best_arch = current
            
            iter_range = trange(iterations_per_restart, desc=f"Hill-Climb R{restart+1}") if TQDM_AVAILABLE else range(iterations_per_restart)
            
            no_improvement_count = 0
            
            for i in iter_range:
                # Get neighbors
                neighbors = self._get_neighbors(current)
                
                # Evaluate neighbors
                best_neighbor = None
                best_neighbor_fitness = current_fitness
                
                for neighbor in neighbors:
                    fitness = self._evaluate(neighbor, eval_mode)
                    if fitness > best_neighbor_fitness:
                        best_neighbor_fitness = fitness
                        best_neighbor = neighbor
                        
                # Accept if improvement found
                if best_neighbor is not None:
                    current = best_neighbor
                    current_fitness = best_neighbor_fitness
                    no_improvement_count = 0
                    
                    if current_fitness > restart_best:
                        restart_best = current_fitness
                        restart_best_arch = current
                else:
                    no_improvement_count += 1
                    
                # Early stop if stuck
                if no_improvement_count > 10:
                    break
                    
                if TQDM_AVAILABLE:
                    iter_range.set_postfix({'current': f'{current_fitness:.4f}', 'best': f'{restart_best:.4f}'})
                    
            # Record restart results
            self.search_history.append({
                'restart': restart,
                'best_fitness': restart_best,
                'iterations': i + 1
            })
            
            # Update global best
            if restart_best > self.best_fitness:
                self.best_fitness = restart_best
                self.best_arch = restart_best_arch
                
        return {
            'best_fitness': self.best_fitness,
            'best_arch': self.best_arch,
            'num_restarts': self.num_restarts,
            'search_history': self.search_history
        }


class NASBench301Baseline:
    """
    NAS-Bench-301 Baseline.
    
    Uses the NAS-Bench-301 surrogate benchmark for architecture evaluation.
    This provides predicted accuracy without actual training, based on a
    surrogate model trained on real NAS experiments.
    
    If NAS-Bench-301 is not available, falls back to zero-shot metrics.
    """
    
    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        logger: Optional[ExperimentLogger] = None
    ):
        self.evaluator = evaluator
        self.logger = logger
        self.best_arch = None
        self.best_fitness = 0.0
        self.history: List[Dict] = []
        
        # Try to load NAS-Bench-301
        self.nb301_available = False
        self.nb301_model = None
        
        try:
            import nasbench301 as nb301
            
            # Load surrogate model (performance model)
            models_dir = os.path.join(os.path.dirname(__file__), 'nb301_models')
            if os.path.exists(models_dir):
                self.nb301_model = nb301.load_ensemble(
                    os.path.join(models_dir, 'xgb_v1.0')
                )
                self.nb301_available = True
                if self.logger:
                    self.logger.info("NAS-Bench-301 loaded successfully")
            else:
                if self.logger:
                    self.logger.info("NAS-Bench-301 models not found, will use zero-shot fallback")
        except ImportError:
            if self.logger:
                self.logger.info("NAS-Bench-301 not installed, will use zero-shot fallback")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"NAS-Bench-301 loading failed: {e}")
                
    def _arch_to_genotype(self, arch: ArchitectureSpec) -> Optional[Any]:
        """
        Convert ArchitectureSpec to DARTS genotype format for NAS-Bench-301.
        
        NAS-Bench-301 expects DARTS-style genotypes with named operations.
        """
        try:
            from collections import namedtuple
            Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
            
            # Map our operations to DARTS operation names
            op_map = {
                OperationType.CONV: 'sep_conv_3x3',
                OperationType.TRANSFORMER: 'skip_connect',  # Closest analogy
                OperationType.RECURRENT: 'dil_conv_3x3',
                OperationType.IDENTITY: 'skip_connect',
                OperationType.ZERO: 'none',
                OperationType.POOLING: 'avg_pool_3x3'
            }
            
            # Build normal cell (simplified: use first 4 edges)
            normal = []
            for i, edge in enumerate(arch.edges[:4]):
                op_name = op_map.get(edge.operation.op_type, 'sep_conv_3x3')
                normal.append((op_name, edge.src_node))
                
            # Pad if needed
            while len(normal) < 8:
                normal.append(('sep_conv_3x3', 0))
                
            # Build reduce cell (same as normal for simplicity)
            reduce = list(normal)
            
            # Concat nodes
            normal_concat = list(range(2, 6))
            reduce_concat = list(range(2, 6))
            
            return Genotype(
                normal=normal[:8],
                normal_concat=normal_concat,
                reduce=reduce[:8],
                reduce_concat=reduce_concat
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Genotype conversion failed: {e}")
            return None
            
    def _evaluate_nb301(self, arch: ArchitectureSpec) -> float:
        """Evaluate architecture using NAS-Bench-301 surrogate."""
        if not self.nb301_available or self.nb301_model is None:
            return -1.0
            
        genotype = self._arch_to_genotype(arch)
        if genotype is None:
            return -1.0
            
        try:
            # Query the surrogate model
            prediction = self.nb301_model.predict(
                config=genotype,
                representation='genotype'
            )
            return float(prediction)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"NB301 prediction failed: {e}")
            return -1.0
            
    def _generate_random_arch(self) -> ArchitectureSpec:
        """Generate random DARTS-compatible architecture."""
        num_nodes = 6  # DARTS standard
        
        edges = []
        for dst in range(2, num_nodes):
            for src in range(dst):
                if random.random() < 0.5:  # Sparse connectivity
                    edges.append(EdgeSpec(
                        src_node=src,
                        dst_node=dst,
                        operation=OperationSpec(random.choice([
                            OperationType.CONV,
                            OperationType.IDENTITY,
                            OperationType.POOLING
                        ]))
                    ))
                    
        # Ensure minimum edges
        while len(edges) < 4:
            src = random.randint(0, 3)
            dst = random.randint(src + 1, 5)
            edges.append(EdgeSpec(
                src_node=src, dst_node=dst,
                operation=OperationSpec(OperationType.CONV)
            ))
            
        return ArchitectureSpec(
            cell_type=CellType.DARTS,
            num_nodes=num_nodes,
            edges=edges[:8]  # DARTS has 8 edges per cell
        )
        
    def run(self, num_iterations: int = 100, eval_mode: str = 'minimal') -> Dict:
        """
        Run NAS-Bench-301 search.
        
        Uses evolutionary search with NB301 predictions for fitness.
        Falls back to zero-shot if NB301 unavailable.
        """
        if self.logger:
            self.logger.info(f"Running NAS-Bench-301 baseline ({num_iterations} iterations)")
            self.logger.info(f"  NB301 available: {self.nb301_available}")
            
        # Initialize population
        population: List[Tuple[float, ArchitectureSpec]] = []
        pop_size = 20
        
        for _ in range(pop_size):
            arch = self._generate_random_arch()
            
            # Try NB301 first
            fitness = self._evaluate_nb301(arch)
            
            # Fallback to zero-shot
            if fitness < 0:
                try:
                    results = self.evaluator.evaluate_zero_shot(arch)
                    fitness = results.get('accuracy', 0.0)
                except:
                    fitness = 0.0
                    
            population.append((fitness, arch))
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_arch = arch
                
        # Evolutionary search
        mutator = SmartMutator()
        
        iter_range = trange(num_iterations, desc="NB301 Search") if TQDM_AVAILABLE else range(num_iterations)
        
        for i in iter_range:
            # Tournament selection
            tournament = random.sample(population, min(5, len(population)))
            parent = max(tournament, key=lambda x: x[0])[1]
            
            # Mutate
            child = mutator.mutate(parent)
            
            # Evaluate
            fitness = self._evaluate_nb301(child)
            if fitness < 0:
                try:
                    results = self.evaluator.evaluate_zero_shot(child)
                    fitness = results.get('accuracy', 0.0)
                except:
                    fitness = 0.0
                    
            # Update population
            population.append((fitness, child))
            population.sort(key=lambda x: x[0], reverse=True)
            population = population[:pop_size]
            
            # Update best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_arch = child
                
            self.history.append({
                'iteration': i,
                'fitness': fitness,
                'best_so_far': self.best_fitness
            })
            
            if TQDM_AVAILABLE:
                iter_range.set_postfix({'best': f'{self.best_fitness:.4f}'})
                
        # Final evaluation with actual training
        if self.best_arch is not None:
            try:
                if self.logger:
                    self.logger.info("Running final evaluation on best architecture...")
                if eval_mode == 'minimal':
                    results = self.evaluator.evaluate_minimal(self.best_arch)
                else:
                    results = self.evaluator.evaluate_zero_shot(self.best_arch)
                self.best_fitness = results.get('accuracy', self.best_fitness)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Final evaluation failed: {e}")
                    
        return {
            'best_fitness': self.best_fitness,
            'best_arch': self.best_arch,
            'nb301_available': self.nb301_available,
            'num_evaluated': len(self.history),
            'history': self.history
        }


# =============================================================================
# FINAL TRAINING
# =============================================================================

def train_architecture_full(
    arch_spec: ArchitectureSpec,
    epochs: int = 25,
    device: str = 'cuda',
    logger: Optional[ExperimentLogger] = None
) -> Tuple[nn.Module, Dict]:
    """
    Full training of a single architecture.
    Returns trained model and training history.
    """
    if logger:
        logger.info(f"Training architecture {arch_spec.arch_id} for {epochs} epochs")
        
    # Build model
    model = NASNetwork(arch_spec).to(device)
    
    # Get data
    dataset_manager = DatasetManager(batch_size=64)
    train_loader = dataset_manager.get_dataset(DatasetType.CIFAR10, train=True)
    val_loader = dataset_manager.get_dataset(DatasetType.CIFAR10, train=False)
    
    # Optimizer with cosine annealing
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=3e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_state = None
    
    epoch_iter = trange(epochs, desc=f"Training {arch_spec.arch_id[:8]}") if TQDM_AVAILABLE else range(epochs)
    
    for epoch in epoch_iter:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
                
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            
        scheduler.step()
        
        if TQDM_AVAILABLE:
            epoch_iter.set_postfix({
                'train_acc': f'{train_acc:.4f}',
                'val_acc': f'{val_acc:.4f}',
                'best': f'{best_val_acc:.4f}'
            })
            
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
        
    if logger:
        logger.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
        
    return model, history


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class NAGMEQDPipeline:
    """
    Complete NAG-ME-QD pipeline with three evaluation versions.
    Supports checkpoint resumption for long-running experiments.
    """
    
    def __init__(
        self,
        config: GlobalConfig = None,
        device: str = None,
        resume: bool = False,
        checkpoint_path: str = None
    ):
        self.config = config or CONFIG
        self.device = device or self.config.device
        self.resume = resume
        self.checkpoint_path = checkpoint_path
        
        # Set seed
        set_seed(self.config.seed)
        
        # Initialize components
        self.dataset_manager = DatasetManager(batch_size=32)
        
        # Initialize logger
        self.logger = ExperimentLogger("NAG_ME_QD", self.config)
        
        # Initialize evaluator
        self.evaluator = ArchitectureEvaluator(
            self.dataset_manager,
            device=self.device,
            logger=self.logger
        )
        
        # Initialize mutator
        self.mutator = SmartMutator()
        
        # Initialize text encoder
        self.encoder = TextToSpecEncoder().to(self.device)
        
        # Results storage
        self.results = {
            'version1_minimal': {},
            'version2_zeroshot': {},
            'version3_full': {},
            'baselines': {},
            'final_training': {}
        }
        
    def run_version1_minimal(
        self,
        num_iterations: int = 100,
        init_population: int = 30,
        epochs_per_eval: int = 3
    ) -> Dict:
        """
        Version 1: Minimal training (2-5 epochs per architecture).
        Fast exploration of architecture space.
        """
        self.logger.info("=" * 60)
        self.logger.info("VERSION 1: MINIMAL TRAINING EVALUATION")
        self.logger.info("=" * 60)
        
        optimizer = MAPElitesOptimizer(
            evaluator=self.evaluator,
            mutator=self.mutator,
            encoder=self.encoder,
            grid_resolution=self.config.grid_resolution,
            logger=self.logger,
            device=self.device
        )
        
        optimizer.set_eval_mode('minimal')
        
        # Text prompts for initialization
        text_prompts = [
            "deep convolutional network for image classification",
            "transformer attention architecture for vision",
            "hybrid conv transformer network",
            "recurrent network with LSTM for sequences",
            "shallow but wide convolutional model",
            "deep narrow network with skip connections",
            "multi-paradigm hybrid architecture",
            "efficient lightweight conv network"
        ]
        
        # Initialize (skip if resuming and archive has entries)
        if not self.resume or optimizer.grid.get_statistics()['num_elites'] == 0:
            optimizer.initialize_population(init_population, text_prompts)
        
        # Run optimization with resume support
        stats = optimizer.run_with_resume(
            num_iterations, 
            checkpoint_interval=25,
            resume=self.resume,
            checkpoint_path=self.checkpoint_path
        )
        
        # Get results
        top_archs = optimizer.get_top_architectures(10)
        
        self.results['version1_minimal'] = {
            'final_stats': stats,
            'top_architectures': [arch.to_dict() for arch in top_archs],
            'grid_coverage': optimizer.grid.get_coverage(),
            'qd_score': optimizer.grid.get_qd_score()
        }
        
        # Save grid
        grid_path = os.path.join(self.config.output_dir, "version1_grid.json")
        optimizer.grid.save(grid_path)
        
        self.logger.info(f"Version 1 complete. Top accuracy: {stats['max_fitness']:.4f}")
        
        return self.results['version1_minimal']
        
    def run_version2_zeroshot(
        self,
        num_iterations: int = 200,
        init_population: int = 50
    ) -> Dict:
        """
        Version 2: Zero-shot evaluation using proxy metrics.
        Very fast, allows massive exploration.
        """
        self.logger.info("=" * 60)
        self.logger.info("VERSION 2: ZERO-SHOT EVALUATION")
        self.logger.info("=" * 60)
        
        optimizer = MAPElitesOptimizer(
            evaluator=self.evaluator,
            mutator=self.mutator,
            encoder=self.encoder,
            grid_resolution=self.config.grid_resolution,
            logger=self.logger,
            device=self.device
        )
        
        optimizer.set_eval_mode('zero_shot')
        
        # Initialize with more architectures (faster evaluation)
        if not self.resume or optimizer.grid.get_statistics()['num_elites'] == 0:
            optimizer.initialize_population(init_population)
        
        # Run more iterations (each is faster) with resume support
        stats = optimizer.run_with_resume(
            num_iterations, 
            checkpoint_interval=50,
            resume=self.resume,
            checkpoint_path=self.checkpoint_path
        )
        
        # Get results
        top_archs = optimizer.get_top_architectures(10)
        
        self.results['version2_zeroshot'] = {
            'final_stats': stats,
            'top_architectures': [arch.to_dict() for arch in top_archs],
            'grid_coverage': optimizer.grid.get_coverage(),
            'qd_score': optimizer.grid.get_qd_score()
        }
        
        # Save grid
        grid_path = os.path.join(self.config.output_dir, "version2_grid.json")
        optimizer.grid.save(grid_path)
        
        self.logger.info(f"Version 2 complete. Top zero-shot score: {stats['max_fitness']:.4f}")
        
        return self.results['version2_zeroshot']
        
    def run_version3_full(
        self,
        num_iterations: int = 50,
        init_population: int = 20,
        patience: int = 5
    ) -> Dict:
        """
        Version 3: Full training with early stopping.
        Most accurate but slowest.
        """
        self.logger.info("=" * 60)
        self.logger.info("VERSION 3: FULL TRAINING WITH EARLY STOPPING")
        self.logger.info("=" * 60)
        
        optimizer = MAPElitesOptimizer(
            evaluator=self.evaluator,
            mutator=self.mutator,
            encoder=self.encoder,
            grid_resolution=self.config.grid_resolution,
            logger=self.logger,
            device=self.device
        )
        
        optimizer.set_eval_mode('full')
        
        # Smaller population (each eval is expensive)
        if not self.resume or optimizer.grid.get_statistics()['num_elites'] == 0:
            optimizer.initialize_population(init_population)
        
        # Fewer iterations with resume support
        stats = optimizer.run_with_resume(
            num_iterations, 
            checkpoint_interval=10,
            resume=self.resume,
            checkpoint_path=self.checkpoint_path
        )
        
        # Get results
        top_archs = optimizer.get_top_architectures(10)
        
        self.results['version3_full'] = {
            'final_stats': stats,
            'top_architectures': [arch.to_dict() for arch in top_archs],
            'grid_coverage': optimizer.grid.get_coverage(),
            'qd_score': optimizer.grid.get_qd_score()
        }
        
        # Save grid
        grid_path = os.path.join(self.config.output_dir, "version3_grid.json")
        optimizer.grid.save(grid_path)
        
        self.logger.info(f"Version 3 complete. Top accuracy: {stats['max_fitness']:.4f}")
        
        return self.results['version3_full']
        
    def run_baselines(self, num_iterations: int = 100, eval_mode: str = 'minimal') -> Dict:
        """
        Run all baseline comparisons.
        
        Baselines:
        1. Random Search - pure random architecture sampling
        2. Regularized Evolution - tournament selection + mutation
        3. Grid Search - systematic hyperparameter exploration
        4. DARTS Gradient-Based - one-shot NAS with architecture weights
        5. Local Search - hill-climbing with restarts
        6. NAS-Bench-301 - surrogate benchmark evaluation
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING ALL BASELINES (6 total)")
        self.logger.info("=" * 60)
        
        baseline_results = {}
        
        # 1. Random Search
        self.logger.info("\n[1/6] Random Search...")
        try:
            random_baseline = RandomSearchBaseline(self.evaluator, self.logger)
            random_results = random_baseline.run(num_iterations, eval_mode)
            baseline_results['random_search'] = random_results
            self.logger.info(f"  Best: {random_results['best_fitness']:.4f}")
        except Exception as e:
            self.logger.error(f"  Random Search failed: {e}")
            baseline_results['random_search'] = {'best_fitness': 0.0, 'error': str(e)}
        
        # 2. Regularized Evolution
        self.logger.info("\n[2/6] Regularized Evolution...")
        try:
            evo_baseline = RegularizedEvolutionBaseline(self.evaluator, logger=self.logger)
            evo_results = evo_baseline.run(num_iterations, eval_mode)
            baseline_results['regularized_evolution'] = evo_results
            self.logger.info(f"  Best: {evo_results['best_fitness']:.4f}")
        except Exception as e:
            self.logger.error(f"  RegEvo failed: {e}")
            baseline_results['regularized_evolution'] = {'best_fitness': 0.0, 'error': str(e)}
        
        # 3. Grid Search
        self.logger.info("\n[3/6] Grid Search...")
        try:
            grid_baseline = GridSearchBaseline(self.evaluator, self.logger)
            grid_results = grid_baseline.run(num_iterations, eval_mode)
            baseline_results['grid_search'] = grid_results
            self.logger.info(f"  Best: {grid_results['best_fitness']:.4f}")
            self.logger.info(f"  Configs evaluated: {grid_results.get('num_evaluated', 0)}")
        except Exception as e:
            self.logger.error(f"  Grid Search failed: {e}")
            baseline_results['grid_search'] = {'best_fitness': 0.0, 'error': str(e)}
        
        # 4. DARTS Gradient-Based
        self.logger.info("\n[4/6] DARTS Gradient-Based (One-Shot NAS)...")
        try:
            darts_baseline = DARTSGradientBaseline(self.evaluator, logger=self.logger)
            darts_results = darts_baseline.run(min(num_iterations // 2, 50), eval_mode)
            baseline_results['darts_gradient'] = darts_results
            self.logger.info(f"  Best: {darts_results['best_fitness']:.4f}")
        except Exception as e:
            self.logger.error(f"  DARTS failed: {e}")
            baseline_results['darts_gradient'] = {'best_fitness': 0.0, 'error': str(e)}
        
        # 5. Local Search (Hill-Climbing)
        self.logger.info("\n[5/6] Local Search (Hill-Climbing)...")
        try:
            local_baseline = LocalSearchBaseline(self.evaluator, num_restarts=5, logger=self.logger)
            local_results = local_baseline.run(num_iterations, eval_mode)
            baseline_results['local_search'] = local_results
            self.logger.info(f"  Best: {local_results['best_fitness']:.4f}")
            self.logger.info(f"  Restarts: {local_results.get('num_restarts', 0)}")
        except Exception as e:
            self.logger.error(f"  Local Search failed: {e}")
            baseline_results['local_search'] = {'best_fitness': 0.0, 'error': str(e)}
        
        # 6. NAS-Bench-301
        self.logger.info("\n[6/6] NAS-Bench-301...")
        try:
            nb301_baseline = NASBench301Baseline(self.evaluator, logger=self.logger)
            nb301_results = nb301_baseline.run(num_iterations, eval_mode)
            baseline_results['nasbench301'] = nb301_results
            self.logger.info(f"  Best: {nb301_results['best_fitness']:.4f}")
            self.logger.info(f"  NB301 available: {nb301_results.get('nb301_available', False)}")
        except Exception as e:
            self.logger.error(f"  NAS-Bench-301 failed: {e}")
            baseline_results['nasbench301'] = {'best_fitness': 0.0, 'error': str(e)}
        
        self.results['baselines'] = baseline_results
        
        # Summary
        self.logger.info("\n" + "-" * 40)
        self.logger.info("BASELINE SUMMARY")
        self.logger.info("-" * 40)
        
        sorted_baselines = sorted(
            [(name, res.get('best_fitness', 0.0)) for name, res in baseline_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (name, fitness) in enumerate(sorted_baselines, 1):
            self.logger.info(f"  {rank}. {name}: {fitness:.4f}")
        
        return self.results['baselines']
        
    def train_top_architectures(
        self,
        top_k: int = 4,
        epochs: int = 25
    ) -> Dict:
        """
        Train top k architectures from EACH version for full epochs.
        
        This trains:
        - Top 4 from Version 1 (minimal training)
        - Top 4 from Version 2 (zero-shot)
        - Top 4 from Version 3 (full training)
        Total: up to 12 architectures (may be fewer if duplicates exist)
        """
        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING TOP {top_k} ARCHITECTURES FROM EACH VERSION")
        self.logger.info(f"Epochs per architecture: {epochs}")
        self.logger.info("=" * 60)
        
        version_names = {
            'version1_minimal': 'v1_minimal',
            'version2_zeroshot': 'v2_zeroshot', 
            'version3_full': 'v3_full'
        }
        
        trained_models = []
        global_seen_ids = set()  # Track globally to avoid training exact duplicates
        global_rank = 0
        
        for version_key, version_short in version_names.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {version_key.upper()}")
            self.logger.info(f"{'='*60}")
            
            # Get top architectures from this version
            version_archs = []
            if version_key in self.results and 'top_architectures' in self.results[version_key]:
                for arch_dict in self.results[version_key]['top_architectures'][:top_k]:
                    arch = ArchitectureSpec.from_dict(arch_dict)
                    version_archs.append(arch)
            
            # If no architectures for this version, generate random ones
            if not version_archs:
                self.logger.warning(f"No architectures found for {version_key}, generating {top_k} random ones")
                for _ in range(top_k):
                    arch = self._generate_random_arch()
                    version_archs.append(arch)
            
            # Filter duplicates (within this version and globally)
            unique_version_archs = []
            for arch in version_archs:
                if arch.arch_id not in global_seen_ids and len(unique_version_archs) < top_k:
                    global_seen_ids.add(arch.arch_id)
                    unique_version_archs.append(arch)
            
            self.logger.info(f"Training {len(unique_version_archs)} unique architectures from {version_key}")
            
            # Create version-specific output directory
            version_output_dir = os.path.join(self.config.output_dir, version_short)
            os.makedirs(version_output_dir, exist_ok=True)
            
            # Train each architecture from this version
            for local_rank, arch in enumerate(unique_version_archs, 1):
                global_rank += 1
                
                self.logger.info(f"\n[{version_short}] Training architecture {local_rank}/{len(unique_version_archs)}")
                self.logger.info(f"  Global rank: {global_rank}")
                self.logger.info(f"  Arch ID: {arch.arch_id}")
                self.logger.info(f"  Cell type: {arch.cell_type.name}")
                self.logger.info(f"  Num nodes: {arch.num_nodes}")
                self.logger.info(f"  Num edges: {len(arch.edges)}")
                self.logger.info(f"  Dominant op: {arch.get_dominant_operation().name}")
                
                try:
                    model, history = train_architecture_full(
                        arch, epochs=epochs, device=self.device, logger=self.logger
                    )
                    
                    best_acc = max(history['val_acc']) if history['val_acc'] else 0.0
                    final_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
                    
                    # Save model and architecture
                    save_path = os.path.join(version_output_dir, f"top{local_rank}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save architecture JSON with full metadata
                    json_path = os.path.join(save_path, "architecture.json")
                    arch_data = arch.to_dict()
                    arch_data['training_metadata'] = {
                        'source_version': version_key,
                        'version_rank': local_rank,
                        'global_rank': global_rank,
                        'epochs_trained': epochs,
                        'best_val_acc': float(best_acc),
                        'final_val_acc': float(final_acc)
                    }
                    with open(json_path, 'w') as f:
                        json.dump(arch_data, f, indent=2)
                        
                    # Save model weights with comprehensive checkpoint
                    pth_path = os.path.join(save_path, "model.pth")
                    torch.save({
                        'state_dict': model.state_dict(),
                        'arch_spec': arch.to_dict(),
                        'history': {
                            'train_loss': history['train_loss'],
                            'val_loss': history['val_loss'],
                            'train_acc': history['train_acc'],
                            'val_acc': history['val_acc']
                        },
                        'best_val_acc': float(best_acc),
                        'final_val_acc': float(final_acc),
                        'num_params': model.get_num_params(),
                        'epochs_trained': epochs,
                        'source_version': version_key,
                        'version_rank': local_rank,
                        'global_rank': global_rank
                    }, pth_path)
                    
                    trained_models.append({
                        'global_rank': global_rank,
                        'version_rank': local_rank,
                        'source_version': version_key,
                        'version_short': version_short,
                        'arch_id': arch.arch_id,
                        'cell_type': arch.cell_type.name,
                        'num_nodes': arch.num_nodes,
                        'num_edges': len(arch.edges),
                        'dominant_op': arch.get_dominant_operation().name,
                        'best_val_acc': float(best_acc),
                        'final_val_acc': float(final_acc),
                        'num_params': model.get_num_params(),
                        'json_path': json_path,
                        'pth_path': pth_path
                    })
                    
                    self.logger.info(f"  Best validation accuracy: {best_acc:.4f}")
                    self.logger.info(f"  Final validation accuracy: {final_acc:.4f}")
                    self.logger.info(f"  Parameters: {model.get_num_params():,}")
                    self.logger.info(f"  Saved to: {save_path}")
                    
                    # Cleanup
                    del model
                    cleanup_memory()
                    
                except Exception as e:
                    self.logger.error(f"Training failed for arch {arch.arch_id}: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    continue
        
        # Sort all trained models by best accuracy
        trained_models_sorted = sorted(trained_models, key=lambda x: x['best_val_acc'], reverse=True)
        
        # Log final summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("TRAINING COMPLETE - FINAL RANKINGS")
        self.logger.info(f"{'='*60}")
        for i, model_info in enumerate(trained_models_sorted, 1):
            self.logger.info(
                f"  #{i}: {model_info['version_short']}/top{model_info['version_rank']} "
                f"({model_info['cell_type']}, {model_info['dominant_op']}) "
                f"- Acc: {model_info['best_val_acc']:.4f}"
            )
                
        self.results['final_training'] = {
            'trained_models': trained_models,
            'trained_models_ranked': trained_models_sorted,
            'epochs': epochs,
            'top_k_per_version': top_k,
            'total_trained': len(trained_models),
            'versions_processed': list(version_names.keys())
        }
        
        return self.results['final_training']
        
    def _generate_random_arch(self) -> ArchitectureSpec:
        """Generate random architecture for fallback."""
        cell_type = random.choice(list(CellType))
        num_nodes = random.randint(4, 6)
        edges = []
        
        for _ in range(random.randint(3, 8)):
            src = random.randint(0, num_nodes - 2)
            dst = random.randint(src + 1, num_nodes - 1)
            edges.append(EdgeSpec(
                src_node=src, dst_node=dst,
                operation=OperationSpec(random.choice(list(OperationType)))
            ))
            
        return ArchitectureSpec(cell_type=cell_type, num_nodes=num_nodes, edges=edges)
        
    def run_full_pipeline(
        self,
        quick_mode: bool = False
    ) -> Dict:
        """
        Run complete pipeline with all versions.
        
        This executes:
        1. Version 1: Minimal Training (3 epochs per architecture)
        2. Version 2: Zero-Shot (no training, proxy metrics only)
        3. Version 3: Full Training (25 epochs with early stopping)
        4. Baselines: Random Search and Regularized Evolution
        5. Final Training: Top 4 from EACH version (12 total) for 25 epochs
        
        Args:
            quick_mode: If True, use reduced iterations for testing
            
        Output Structure:
            ./output_dir/
                v1_minimal/top{1-4}/architecture.json + model.pth
                v2_zeroshot/top{1-4}/architecture.json + model.pth
                v3_full/top{1-4}/architecture.json + model.pth
                version{1-3}_grid.json
                final_results.json
        """
        start_time = time.time()
        
        if quick_mode:
            # Quick mode for testing
            v1_iters, v2_iters, v3_iters = 20, 30, 10
            init_pop = 10
            final_epochs = 5
        else:
            # Full mode
            v1_iters, v2_iters, v3_iters = 100, 200, 50
            init_pop = 30
            final_epochs = 25
            
        self.logger.info("=" * 70)
        self.logger.info("NEURAL ARCHITECTURE GENERATION WITH MAP-ELITES QUALITY-DIVERSITY")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Quick mode: {quick_mode}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        
        # Version 1: Minimal Training
        self.logger.info("\n[1/5] Running Version 1: Minimal Training...")
        v1_results = self.run_version1_minimal(
            num_iterations=v1_iters,
            init_population=init_pop
        )
        
        # Version 2: Zero-Shot
        self.logger.info("\n[2/5] Running Version 2: Zero-Shot...")
        v2_results = self.run_version2_zeroshot(
            num_iterations=v2_iters,
            init_population=init_pop * 2
        )
        
        # Version 3: Full Training
        self.logger.info("\n[3/5] Running Version 3: Full Training...")
        v3_results = self.run_version3_full(
            num_iterations=v3_iters,
            init_population=init_pop // 2
        )
        
        # Baselines
        self.logger.info("\n[4/5] Running Baselines...")
        baseline_results = self.run_baselines(
            num_iterations=v1_iters,
            eval_mode='minimal'
        )
        
        # Final Training - Top 4 from EACH version = 12 architectures total
        self.logger.info("\n[5/5] Training Top 4 Architectures from EACH Version (up to 12 total)...")
        final_results = self.train_top_architectures(
            top_k=4,
            epochs=final_epochs
        )
        
        # Save complete results
        total_time = time.time() - start_time
        self.results['metadata'] = {
            'total_time_seconds': total_time,
            'device': self.device,
            'quick_mode': quick_mode,
            'config': {
                'seed': self.config.seed,
                'grid_resolution': self.config.grid_resolution,
                'max_nodes': self.config.max_nodes,
                'max_edges': self.config.max_edges
            }
        }
        
        # Save final results
        results_path = os.path.join(self.config.output_dir, "final_results.json")
        
        # Make results JSON serializable
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Backup to Google Drive if enabled
        if self.config.gdrive_backup and self.config.gdrive_path:
            import shutil
            try:
                # Copy final results
                gdrive_results = os.path.join(self.config.gdrive_path, "final_results.json")
                shutil.copy(results_path, gdrive_results)
                
                # Copy entire output directory for complete backup
                gdrive_output = os.path.join(self.config.gdrive_path, "full_output")
                if os.path.exists(gdrive_output):
                    shutil.rmtree(gdrive_output)
                shutil.copytree(self.config.output_dir, gdrive_output)
                
                self.logger.info(f"☁️ Full results backed up to Google Drive: {self.config.gdrive_path}")
            except Exception as e:
                self.logger.warning(f"Drive backup failed: {e}")
            
        # Save experiment summary
        self.logger.save_summary()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def run_single_version(
        self,
        version: str,
        quick_mode: bool = False,
        top_k: int = 4,
        final_epochs: int = 25,
        num_iterations: Optional[int] = None
    ) -> Dict:
        """
        Run a single version of the pipeline with its own final training.
        
        Args:
            version: Which version to run ('v1', 'v2', or 'v3')
            quick_mode: If True, use reduced iterations
            top_k: Number of top architectures to train
            final_epochs: Number of epochs for final training
            num_iterations: Override default iteration count
            
        Output Structure:
            ./output_dir/
                {version_short}/top{1-k}/architecture.json + model.pth
                {version}_grid.json
                {version}_results.json
        """
        import time
        start_time = time.time()
        
        # Version configuration
        version_config = {
            'v1': {
                'name': 'version1_minimal',
                'short': 'v1_minimal',
                'default_iters': 100 if not quick_mode else 20,
                'init_pop': 30 if not quick_mode else 10,
                'run_func': self.run_version1_minimal
            },
            'v2': {
                'name': 'version2_zeroshot', 
                'short': 'v2_zeroshot',
                'default_iters': 200 if not quick_mode else 30,
                'init_pop': 60 if not quick_mode else 20,
                'run_func': self.run_version2_zeroshot
            },
            'v3': {
                'name': 'version3_full',
                'short': 'v3_full', 
                'default_iters': 50 if not quick_mode else 10,
                'init_pop': 15 if not quick_mode else 5,
                'run_func': self.run_version3_full
            }
        }
        
        if version not in version_config:
            raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3'")
            
        cfg = version_config[version]
        iterations = num_iterations if num_iterations else cfg['default_iters']
        
        self.logger.info("=" * 70)
        self.logger.info(f"RUNNING SINGLE VERSION: {cfg['name'].upper()}")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Quick mode: {quick_mode}")
        self.logger.info(f"Iterations: {iterations}")
        self.logger.info(f"Top-k: {top_k}")
        self.logger.info(f"Final epochs: {final_epochs}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        
        # Step 1: Run MAP-Elites search for this version
        self.logger.info(f"\n[1/2] Running MAP-Elites Search ({cfg['name']})...")
        cfg['run_func'](
            num_iterations=iterations,
            init_population=cfg['init_pop']
        )
        
        # Step 2: Train top-k architectures from THIS version only
        self.logger.info(f"\n[2/2] Training Top {top_k} Architectures for {final_epochs} Epochs...")
        trained_models = self._train_single_version_top_k(
            version_key=cfg['name'],
            version_short=cfg['short'],
            top_k=top_k,
            epochs=final_epochs
        )
        
        # Save results
        total_time = time.time() - start_time
        self.results['metadata'] = {
            'version': version,
            'version_name': cfg['name'],
            'total_time_seconds': total_time,
            'device': self.device,
            'quick_mode': quick_mode,
            'iterations': iterations,
            'top_k': top_k,
            'final_epochs': final_epochs
        }
        
        # Save version-specific results
        results_path = os.path.join(self.config.output_dir, f"{cfg['short']}_results.json")
        serializable_results = self._make_serializable(self.results)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.save_summary()
        self._print_single_version_summary(cfg['name'], trained_models, total_time)
        
        return self.results
    
    def _train_single_version_top_k(
        self,
        version_key: str,
        version_short: str,
        top_k: int,
        epochs: int
    ) -> List[Dict]:
        """Train top-k architectures from a single version."""
        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING TOP {top_k} FROM {version_key.upper()}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info("=" * 60)
        
        # Get architectures from this version
        version_archs = []
        if version_key in self.results and 'top_architectures' in self.results[version_key]:
            for arch_dict in self.results[version_key]['top_architectures'][:top_k]:
                arch = ArchitectureSpec.from_dict(arch_dict)
                version_archs.append(arch)
        
        if not version_archs:
            self.logger.warning(f"No architectures found for {version_key}, generating {top_k} random")
            for _ in range(top_k):
                version_archs.append(self._generate_random_arch())
        
        # Create output directory
        version_output_dir = os.path.join(self.config.output_dir, version_short)
        os.makedirs(version_output_dir, exist_ok=True)
        
        trained_models = []
        
        for rank, arch in enumerate(version_archs[:top_k], 1):
            self.logger.info(f"\n[{rank}/{min(top_k, len(version_archs))}] Training architecture")
            self.logger.info(f"  Arch ID: {arch.arch_id}")
            self.logger.info(f"  Cell type: {arch.cell_type.name}")
            self.logger.info(f"  Num nodes: {arch.num_nodes}")
            self.logger.info(f"  Num edges: {len(arch.edges)}")
            self.logger.info(f"  Dominant op: {arch.get_dominant_operation().name}")
            
            try:
                model, history = train_architecture_full(
                    arch, epochs=epochs, device=self.device, logger=self.logger
                )
                
                best_acc = max(history['val_acc']) if history['val_acc'] else 0.0
                final_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
                
                # Save
                save_path = os.path.join(version_output_dir, f"top{rank}")
                os.makedirs(save_path, exist_ok=True)
                
                # Architecture JSON
                json_path = os.path.join(save_path, "architecture.json")
                arch_data = arch.to_dict()
                arch_data['training_metadata'] = {
                    'source_version': version_key,
                    'rank': rank,
                    'epochs_trained': epochs,
                    'best_val_acc': float(best_acc),
                    'final_val_acc': float(final_acc)
                }
                with open(json_path, 'w') as f:
                    json.dump(arch_data, f, indent=2)
                
                # Model PTH
                pth_path = os.path.join(save_path, "model.pth")
                torch.save({
                    'state_dict': model.state_dict(),
                    'arch_spec': arch.to_dict(),
                    'history': {
                        'train_loss': history['train_loss'],
                        'val_loss': history['val_loss'],
                        'train_acc': history['train_acc'],
                        'val_acc': history['val_acc']
                    },
                    'best_val_acc': float(best_acc),
                    'final_val_acc': float(final_acc),
                    'num_params': model.get_num_params(),
                    'epochs_trained': epochs,
                    'source_version': version_key,
                    'rank': rank
                }, pth_path)
                
                trained_models.append({
                    'rank': rank,
                    'arch_id': arch.arch_id,
                    'cell_type': arch.cell_type.name,
                    'num_nodes': arch.num_nodes,
                    'num_edges': len(arch.edges),
                    'dominant_op': arch.get_dominant_operation().name,
                    'best_val_acc': float(best_acc),
                    'final_val_acc': float(final_acc),
                    'num_params': model.get_num_params(),
                    'json_path': json_path,
                    'pth_path': pth_path
                })
                
                self.logger.info(f"  Best accuracy: {best_acc:.4f}")
                self.logger.info(f"  Saved to: {save_path}")
                
                del model
                cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"Training failed: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        # Store results
        self.results['final_training'] = {
            'version': version_key,
            'trained_models': trained_models,
            'epochs': epochs,
            'top_k': top_k
        }
        
        return trained_models
    
    def _print_single_version_summary(
        self,
        version_key: str,
        trained_models: List[Dict],
        total_time: float
    ):
        """Print summary for single version run."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"SUMMARY: {version_key.upper()}")
        self.logger.info("=" * 70)
        
        # Search stats
        if version_key in self.results and 'final_stats' in self.results[version_key]:
            stats = self.results[version_key]['final_stats']
            self.logger.info(f"\nMAP-Elites Search:")
            self.logger.info(f"  Elites discovered: {stats.get('num_elites', 0)}")
            self.logger.info(f"  Max fitness: {stats.get('max_fitness', 0):.4f}")
            self.logger.info(f"  QD Score: {stats.get('qd_score', 0):.2f}")
        
        # Trained models
        self.logger.info(f"\nTrained Models ({len(trained_models)}):")
        for m in trained_models:
            self.logger.info(
                f"  top{m['rank']}: {m['cell_type']} ({m['dominant_op']}) "
                f"- Acc: {m['best_val_acc']:.4f} - Params: {m['num_params']:,}"
            )
        
        if trained_models:
            best = max(trained_models, key=lambda x: x['best_val_acc'])
            self.logger.info(f"\nBest: top{best['rank']} with {best['best_val_acc']:.4f} accuracy")
        
        self.logger.info(f"\nTotal time: {total_time/60:.1f} minutes")
        self.logger.info("=" * 70)
        
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
            
    def _print_summary(self):
        """Print final summary."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 70)
        
        # Version results
        for version in ['version1_minimal', 'version2_zeroshot', 'version3_full']:
            if version in self.results and 'final_stats' in self.results[version]:
                stats = self.results[version]['final_stats']
                self.logger.info(f"\n{version.upper()}:")
                self.logger.info(f"  Elites discovered: {stats.get('num_elites', 0)}")
                self.logger.info(f"  Max fitness: {stats.get('max_fitness', 0):.4f}")
                self.logger.info(f"  QD Score: {stats.get('qd_score', 0):.2f}")
                
        # Baseline results (sorted by fitness)
        if 'baselines' in self.results:
            self.logger.info("\nBASELINES (6 methods):")
            
            # Sort baselines by fitness
            baseline_scores = []
            for name, results in self.results['baselines'].items():
                if isinstance(results, dict):
                    fitness = results.get('best_fitness', 0)
                    baseline_scores.append((name, fitness, results))
            
            baseline_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (name, fitness, results) in enumerate(baseline_scores, 1):
                extra_info = ""
                if name == 'grid_search':
                    extra_info = f" ({results.get('num_evaluated', 0)} configs)"
                elif name == 'local_search':
                    extra_info = f" ({results.get('num_restarts', 0)} restarts)"
                elif name == 'nasbench301':
                    extra_info = f" (NB301: {'Yes' if results.get('nb301_available') else 'No'})"
                elif name == 'darts_gradient':
                    extra_info = " (one-shot)"
                    
                self.logger.info(f"  {rank}. {name}: {fitness:.4f}{extra_info}")
                    
        # Final training results
        if 'final_training' in self.results:
            self.logger.info(f"\nFINAL TRAINED MODELS ({self.results['final_training'].get('total_trained', 0)} total):")
            
            # Group by version
            version_groups = {}
            for model_info in self.results['final_training'].get('trained_models', []):
                v = model_info.get('version_short', 'unknown')
                if v not in version_groups:
                    version_groups[v] = []
                version_groups[v].append(model_info)
            
            for version, models in version_groups.items():
                self.logger.info(f"\n  {version.upper()}:")
                for model_info in models:
                    self.logger.info(
                        f"    top{model_info['version_rank']}: {model_info['cell_type']} "
                        f"({model_info['dominant_op']}) - Acc: {model_info['best_val_acc']:.4f} "
                        f"- Params: {model_info['num_params']:,}"
                    )
            
            # Show overall best
            ranked = self.results['final_training'].get('trained_models_ranked', [])
            if ranked:
                best = ranked[0]
                self.logger.info(f"\n  OVERALL BEST: {best['version_short']}/top{best['version_rank']} "
                               f"with {best['best_val_acc']:.4f} accuracy")
                
        # Timing
        if 'metadata' in self.results:
            total_time = self.results['metadata'].get('total_time_seconds', 0)
            self.logger.info(f"\nTotal time: {total_time/60:.1f} minutes")
            
        self.logger.info("\n" + "=" * 70)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NAG-ME-QD: Neural Architecture Generation with MAP-Elites"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run in quick mode with reduced iterations'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./nag_mapelites_output',
        help='Output directory'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--version', type=str, default='all',
        choices=['all', 'v1', 'v2', 'v3', 'baselines'],
        help='Which version(s) to run: v1=minimal, v2=zero-shot, v3=full, baselines=only run baselines'
    )
    parser.add_argument(
        '--epochs', type=int, default=25,
        help='Number of epochs for final training'
    )
    parser.add_argument(
        '--top-k', type=int, default=4,
        help='Number of top architectures to train per version'
    )
    parser.add_argument(
        '--iterations', type=int, default=None,
        help='Override number of MAP-Elites iterations'
    )
    parser.add_argument(
        '--eval-mode', type=str, default='minimal',
        choices=['minimal', 'zero_shot', 'full'],
        help='Evaluation mode for baselines'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from latest checkpoint if available'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to specific checkpoint file to resume from'
    )
    
    args = parser.parse_args()
    
    # Setup config
    config = GlobalConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Create pipeline with resume support
    pipeline = NAGMEQDPipeline(
        config=config, 
        device=config.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )
    
    # Print resume status
    if args.resume:
        print("🔄 Resume mode enabled - will continue from latest checkpoint if available")
    if args.checkpoint:
        print(f"📂 Using checkpoint: {args.checkpoint}")
    
    # Run
    if args.version == 'all':
        results = pipeline.run_full_pipeline(quick_mode=args.quick)
    elif args.version == 'baselines':
        # Run only baselines
        iterations = args.iterations if args.iterations else (50 if args.quick else 100)
        results = pipeline.run_baselines(
            num_iterations=iterations,
            eval_mode=args.eval_mode
        )
        # Save baseline results
        results_path = os.path.join(config.output_dir, "baseline_results.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(pipeline._make_serializable(results), f, indent=2)
        print(f"\nBaseline results saved to: {results_path}")
    else:
        # Run specific version with its own final training
        results = pipeline.run_single_version(
            version=args.version,
            quick_mode=args.quick,
            top_k=args.top_k,
            final_epochs=args.epochs,
            num_iterations=args.iterations
        )
        
    print("\nPipeline complete!")
    print(f"Results saved to: {config.output_dir}")
    

if __name__ == "__main__":
    main()
