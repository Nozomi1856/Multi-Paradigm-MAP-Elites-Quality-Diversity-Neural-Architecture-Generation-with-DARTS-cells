#!/usr/bin/env python3
"""
Neural Architecture Generation with MAP-Elites Quality-Diversity (NAG-ME-QD)
============================================================================

A research-grade implementation featuring:
- DARTS cells, DAG cells, Hybrid cells
- Convolution, Transformer, and Recurrent networks
- Multiple datasets with automatic selection based on cell type dominance
- Three evaluation modes: Minimal epochs, Zero-shot, Full-scale with early stopping
- Text-to-specification encoder integrated with MAP-Elites
- Comprehensive quality-diversity optimization

Author: Research Implementation
Version: 1.0.0
"""

import os
import sys
import json
import copy
import math
import time
import random
import logging
import hashlib
import warnings
import traceback
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable
    def trange(*args, **kwargs):
        return range(*args)

try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

@dataclass
class GlobalConfig:
    """Global configuration for the NAG-ME-QD system."""
    # Paths
    output_dir: str = "./nag_mapelites_output"
    checkpoint_dir: str = "./nag_mapelites_checkpoints"
    log_dir: str = "./nag_mapelites_logs"
    
    # Random seed
    seed: int = 42
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MAP-Elites
    grid_resolution: int = 10
    num_iterations: int = 100
    batch_size_eval: int = 32
    
    # Architecture limits
    max_nodes: int = 8
    max_edges: int = 16
    
    # Training
    final_training_epochs: int = 25
    
    # Parallelism
    num_workers: int = 4
    
    # Logging
    log_level: str = "INFO"
    
    # Google Drive backup
    gdrive_backup: bool = True
    gdrive_folder: str = "NAG_MP_QD_checkpoints"
    gdrive_path: str = None  # Set automatically if in Colab
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup Google Drive if enabled
        if self.gdrive_backup:
            self._setup_gdrive()
    
    def _setup_gdrive(self):
        """Mount Google Drive and create backup folder."""
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            self.gdrive_path = f'/content/drive/MyDrive/{self.gdrive_folder}'
            os.makedirs(self.gdrive_path, exist_ok=True)
            print(f"✓ Google Drive backup enabled: {self.gdrive_path}")
        except ImportError:
            print("ℹ Not in Colab - Google Drive backup disabled")
            self.gdrive_backup = False
            self.gdrive_path = None
        except Exception as e:
            print(f"⚠ Could not mount Drive: {e}")
            self.gdrive_backup = False
            self.gdrive_path = None


CONFIG = GlobalConfig()

# =============================================================================
# LOGGING SETUP
# =============================================================================

class ExperimentLogger:
    """Comprehensive experiment logging and tracking."""
    
    def __init__(self, experiment_name: str, config: GlobalConfig):
        self.experiment_name = experiment_name
        self.config = config
        self.start_time = datetime.now()
        self.metrics_history: List[Dict] = []
        self.architecture_history: List[Dict] = []
        self.checkpoints: List[str] = []
        
        # Setup file logger
        log_file = os.path.join(
            config.log_dir, 
            f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Experiment '{experiment_name}' initialized")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Device: {config.device}")
        
    def log_metric(self, iteration: int, metrics: Dict[str, Any]):
        """Log metrics for an iteration."""
        metrics['iteration'] = iteration
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        self.logger.info(f"Iteration {iteration}: {metrics}")
        
    def log_architecture(self, arch_id: str, arch_desc: Dict, fitness: float, 
                         behavior: Tuple[float, ...]):
        """Log architecture discovery."""
        record = {
            'arch_id': arch_id,
            'architecture': arch_desc,
            'fitness': fitness,
            'behavior': behavior,
            'timestamp': datetime.now().isoformat()
        }
        self.architecture_history.append(record)
        self.logger.debug(f"Architecture {arch_id}: fitness={fitness:.4f}, behavior={behavior}")
        
    def log_checkpoint(self, checkpoint_path: str):
        """Log checkpoint creation."""
        self.checkpoints.append(checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def save_summary(self):
        """Save experiment summary to JSON."""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'config': asdict(self.config),
            'num_iterations': len(self.metrics_history),
            'num_architectures_discovered': len(self.architecture_history),
            'checkpoints': self.checkpoints,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else None
        }
        
        summary_path = os.path.join(
            self.config.output_dir,
            f"{self.experiment_name}_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved: {summary_path}")
        return summary_path

    def info(self, msg: str):
        self.logger.info(msg)
        
    def debug(self, msg: str):
        self.logger.debug(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)


# =============================================================================
# SEED MANAGEMENT
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def cleanup_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class CellType(Enum):
    """Types of cells in the architecture."""
    DAG = auto()
    DARTS = auto()
    HYBRID = auto()


class OperationType(Enum):
    """Types of operations."""
    CONV = auto()
    TRANSFORMER = auto()
    RECURRENT = auto()
    IDENTITY = auto()
    ZERO = auto()
    POOLING = auto()


class DatasetType(Enum):
    """Supported datasets."""
    MNIST = auto()
    CIFAR10 = auto()
    CIFAR100 = auto()
    FASHION_MNIST = auto()
    SEQUENTIAL_MNIST = auto()  # For RNN testing
    TEXT_CLASSIFICATION = auto()  # For Transformer testing


@dataclass
class OperationSpec:
    """Specification for an operation."""
    op_type: OperationType
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    num_heads: int = 4
    hidden_dim: int = 64
    dropout: float = 0.1
    
    def to_dict(self) -> Dict:
        return {
            'op_type': self.op_type.name,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'num_heads': self.num_heads,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'OperationSpec':
        d = d.copy()
        d['op_type'] = OperationType[d['op_type']]
        return cls(**d)


@dataclass
class EdgeSpec:
    """Specification for an edge in the architecture graph."""
    src_node: int
    dst_node: int
    operation: OperationSpec
    weight: float = 1.0  # For continuous relaxation
    
    def to_dict(self) -> Dict:
        return {
            'src_node': self.src_node,
            'dst_node': self.dst_node,
            'operation': self.operation.to_dict(),
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'EdgeSpec':
        d = d.copy()
        d['operation'] = OperationSpec.from_dict(d['operation'])
        return cls(**d)


@dataclass
class ArchitectureSpec:
    """Complete architecture specification."""
    cell_type: CellType
    num_nodes: int
    edges: List[EdgeSpec]
    input_channels: int = 3
    num_classes: int = 10
    initial_channels: int = 16
    num_cells: int = 3
    
    # Metadata
    arch_id: str = ""
    creation_time: str = ""
    
    def __post_init__(self):
        if not self.arch_id:
            self.arch_id = self._generate_id()
        if not self.creation_time:
            self.creation_time = datetime.now().isoformat()
            
    def _generate_id(self) -> str:
        """Generate unique architecture ID."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        return {
            'cell_type': self.cell_type.name,
            'num_nodes': self.num_nodes,
            'edges': [e.to_dict() for e in self.edges],
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'initial_channels': self.initial_channels,
            'num_cells': self.num_cells,
            'arch_id': self.arch_id,
            'creation_time': self.creation_time
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ArchitectureSpec':
        d = d.copy()
        d['cell_type'] = CellType[d['cell_type']]
        d['edges'] = [EdgeSpec.from_dict(e) for e in d['edges']]
        return cls(**d)
    
    def get_operation_counts(self) -> Dict[OperationType, int]:
        """Count operations by type."""
        counts = defaultdict(int)
        for edge in self.edges:
            counts[edge.operation.op_type] += 1
        return dict(counts)
    
    def get_dominant_operation(self) -> OperationType:
        """Get the most common operation type."""
        counts = self.get_operation_counts()
        if not counts:
            return OperationType.CONV
        return max(counts, key=counts.get)
    
    def get_behavior_descriptor(self) -> Tuple[float, ...]:
        """
        Compute behavior descriptor for MAP-Elites.
        Returns: (depth_ratio, conv_ratio, transformer_ratio, recurrent_ratio, connectivity)
        """
        if not self.edges:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Depth: max path length normalized
        max_depth = self._compute_max_depth()
        depth_ratio = max_depth / max(self.num_nodes, 1)
        
        # Operation ratios
        counts = self.get_operation_counts()
        total_ops = sum(counts.values())
        conv_ratio = counts.get(OperationType.CONV, 0) / max(total_ops, 1)
        transformer_ratio = counts.get(OperationType.TRANSFORMER, 0) / max(total_ops, 1)
        recurrent_ratio = counts.get(OperationType.RECURRENT, 0) / max(total_ops, 1)
        
        # Connectivity: edges / max possible edges
        max_edges = self.num_nodes * (self.num_nodes - 1) // 2
        connectivity = len(self.edges) / max(max_edges, 1)
        
        return (depth_ratio, conv_ratio, transformer_ratio, recurrent_ratio, connectivity)
    
    def _compute_max_depth(self) -> int:
        """Compute maximum path depth in the architecture graph."""
        if not self.edges:
            return 0
            
        # Build adjacency list
        adj = defaultdict(list)
        for edge in self.edges:
            adj[edge.src_node].append(edge.dst_node)
        
        # DFS to find max depth
        def dfs(node: int, visited: Set[int]) -> int:
            if node in visited:
                return 0
            visited.add(node)
            max_child_depth = 0
            for child in adj[node]:
                max_child_depth = max(max_child_depth, dfs(child, visited))
            visited.remove(node)
            return 1 + max_child_depth
        
        max_depth = 0
        for start_node in range(self.num_nodes):
            max_depth = max(max_depth, dfs(start_node, set()))
        
        return max_depth


# =============================================================================
# OPERATIONS
# =============================================================================

class AdaptiveChannelLayer(nn.Module):
    """Dynamically adapts channel dimensions."""
    
    def __init__(self, target_channels: int):
        super().__init__()
        self.target_channels = target_channels
        self.adapters = nn.ModuleDict()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # Conv: (B, C, H, W)
            current_channels = x.shape[1]
        elif x.dim() == 3:  # Sequence: (B, L, C)
            current_channels = x.shape[2]
        else:
            return x
            
        if current_channels == self.target_channels:
            return x
            
        # Create adapter on the fly if needed
        key = f"adapt_{current_channels}_{self.target_channels}"
        if key not in self.adapters:
            if x.dim() == 4:
                self.adapters[key] = nn.Conv2d(
                    current_channels, self.target_channels, 1
                ).to(x.device)
            else:
                self.adapters[key] = nn.Linear(
                    current_channels, self.target_channels
                ).to(x.device)
                
        return self.adapters[key](x)


class ConvOperation(nn.Module):
    """Convolutional operation with various configurations."""
    
    def __init__(self, in_channels: int, out_channels: int, spec: OperationSpec):
        super().__init__()
        self.spec = spec
        
        # Main convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(spec.dropout)
        
        # Channel adapter for input
        self.input_adapter = AdaptiveChannelLayer(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input formats
        if x.dim() == 3:  # (B, L, C) -> (B, C, H, W)
            b, l, c = x.shape
            h = w = int(math.sqrt(l))
            if h * w != l:
                # Pad to nearest square
                h = w = int(math.ceil(math.sqrt(l)))
                pad_len = h * w - l
                x = F.pad(x, (0, 0, 0, pad_len))
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
        
        # Adapt channels if needed
        if x.shape[1] != self.conv.in_channels:
            x = self.input_adapter(x)
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SepConvOperation(nn.Module):
    """Separable convolution operation."""
    
    def __init__(self, in_channels: int, out_channels: int, spec: OperationSpec):
        super().__init__()
        self.spec = spec
        
        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.input_adapter = AdaptiveChannelLayer(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            b, l, c = x.shape
            h = w = int(math.ceil(math.sqrt(l)))
            pad_len = h * w - l
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
            
        if x.shape[1] != self.depthwise.in_channels:
            x = self.input_adapter(x)
            
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class DilConvOperation(nn.Module):
    """Dilated convolution operation."""
    
    def __init__(self, in_channels: int, out_channels: int, spec: OperationSpec):
        super().__init__()
        self.spec = spec
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding * spec.dilation,
            dilation=spec.dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.input_adapter = AdaptiveChannelLayer(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            b, l, c = x.shape
            h = w = int(math.ceil(math.sqrt(l)))
            pad_len = h * w - l
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
            
        if x.shape[1] != self.conv.in_channels:
            x = self.input_adapter(x)
            
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransformerOperation(nn.Module):
    """Self-attention transformer block."""
    
    def __init__(self, embed_dim: int, spec: OperationSpec):
        super().__init__()
        self.spec = spec
        self.embed_dim = embed_dim
        
        # Ensure num_heads divides embed_dim
        num_heads = spec.num_heads
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=spec.dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.hidden_dim, embed_dim),
            nn.Dropout(spec.dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.input_adapter = AdaptiveChannelLayer(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from (B, C, H, W) to (B, L, C) if needed
        original_shape = x.shape
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            
        # Adapt channel dimension
        if x.shape[-1] != self.embed_dim:
            x = self.input_adapter(x)
            
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Convert back if needed
        if len(original_shape) == 4:
            x = x.permute(0, 2, 1).reshape(original_shape[0], -1, original_shape[2], original_shape[3])
            
        return x


class RecurrentOperation(nn.Module):
    """Recurrent (LSTM/GRU) operation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, spec: OperationSpec):
        super().__init__()
        self.spec = spec
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=spec.dropout if spec.dropout > 0 else 0
        )
        
        # Project back to input dimension
        self.projection = nn.Linear(hidden_dim * 2, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        
        self.input_adapter = AdaptiveChannelLayer(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        
        # Convert from (B, C, H, W) to (B, L, C) if needed
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            
        # Adapt input dimension
        if x.shape[-1] != self.input_dim:
            x = self.input_adapter(x)
            
        # RNN forward
        rnn_out, _ = self.rnn(x)
        
        # Project and normalize
        x = self.projection(rnn_out)
        x = self.norm(x)
        
        # Convert back if needed
        if len(original_shape) == 4:
            x = x.permute(0, 2, 1).reshape(original_shape[0], -1, original_shape[2], original_shape[3])
            
        return x


class IdentityOperation(nn.Module):
    """Identity operation with optional channel adaptation."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.adapter = AdaptiveChannelLayer(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] != self.channels:
            return self.adapter(x)
        elif x.dim() == 3 and x.shape[-1] != self.channels:
            return self.adapter(x)
        return x


class ZeroOperation(nn.Module):
    """Zero operation (no information flow)."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class PoolingOperation(nn.Module):
    """Pooling operation (max or average)."""
    
    def __init__(self, channels: int, pool_type: str = 'max', kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        else:
            self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
            
        self.adapter = AdaptiveChannelLayer(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            b, l, c = x.shape
            h = w = int(math.ceil(math.sqrt(l)))
            pad_len = h * w - l
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
            
        if x.shape[1] != self.channels:
            x = self.adapter(x)
            
        return self.pool(x)


def create_operation(op_spec: OperationSpec, in_channels: int, out_channels: int) -> nn.Module:
    """Factory function to create operations."""
    if op_spec.op_type == OperationType.CONV:
        return ConvOperation(in_channels, out_channels, op_spec)
    elif op_spec.op_type == OperationType.TRANSFORMER:
        return TransformerOperation(out_channels, op_spec)
    elif op_spec.op_type == OperationType.RECURRENT:
        return RecurrentOperation(in_channels, out_channels, op_spec)
    elif op_spec.op_type == OperationType.IDENTITY:
        return IdentityOperation(out_channels)
    elif op_spec.op_type == OperationType.ZERO:
        return ZeroOperation(out_channels)
    elif op_spec.op_type == OperationType.POOLING:
        return PoolingOperation(out_channels)
    else:
        return IdentityOperation(out_channels)


# =============================================================================
# CELL IMPLEMENTATIONS
# =============================================================================

class DAGCell(nn.Module):
    """
    Direct Acyclic Graph cell with arbitrary connections.
    Each node can receive input from multiple previous nodes.
    """
    
    def __init__(self, arch_spec: ArchitectureSpec, channels: int):
        super().__init__()
        self.arch_spec = arch_spec
        self.channels = channels
        self.num_nodes = arch_spec.num_nodes
        
        # Create operations for each edge
        self.edge_ops = nn.ModuleDict()
        for i, edge in enumerate(arch_spec.edges):
            op = create_operation(edge.operation, channels, channels)
            self.edge_ops[f"edge_{edge.src_node}_{edge.dst_node}"] = op
            
        # Build adjacency structure
        self.node_inputs = defaultdict(list)
        self.edge_weights = {}
        for edge in arch_spec.edges:
            self.node_inputs[edge.dst_node].append(edge.src_node)
            self.edge_weights[(edge.src_node, edge.dst_node)] = edge.weight
            
        # Input projection
        self.input_proj = nn.Conv2d(channels, channels, 1)
        
        # Output combination
        self.output_combine = nn.Conv2d(channels * self.num_nodes, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)
        
        # Initialize node outputs
        node_outputs = {0: x}  # Node 0 receives the input
        
        # Process nodes in order
        for node_id in range(1, self.num_nodes):
            inputs = []
            for src_node in self.node_inputs.get(node_id, []):
                if src_node in node_outputs:
                    edge_key = f"edge_{src_node}_{node_id}"
                    if edge_key in self.edge_ops:
                        weight = self.edge_weights.get((src_node, node_id), 1.0)
                        op_out = self.edge_ops[edge_key](node_outputs[src_node])
                        inputs.append(weight * op_out)
                        
            if inputs:
                # Sum all inputs to this node
                node_outputs[node_id] = sum(inputs)
            else:
                # No inputs - use zero or identity from previous node
                node_outputs[node_id] = torch.zeros_like(x)
                
        # Combine all node outputs
        all_outputs = []
        for i in range(self.num_nodes):
            if i in node_outputs:
                all_outputs.append(node_outputs[i])
            else:
                all_outputs.append(torch.zeros_like(x))
                
        combined = torch.cat(all_outputs, dim=1)
        return self.output_combine(combined)


class DARTSCell(nn.Module):
    """
    DARTS-style cell with continuous relaxation.
    Uses architecture weights to blend multiple operations.
    """
    
    def __init__(self, arch_spec: ArchitectureSpec, channels: int, num_ops: int = 4):
        super().__init__()
        self.arch_spec = arch_spec
        self.channels = channels
        self.num_nodes = arch_spec.num_nodes
        self.num_ops = num_ops
        
        # Create operation choices for each edge position
        self.op_choices = nn.ModuleDict()
        
        # Operation primitives for DARTS
        self.op_types = [
            OperationSpec(OperationType.CONV, kernel_size=3),
            OperationSpec(OperationType.CONV, kernel_size=5),
            OperationSpec(OperationType.POOLING),
            OperationSpec(OperationType.IDENTITY),
        ]
        
        # Create edges with multiple operation choices
        for edge in arch_spec.edges:
            edge_key = f"edge_{edge.src_node}_{edge.dst_node}"
            ops = nn.ModuleList()
            for op_spec in self.op_types:
                ops.append(create_operation(op_spec, channels, channels))
            self.op_choices[edge_key] = ops
            
        # Architecture weights (alpha) for continuous relaxation
        num_edges = len(arch_spec.edges)
        self.arch_weights = nn.Parameter(
            torch.randn(num_edges, len(self.op_types)) * 0.001
        )
        
        # Build adjacency structure
        self.node_inputs = defaultdict(list)
        self.edge_indices = {}
        for i, edge in enumerate(arch_spec.edges):
            self.node_inputs[edge.dst_node].append(edge.src_node)
            self.edge_indices[(edge.src_node, edge.dst_node)] = i
            
        # Input and output projections
        self.input_proj = nn.Conv2d(channels, channels, 1)
        self.output_combine = nn.Conv2d(channels * self.num_nodes, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        node_outputs = {0: x}
        
        # Get softmax weights
        weights = F.softmax(self.arch_weights, dim=-1)
        
        for node_id in range(1, self.num_nodes):
            inputs = []
            for src_node in self.node_inputs.get(node_id, []):
                if src_node in node_outputs:
                    edge_key = f"edge_{src_node}_{node_id}"
                    if edge_key in self.op_choices:
                        edge_idx = self.edge_indices[(src_node, node_id)]
                        
                        # Weighted sum of all operations (continuous relaxation)
                        mixed_op = 0
                        for op_idx, op in enumerate(self.op_choices[edge_key]):
                            try:
                                op_out = op(node_outputs[src_node])
                                mixed_op = mixed_op + weights[edge_idx, op_idx] * op_out
                            except Exception:
                                pass
                                
                        if isinstance(mixed_op, torch.Tensor):
                            inputs.append(mixed_op)
                            
            if inputs:
                node_outputs[node_id] = sum(inputs)
            else:
                node_outputs[node_id] = torch.zeros_like(x)
                
        all_outputs = [node_outputs.get(i, torch.zeros_like(x)) for i in range(self.num_nodes)]
        combined = torch.cat(all_outputs, dim=1)
        return self.output_combine(combined)
    
    def get_discrete_architecture(self) -> List[Tuple[int, int, int]]:
        """Get discrete architecture by selecting best operation per edge."""
        weights = F.softmax(self.arch_weights, dim=-1)
        best_ops = weights.argmax(dim=-1).tolist()
        
        result = []
        for i, edge in enumerate(self.arch_spec.edges):
            result.append((edge.src_node, edge.dst_node, best_ops[i]))
        return result


class HybridCell(nn.Module):
    """
    Hybrid cell combining DAG flexibility with DARTS continuous relaxation.
    Supports mixed operation types (Conv, Transformer, Recurrent).
    """
    
    def __init__(self, arch_spec: ArchitectureSpec, channels: int):
        super().__init__()
        self.arch_spec = arch_spec
        self.channels = channels
        self.num_nodes = arch_spec.num_nodes
        
        # Create operations with their specified types
        self.edge_ops = nn.ModuleDict()
        self.edge_weights = nn.ParameterDict()
        
        for i, edge in enumerate(arch_spec.edges):
            edge_key = f"edge_{edge.src_node}_{edge.dst_node}"
            self.edge_ops[edge_key] = create_operation(edge.operation, channels, channels)
            self.edge_weights[edge_key] = nn.Parameter(torch.tensor(edge.weight))
            
        # Build adjacency structure
        self.node_inputs = defaultdict(list)
        for edge in arch_spec.edges:
            self.node_inputs[edge.dst_node].append(edge.src_node)
            
        # Projections
        self.input_proj = nn.Conv2d(channels, channels, 1)
        self.output_combine = nn.Conv2d(channels * self.num_nodes, channels, 1)
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([channels]) for _ in range(self.num_nodes)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        node_outputs = {0: x}
        
        for node_id in range(1, self.num_nodes):
            inputs = []
            for src_node in self.node_inputs.get(node_id, []):
                if src_node in node_outputs:
                    edge_key = f"edge_{src_node}_{node_id}"
                    if edge_key in self.edge_ops:
                        weight = torch.sigmoid(self.edge_weights[edge_key])
                        try:
                            op_out = self.edge_ops[edge_key](node_outputs[src_node])
                            # Ensure shape compatibility
                            if op_out.shape == node_outputs[src_node].shape:
                                inputs.append(weight * op_out)
                            else:
                                # Adapt shape
                                if op_out.dim() == 4 and node_outputs[src_node].dim() == 4:
                                    if op_out.shape[1] != node_outputs[src_node].shape[1]:
                                        adapter = nn.Conv2d(
                                            op_out.shape[1], 
                                            node_outputs[src_node].shape[1], 1
                                        ).to(op_out.device)
                                        op_out = adapter(op_out)
                                inputs.append(weight * op_out)
                        except Exception:
                            pass
                            
            if inputs:
                node_outputs[node_id] = sum(inputs)
            else:
                node_outputs[node_id] = torch.zeros_like(x)
                
        all_outputs = [node_outputs.get(i, torch.zeros_like(x)) for i in range(self.num_nodes)]
        combined = torch.cat(all_outputs, dim=1)
        return self.output_combine(combined)


def create_cell(arch_spec: ArchitectureSpec, channels: int) -> nn.Module:
    """Factory function to create cells based on type."""
    if arch_spec.cell_type == CellType.DAG:
        return DAGCell(arch_spec, channels)
    elif arch_spec.cell_type == CellType.DARTS:
        return DARTSCell(arch_spec, channels)
    elif arch_spec.cell_type == CellType.HYBRID:
        return HybridCell(arch_spec, channels)
    else:
        return DAGCell(arch_spec, channels)


# =============================================================================
# COMPLETE NETWORK
# =============================================================================

class NASNetwork(nn.Module):
    """
    Complete network built from architecture specification.
    Automatically selects network structure based on dominant operation type.
    """
    
    def __init__(self, arch_spec: ArchitectureSpec):
        super().__init__()
        self.arch_spec = arch_spec
        self.dominant_op = arch_spec.get_dominant_operation()
        
        channels = arch_spec.initial_channels
        
        # Stem based on input type
        if arch_spec.input_channels in [1, 3]:  # Image input
            self.stem = nn.Sequential(
                nn.Conv2d(arch_spec.input_channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Linear(arch_spec.input_channels, channels)
            
        # Build cells
        self.cells = nn.ModuleList()
        for i in range(arch_spec.num_cells):
            cell = create_cell(arch_spec, channels)
            self.cells.append(cell)
            
            # Reduction every other cell
            if i % 2 == 1 and i < arch_spec.num_cells - 1:
                self.cells.append(nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels * 2),
                    nn.ReLU(inplace=True)
                ))
                channels *= 2
                
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, arch_spec.num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        
        # Ensure 4D tensor for conv cells
        if x.dim() == 2:
            b, c = x.shape
            h = w = int(math.ceil(math.sqrt(c)))
            x = F.pad(x, (0, h * w - c))
            x = x.view(b, 1, h, w)
            
        # Process through cells
        for cell in self.cells:
            try:
                x = cell(x)
            except Exception as e:
                # Skip problematic cells
                pass
                
        # Global pool and classify
        if x.dim() == 4:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        elif x.dim() == 3:
            x = x.mean(dim=1)
            
        x = self.classifier(x)
        return x
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_flops(self, input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
        """Estimate FLOPs (rough approximation)."""
        # Simple estimation based on parameters
        return self.get_num_params() * 2


# =============================================================================
# TEXT-TO-SPECIFICATION ENCODER
# =============================================================================

class TextToSpecEncoder(nn.Module):
    """
    Encoder that converts text descriptions to architecture specifications.
    Uses a transformer-based architecture with learned embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 128,
        max_nodes: int = 8,
        max_edges: int = 16,
        num_op_types: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.num_op_types = num_op_types
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.cell_type_head = nn.Linear(embed_dim, 3)  # DAG, DARTS, HYBRID
        self.num_nodes_head = nn.Linear(embed_dim, max_nodes)
        self.num_edges_head = nn.Linear(embed_dim, max_edges)
        
        # Edge prediction heads (for each possible edge)
        self.edge_exists_head = nn.Linear(embed_dim, max_nodes * max_nodes)
        self.edge_op_head = nn.Linear(embed_dim, max_nodes * max_nodes * num_op_types)
        
        # Operation parameter heads
        self.kernel_size_head = nn.Linear(embed_dim, 3)  # 3, 5, 7
        self.num_heads_head = nn.Linear(embed_dim, 4)  # 1, 2, 4, 8
        
        # Simple tokenizer
        self.vocab = self._build_vocab()
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for tokenization."""
        vocab = {
            '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3,
            # Cell types
            'dag': 4, 'darts': 5, 'hybrid': 6, 'cell': 7,
            # Operations
            'conv': 8, 'convolution': 9, 'transformer': 10, 'attention': 11,
            'recurrent': 12, 'rnn': 13, 'lstm': 14, 'gru': 15,
            'pooling': 16, 'pool': 17, 'identity': 18, 'skip': 19,
            # Modifiers
            'deep': 20, 'shallow': 21, 'wide': 22, 'narrow': 23,
            'fast': 24, 'accurate': 25, 'efficient': 26, 'large': 27,
            'small': 28, 'simple': 29, 'complex': 30,
            # Tasks
            'image': 31, 'classification': 32, 'vision': 33, 'text': 34,
            'sequence': 35, 'time': 36, 'series': 37,
            # Numbers
            'one': 38, 'two': 39, 'three': 40, 'four': 41, 'five': 42,
            'six': 43, 'seven': 44, 'eight': 45,
            # Connectors
            'and': 46, 'with': 47, 'for': 48, 'using': 49, 'network': 50,
            'architecture': 51, 'model': 52, 'nodes': 53, 'layers': 54,
            'connections': 55, 'edges': 56,
            # Additional terms
            'separable': 57, 'dilated': 58, 'dense': 59, 'sparse': 60,
            'residual': 61, 'multi': 62, 'head': 63, 'self': 64,
        }
        # Add more generic tokens
        for i in range(65, 1000):
            vocab[f'<TOKEN_{i}>'] = i
        return vocab
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text input."""
        tokens = [self.vocab.get('<BOS>', 2)]
        for word in text.lower().split():
            word = ''.join(c for c in word if c.isalnum())
            tokens.append(self.vocab.get(word, self.vocab['<UNK>']))
        tokens.append(self.vocab.get('<EOS>', 3))
        
        # Pad to max length
        max_len = 128
        if len(tokens) < max_len:
            tokens.extend([self.vocab['<PAD>']] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
            
        return torch.tensor(tokens, dtype=torch.long)
    
    def forward(self, text: Union[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate architecture specification from text.
        
        Args:
            text: Text description or tokenized tensor
            
        Returns:
            Dictionary with architecture predictions
        """
        if isinstance(text, str):
            tokens = self.tokenize(text).unsqueeze(0)
            device = next(self.parameters()).device
            tokens = tokens.to(device)
        else:
            tokens = text
            
        batch_size, seq_len = tokens.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global representation (mean pooling)
        x_global = x.mean(dim=1)
        
        # Predictions
        predictions = {
            'cell_type_logits': self.cell_type_head(x_global),
            'num_nodes_logits': self.num_nodes_head(x_global),
            'num_edges_logits': self.num_edges_head(x_global),
            'edge_exists_logits': self.edge_exists_head(x_global).view(
                batch_size, self.max_nodes, self.max_nodes
            ),
            'edge_op_logits': self.edge_op_head(x_global).view(
                batch_size, self.max_nodes, self.max_nodes, self.num_op_types
            ),
            'kernel_size_logits': self.kernel_size_head(x_global),
            'num_heads_logits': self.num_heads_head(x_global),
            'embedding': x_global
        }
        
        return predictions
    
    def generate_architecture(
        self, 
        text: str, 
        temperature: float = 1.0,
        input_channels: int = 3,
        num_classes: int = 10
    ) -> ArchitectureSpec:
        """
        Generate architecture specification from text description.
        
        Args:
            text: Natural language description
            temperature: Sampling temperature
            input_channels: Number of input channels
            num_classes: Number of output classes
            
        Returns:
            ArchitectureSpec object
        """
        self.eval()
        with torch.no_grad():
            preds = self(text)
            
            # Sample cell type
            cell_type_probs = F.softmax(preds['cell_type_logits'] / temperature, dim=-1)
            cell_type_idx = torch.multinomial(cell_type_probs, 1).item()
            cell_types = [CellType.DAG, CellType.DARTS, CellType.HYBRID]
            cell_type = cell_types[cell_type_idx]
            
            # Sample number of nodes
            num_nodes_probs = F.softmax(preds['num_nodes_logits'] / temperature, dim=-1)
            num_nodes = torch.multinomial(num_nodes_probs, 1).item() + 2  # Minimum 2 nodes
            num_nodes = min(num_nodes, self.max_nodes)
            
            # Sample edges
            edge_exists_probs = torch.sigmoid(preds['edge_exists_logits'][0] / temperature)
            edge_op_probs = F.softmax(preds['edge_op_logits'][0] / temperature, dim=-1)
            
            edges = []
            op_types = [
                OperationType.CONV, OperationType.TRANSFORMER, OperationType.RECURRENT,
                OperationType.IDENTITY, OperationType.POOLING, OperationType.ZERO
            ]
            
            for src in range(num_nodes):
                for dst in range(src + 1, num_nodes):
                    if edge_exists_probs[src, dst] > 0.3:  # Threshold
                        op_idx = torch.multinomial(edge_op_probs[src, dst], 1).item()
                        op_type = op_types[op_idx % len(op_types)]
                        
                        # Get kernel size
                        kernel_sizes = [3, 5, 7]
                        ks_probs = F.softmax(preds['kernel_size_logits'] / temperature, dim=-1)
                        ks_idx = torch.multinomial(ks_probs, 1).item()
                        kernel_size = kernel_sizes[ks_idx]
                        
                        # Get num heads
                        num_heads_options = [1, 2, 4, 8]
                        nh_probs = F.softmax(preds['num_heads_logits'] / temperature, dim=-1)
                        nh_idx = torch.multinomial(nh_probs, 1).item()
                        num_heads = num_heads_options[nh_idx]
                        
                        op_spec = OperationSpec(
                            op_type=op_type,
                            kernel_size=kernel_size,
                            num_heads=num_heads
                        )
                        
                        edge = EdgeSpec(
                            src_node=src,
                            dst_node=dst,
                            operation=op_spec,
                            weight=edge_exists_probs[src, dst].item()
                        )
                        edges.append(edge)
                        
            # Ensure at least one edge
            if not edges:
                edges.append(EdgeSpec(
                    src_node=0,
                    dst_node=1,
                    operation=OperationSpec(OperationType.CONV),
                    weight=1.0
                ))
                
            # Limit edges
            if len(edges) > self.max_edges:
                edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:self.max_edges]
                
            return ArchitectureSpec(
                cell_type=cell_type,
                num_nodes=num_nodes,
                edges=edges,
                input_channels=input_channels,
                num_classes=num_classes
            )


class TextToSpecEncoderTrainer:
    """Trainer for the text-to-specification encoder."""
    
    def __init__(
        self,
        encoder: TextToSpecEncoder,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        grad_clip: float = 1.0
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
        self.grad_clip = grad_clip
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def train_step(
        self,
        text_batch: List[str],
        target_specs: List[ArchitectureSpec]
    ) -> Dict[str, float]:
        """Single training step."""
        self.encoder.train()
        self.optimizer.zero_grad()
        
        # Tokenize batch
        tokens = torch.stack([self.encoder.tokenize(t) for t in text_batch]).to(self.device)
        
        # Forward pass
        preds = self.encoder(tokens)
        
        # Compute losses
        losses = {}
        total_loss = 0
        
        # Cell type loss
        cell_type_targets = torch.tensor([
            [CellType.DAG, CellType.DARTS, CellType.HYBRID].index(s.cell_type)
            for s in target_specs
        ], device=self.device)
        loss_ct = self.ce_loss(preds['cell_type_logits'], cell_type_targets)
        losses['cell_type'] = loss_ct.item()
        total_loss += loss_ct
        
        # Num nodes loss
        num_nodes_targets = torch.tensor([
            min(s.num_nodes - 2, self.encoder.max_nodes - 1)
            for s in target_specs
        ], device=self.device)
        loss_nn = self.ce_loss(preds['num_nodes_logits'], num_nodes_targets)
        losses['num_nodes'] = loss_nn.item()
        total_loss += loss_nn
        
        # Edge existence loss
        edge_targets = torch.zeros(
            len(target_specs), self.encoder.max_nodes, self.encoder.max_nodes,
            device=self.device
        )
        for i, spec in enumerate(target_specs):
            for edge in spec.edges:
                if edge.src_node < self.encoder.max_nodes and edge.dst_node < self.encoder.max_nodes:
                    edge_targets[i, edge.src_node, edge.dst_node] = 1.0
                    
        loss_ee = self.bce_loss(preds['edge_exists_logits'], edge_targets)
        losses['edge_exists'] = loss_ee.item()
        total_loss += loss_ee
        
        losses['total'] = total_loss.item()
        
        # Backward pass with gradient clipping
        total_loss.backward()
        clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
        self.optimizer.step()
        
        return losses


# =============================================================================
# DATASETS
# =============================================================================

class DatasetManager:
    """Manages dataset loading and preprocessing."""
    
    def __init__(self, data_dir: str = './data', batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        os.makedirs(data_dir, exist_ok=True)
        
    def get_dataset(
        self, 
        dataset_type: DatasetType,
        train: bool = True,
        subset_size: Optional[int] = None
    ) -> DataLoader:
        """Get dataloader for specified dataset."""
        
        if not TORCHVISION_AVAILABLE:
            # Return dummy data if torchvision not available
            return self._get_dummy_loader(dataset_type, train, subset_size)
            
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        transform = transform_train if train else transform_test
        
        if dataset_type == DatasetType.MNIST:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = torchvision.datasets.MNIST(
                self.data_dir, train=train, download=True, transform=transform
            )
            
        elif dataset_type == DatasetType.FASHION_MNIST:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            dataset = torchvision.datasets.FashionMNIST(
                self.data_dir, train=train, download=True, transform=transform
            )
            
        elif dataset_type == DatasetType.CIFAR10:
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            dataset = torchvision.datasets.CIFAR10(
                self.data_dir, train=train, download=True, transform=transform
            )
            
        elif dataset_type == DatasetType.CIFAR100:
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            dataset = torchvision.datasets.CIFAR100(
                self.data_dir, train=train, download=True, transform=transform
            )
            
        else:
            # Default to CIFAR10
            dataset = torchvision.datasets.CIFAR10(
                self.data_dir, train=train, download=True, transform=transform_train if train else transform_test
            )
            
        # Subset if requested
        if subset_size is not None and subset_size < len(dataset):
            indices = random.sample(range(len(dataset)), subset_size)
            dataset = Subset(dataset, indices)
            
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def _get_dummy_loader(
        self,
        dataset_type: DatasetType,
        train: bool,
        subset_size: Optional[int]
    ) -> DataLoader:
        """Get dummy dataloader when torchvision is unavailable."""
        
        class DummyDataset(Dataset):
            def __init__(self, size, channels, img_size, num_classes):
                self.size = size
                self.channels = channels
                self.img_size = img_size
                self.num_classes = num_classes
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                x = torch.randn(self.channels, self.img_size, self.img_size)
                y = random.randint(0, self.num_classes - 1)
                return x, y
                
        configs = {
            DatasetType.MNIST: (1, 32, 10),
            DatasetType.FASHION_MNIST: (1, 32, 10),
            DatasetType.CIFAR10: (3, 32, 10),
            DatasetType.CIFAR100: (3, 32, 100),
        }
        
        channels, img_size, num_classes = configs.get(
            dataset_type, (3, 32, 10)
        )
        
        size = subset_size if subset_size else (50000 if train else 10000)
        dataset = DummyDataset(size, channels, img_size, num_classes)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train
        )
        
    def get_dataset_for_architecture(
        self,
        arch_spec: ArchitectureSpec,
        train: bool = True,
        subset_size: Optional[int] = None
    ) -> DataLoader:
        """
        Select appropriate dataset based on architecture's dominant operation.
        """
        dominant_op = arch_spec.get_dominant_operation()
        
        # Map operations to suitable datasets
        if dominant_op == OperationType.RECURRENT:
            # Sequential data for RNNs
            dataset_type = DatasetType.SEQUENTIAL_MNIST
        elif dominant_op == OperationType.TRANSFORMER:
            # Can use vision for Vision Transformers
            dataset_type = DatasetType.CIFAR10
        else:
            # Default to CIFAR10 for conv-heavy architectures
            dataset_type = DatasetType.CIFAR10
            
        return self.get_dataset(dataset_type, train, subset_size)


# Continue in next part...
if __name__ == "__main__":
    print("NAG-ME-QD Module 1 loaded successfully")
    print(f"TQDM available: {TQDM_AVAILABLE}")
    print(f"Torchvision available: {TORCHVISION_AVAILABLE}")
    print(f"Device: {CONFIG.device}")
