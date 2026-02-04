"""Data loading and optimization utilities"""

from virnucpro.data.dataloader_utils import (
    create_optimized_dataloader,
    get_optimal_workers,
    create_sequence_dataloader,
    estimate_memory_usage,
    create_async_dataloader,
    cuda_safe_worker_init,
)
from virnucpro.data.sequence_dataset import SequenceDataset
from virnucpro.data.collators import VarlenCollator
from virnucpro.data.packing import GreedyPacker, calculate_token_budget, validate_packed_equivalence
from virnucpro.data.shard_index import (
    SequenceEntry,
    create_sequence_index,
    get_worker_indices,
    load_sequence_index,
)

__all__ = [
    'create_optimized_dataloader',
    'get_optimal_workers',
    'create_sequence_dataloader',
    'estimate_memory_usage',
    'create_async_dataloader',
    'cuda_safe_worker_init',
    'SequenceDataset',
    'VarlenCollator',
    'GreedyPacker',
    'calculate_token_budget',
    'validate_packed_equivalence',
    'SequenceEntry',
    'create_sequence_index',
    'get_worker_indices',
    'load_sequence_index',
]
