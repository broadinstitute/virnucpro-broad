"""CPU-only FASTA streaming dataset for async DataLoader architecture.

This module provides a CUDA-safe IterableDataset for streaming FASTA files
in DataLoader worker processes. Workers perform pure I/O (FASTA parsing) and
yield raw sequence strings, with tokenization deferred to the main process
via collate_fn.

CRITICAL SAFETY:
    Workers MUST NEVER initialize CUDA. This is enforced by:
    1. Using spawn multiprocessing context (not fork)
    2. Setting CUDA_VISIBLE_DEVICES='' in worker_init_fn
    3. Explicit validation in __iter__ that torch.cuda.is_available() == False

Architecture:
    DataLoader Workers (CPU-only):
        └── Parse FASTA files → Yield sequence strings

    Main Process (GPU):
        └── Tokenize in collate_fn → GPU inference

Pattern:
    This dataset is designed for use with VarlenCollator which handles
    tokenization and produces packed batch format for FlashAttention varlen.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info
from Bio import SeqIO

logger = logging.getLogger('virnucpro.data.sequence_dataset')


class SequenceDataset(IterableDataset):
    """CPU-only dataset that streams FASTA files and yields sequence strings.

    This dataset is designed for async DataLoader architecture where:
    - Workers parse FASTA files (CPU I/O only)
    - Workers yield raw sequence strings
    - Tokenization happens in main process (collate_fn)
    - Workers have NO CUDA access (validated at runtime)

    The dataset automatically shards files across workers using round-robin
    distribution based on worker ID.

    Attributes:
        files: List of FASTA file paths to process
        max_length: Maximum sequence length (sequences truncated beyond this)
        _validated: Flag to skip redundant CUDA validation checks

    Example:
        >>> from pathlib import Path
        >>> dataset = SequenceDataset([Path("sequences.fasta")], max_length=1024)
        >>> for item in dataset:
        ...     print(item['id'], len(item['sequence']))
        ...     # {'id': 'seq1', 'sequence': 'MKTAYIAK...', 'file': 'sequences.fasta'}
    """

    def __init__(self, fasta_files: List[Path], max_length: int = 1024):
        """Initialize dataset with FASTA files.

        Args:
            fasta_files: List of FASTA file paths to stream
            max_length: Maximum sequence length (sequences truncated beyond this)

        Note:
            CUDA validation happens in __iter__, not here, because __init__
            runs in main process during Dataset creation. Workers are spawned
            later when DataLoader iteration begins.
        """
        super().__init__()
        self.files = fasta_files
        self.max_length = max_length
        self._validated = False

    def _validate_cuda_isolation(self):
        """Validate that worker process has NO CUDA access.

        This check runs once per worker when iteration starts. It ensures
        workers are CPU-only by checking:
        1. CUDA_VISIBLE_DEVICES is empty (set by worker_init_fn)
        2. torch.cuda.is_available() returns False

        Raises:
            RuntimeError: If worker has CUDA access

        Note:
            This only validates in worker processes, not main process.
            Main process (worker_info is None) is allowed to have CUDA.
        """
        if self._validated:
            return

        worker_info = get_worker_info()

        # Only validate in worker process (not main process)
        if worker_info is not None:
            worker_id = worker_info.id

            # Check CUDA_VISIBLE_DEVICES is hidden
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible != '':
                raise RuntimeError(
                    f"Worker {worker_id}: CUDA_VISIBLE_DEVICES not empty: '{cuda_visible}'. "
                    "Workers must have CUDA_VISIBLE_DEVICES='' for safety."
                )

            # Check torch doesn't see CUDA
            if torch.cuda.is_available():
                raise RuntimeError(
                    f"Worker {worker_id}: CUDA access detected. Workers must be CPU-only. "
                    "Check that spawn multiprocessing context is used and worker_init_fn "
                    "sets CUDA_VISIBLE_DEVICES=''."
                )

            logger.debug(f"Worker {worker_id}: CUDA isolation validated")

        self._validated = True

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Yield sequence dictionaries from FASTA files.

        This method:
        1. Validates CUDA isolation (once per worker)
        2. Shards files across workers (if multi-worker)
        3. Parses FASTA files and yields sequence dictionaries

        Yields:
            Dictionary with keys:
                - 'id': Sequence ID from FASTA record
                - 'sequence': Sequence string (truncated to max_length)
                - 'file': Source filename

        Note:
            File sharding uses round-robin: worker i processes files
            where index % num_workers == worker_id. This ensures
            deterministic distribution across workers.
        """
        # Validate CUDA isolation at start of iteration
        self._validate_cuda_isolation()

        # Get worker info for file sharding
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process loading: process all files
            files_to_process = self.files
            logger.debug(f"Single-process mode: processing {len(files_to_process)} files")
        else:
            # Multi-worker: shard files across workers using round-robin
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = [
                f for i, f in enumerate(self.files)
                if i % num_workers == worker_id
            ]
            logger.debug(
                f"Worker {worker_id}/{num_workers}: assigned {len(files_to_process)} files"
            )

        # Parse FASTA files and yield sequences
        for file_path in files_to_process:
            try:
                for record in SeqIO.parse(file_path, 'fasta'):
                    # Truncate sequence to max_length
                    sequence = str(record.seq)[:self.max_length]

                    yield {
                        'id': record.id,
                        'sequence': sequence,
                        'file': file_path.name
                    }
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                raise
