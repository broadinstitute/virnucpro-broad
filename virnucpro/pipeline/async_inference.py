"""Async inference runner for single-GPU processing with stream overlap.

This module implements the async DataLoader architecture where:
- DataLoader workers handle CPU-only I/O (FASTA parsing)
- Main process tokenizes and packs sequences via collator
- GPU receives prefetched batches
- CUDA streams overlap data transfer with computation

Critical: Stateful collators (VarlenCollator with enable_packing=True)
run in the MAIN PROCESS, not in DataLoader workers. This prevents data
loss from PyTorch pickling stateful collators to worker subprocesses,
where each worker's buffer would accumulate sequences that are never
flushed when the worker process ends.

This is the single-GPU foundation. Multi-GPU coordination is Phase 7.
"""

from __future__ import annotations

import torch
import time
import logging
import os
import hashlib
from typing import Optional, List, Dict, Any, Callable, Iterator, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader

from virnucpro.cuda.stream_manager import StreamProcessor
from virnucpro.utils.gpu_monitor import NvitopMonitor
from virnucpro.core.env_config import get_env_config
from virnucpro.pipeline.checkpoint_writer import (
    CheckpointTrigger,
    AsyncCheckpointWriter,
    validate_checkpoint_pt,
    resume_from_checkpoints
)

if TYPE_CHECKING:
    from virnucpro.pipeline.checkpoint_manifest import CheckpointManifest

logger = logging.getLogger('virnucpro.pipeline.async_inference')


def check_numerical_stability(embeddings: torch.Tensor, context: str = "embeddings") -> None:
    """Detect NaN/Inf in tensors with minimal CUDA synchronization.

    Optimized to perform single sync point - batches all GPU operations before
    calling .item(). Reduces latency from ~5-10ms to <1ms per batch.

    Args:
        embeddings: Tensor to check (any shape)
        context: Description for error message (e.g., "batch_42")

    Raises:
        RuntimeError: If NaN or Inf detected, with diagnostic stats
    """
    # Batch all GPU operations (no sync yet)
    nan_mask = torch.isnan(embeddings)
    inf_mask = torch.isinf(embeddings)
    has_nan = nan_mask.any()
    has_inf = inf_mask.any()

    # Single sync point - only sync if error detected
    if has_nan.item() or has_inf.item():
        # Now collect diagnostics (already computed masks above)
        valid_mask = ~(nan_mask | inf_mask)
        valid_vals = embeddings[valid_mask]

        # Batch remaining .item() calls
        stats = {
            "nan_count": nan_mask.sum().item(),
            "inf_count": inf_mask.sum().item(),
            "valid_min": valid_vals.min().item() if valid_vals.numel() > 0 else float('nan'),
            "valid_max": valid_vals.max().item() if valid_vals.numel() > 0 else float('nan'),
        }

        raise RuntimeError(
            f"Numerical instability in {context}: "
            f"NaN={stats['nan_count']}, Inf={stats['inf_count']}, "
            f"valid range=[{stats['valid_min']:.2e}, {stats['valid_max']:.2e}]. "
            f"This may indicate FP16 overflow. Try VIRNUCPRO_DISABLE_FP16=1"
        )


@dataclass
class InferenceResult:
    """Result from async inference."""
    sequence_ids: List[str]
    embeddings: torch.Tensor  # Shape: (num_sequences, embedding_dim)
    batch_idx: int


class AsyncInferenceRunner:
    """
    Single-GPU async inference runner with stream-based I/O overlap.

    Architecture:
    1. DataLoader workers parse FASTA files (CPU-only)
    2. Collate_fn tokenizes in main process
    3. Pinned tensors transferred to GPU (non_blocking)
    4. CUDA streams overlap H2D, compute, D2H
    5. GPU monitor tracks utilization and bottlenecks

    Attributes:
        device: CUDA device for inference
        model: Model with forward() method (e.g., ESM2WithFlashAttention)
        stream_processor: CUDA stream orchestrator for async ops
        monitor: GPU utilization and DataLoader metrics tracker
        checkpoint_dir: Base directory for checkpoint files (None if disabled)
        shard_checkpoint_dir: Shard-specific checkpoint directory (checkpoint_dir/shard_{rank})
        rank: Shard rank for per-GPU isolation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        enable_streams: bool = True,
        monitor_interval: float = 1.0,
        log_file: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        rank: int = 0,
        checkpoint_seq_threshold: int = 10000,
        checkpoint_time_threshold: float = 300.0,
        manifest: Optional['CheckpointManifest'] = None,
        input_fingerprint: str = "",
        model_config_hash: str = "",
    ):
        """
        Initialize async inference runner.

        Args:
            model: Model for inference (already on device)
            device: CUDA device (e.g., torch.device('cuda:0'))
            enable_streams: Use CUDA streams for I/O overlap (default: True)
            monitor_interval: GPU sampling interval in seconds (default: 1.0)
            log_file: Path for GPU metrics log (default: auto-generated)
            checkpoint_dir: Path for checkpoint files (None = checkpointing disabled)
            rank: Shard rank for per-GPU isolation (default: 0)
            checkpoint_seq_threshold: Sequence count trigger (default: 10000)
            checkpoint_time_threshold: Time trigger in seconds (default: 300.0)
            manifest: Optional CheckpointManifest for multi-GPU coordination
            input_fingerprint: SHA256 of input data for cross-run validation
            model_config_hash: Hash of model architecture/weights for compatibility checks

        Note:
            When checkpoint_dir is provided, shard-specific checkpoints will be
            written to checkpoint_dir/shard_{rank}/
        """
        self.model = model
        self.device = device
        self.enable_streams = enable_streams

        # Initialize stream processor for async GPU ops
        self.stream_processor = StreamProcessor(
            device=device,
            enable_streams=enable_streams,
            verbose=False
        )

        # Initialize GPU monitor with DataLoader tracking
        device_id = device.index if device.index is not None else 0
        self.monitor = NvitopMonitor(
            device_ids=[device_id],
            log_interval=monitor_interval,
            log_file=log_file
        )

        # Inference state
        self._batch_count = 0
        self._total_sequences = 0

        # Checkpointing setup
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")

        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.manifest = manifest

        if checkpoint_dir is not None:
            # Create shard-specific directory (always, regardless of manifest)
            self.shard_checkpoint_dir = checkpoint_dir / f"shard_{rank}"
            self.shard_checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Initialize checkpoint trigger
            self.trigger = CheckpointTrigger(
                seq_threshold=checkpoint_seq_threshold,
                time_threshold_sec=checkpoint_time_threshold
            )

            # Initialize async checkpoint writer with manifest integration
            self.writer = AsyncCheckpointWriter(
                max_workers=1,
                manifest=manifest,
                rank=rank
            )

            # Initialize accumulator state
            self._ckpt_embeddings: List[np.ndarray] = []
            self._ckpt_ids: List[str] = []
            self._ckpt_batch_idx: int = 0
            self._last_packing_stats: Dict[str, Any] = {}

            # Store metadata for checkpoints
            self._input_fingerprint = input_fingerprint
            # Note: _input_fingerprint and _model_config_hash will be validated
            # on resume to ensure checkpoint compatibility with current run.
            # Validation logic to be implemented in future commit.
            self._model_config_hash = model_config_hash or self._compute_model_config_hash()

            logger.info(
                f"Checkpointing enabled: shard {rank}, "
                f"seq_threshold={checkpoint_seq_threshold}, "
                f"time_threshold={checkpoint_time_threshold}s"
            )
        else:
            self.trigger = None
            self.writer = None
            self.shard_checkpoint_dir = None
            self._ckpt_embeddings = []
            self._ckpt_ids = []
            self._ckpt_batch_idx = 0
            self._last_packing_stats = {}
            self._input_fingerprint = input_fingerprint
            self._model_config_hash = model_config_hash or self._compute_model_config_hash()

        logger.info(
            f"AsyncInferenceRunner initialized on {device} "
            f"(streams={'enabled' if enable_streams else 'disabled'})"
        )

    @property
    def _checkpointing_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self.checkpoint_dir is not None

    def _compute_model_config_hash(self) -> str:
        """Compute lightweight model config hash from dtype + parameter count.

        Returns:
            SHA256 hex digest truncated to 16 chars
        """
        params = list(self.model.parameters())
        if not params:
            raise ValueError("Model has no parameters to compute config hash")
        model_dtype = str(params[0].dtype)
        param_count = sum(p.numel() for p in params)

        # Compute hash of dtype + param_count
        fingerprint = f"{model_dtype}_{param_count}"
        hash_obj = hashlib.sha256(fingerprint.encode('utf-8'))
        return hash_obj.hexdigest()[:16]

    def _validate_pinned_memory(self, batch: Dict[str, Any]) -> None:
        """
        FIX 5: Validate tensors are actually pinned (critical for performance).

        Called on first batch to ensure pin_memory=True is working.
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if not value.is_pinned():
                    logger.warning(
                        f"Tensor '{key}' is NOT pinned! "
                        f"Ensure DataLoader has pin_memory=True. "
                        f"Performance will be degraded."
                    )
                else:
                    logger.debug(f"Tensor '{key}' is pinned correctly")

    def _transfer_to_gpu(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Transfer batch tensors to GPU with non_blocking for pinned memory."""
        gpu_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                gpu_batch[key] = value.to(self.device, non_blocking=True)
            else:
                gpu_batch[key] = value  # Keep non-tensor data as-is
        return gpu_batch

    def _run_inference(self, gpu_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run model inference on GPU batch.

        FIX 7: Embeddings are computed in FP16 (via model.half() or autocast)
        but returned as FP32 for stability. Conversion happens in _extract_embeddings.

        FIX 3: Phase 5 uses standard attention (padded). Phase 6 will replace
        this with FlashAttention varlen (flash_attn_varlen_func with cu_seqlens).
        """
        with torch.no_grad():
            # ESM-2 forward expects tokens tensor
            input_ids = gpu_batch['input_ids']

            # Kill switch for emergency rollback (Gap 2)
            env = get_env_config()
            DISABLE_PACKING = env.disable_packing

            if 'cu_seqlens' in gpu_batch and not DISABLE_PACKING:
                # PHASE 6: Packed format with FlashAttention varlen
                cu_seqlens = gpu_batch['cu_seqlens']
                max_seqlen = gpu_batch['max_seqlen']

                # Forward pass with packed sequences
                outputs = self.model.forward_packed(
                    input_ids=input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    repr_layers=[36]
                )
                # Packed output shape: [total_tokens, hidden_dim]
                representations = outputs['representations'][36]

                logger.debug(
                    f"Packed inference: {len(gpu_batch['sequence_ids'])} sequences, "
                    f"{input_ids.numel()} tokens, max_seqlen={max_seqlen}"
                )

                # Check numerical stability (catches FP16 overflow)
                input_ids_list = gpu_batch.get('sequence_ids', [])
                check_numerical_stability(representations, context=f"batch_{self._batch_count}_seqs_{len(input_ids_list)}")
                return representations
            else:
                # Unpacked path (fallback):
                # - When VIRNUCPRO_DISABLE_PACKING=true (emergency rollback)
                # - When enable_packing=False in collator (testing)
                # - Before buffer fills (first <buffer_size sequences)
                # - When FlashAttention unavailable
                # Standard padded format (Phase 5 baseline)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

                # Run model forward
                # Note: Model may be in FP16 (model.half()) or use autocast
                outputs = self.model(input_ids, repr_layers=[36])
                representations = outputs['representations'][36]

                # Check numerical stability (catches FP16 overflow)
                input_ids_list = gpu_batch.get('sequence_ids', [])
                check_numerical_stability(representations, context=f"batch_{self._batch_count}_seqs_{len(input_ids_list)}")
                return representations

    def _extract_embeddings(
        self,
        representations: torch.Tensor,
        gpu_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract per-sequence embeddings from model output.

        FIX 7: Input representations may be FP16 (from model.half() or autocast).
        Output is always FP32 for numerical stability in downstream operations.

        Args:
            representations: Model output (batch_size, seq_len, hidden_dim)
                May be FP16 depending on model precision
            gpu_batch: Batch dict with tensors already on GPU
                (passed from process_batch to avoid double transfer)

        Returns:
            Embeddings tensor in FP32 (num_sequences, hidden_dim)
        """
        cu_seqlens = gpu_batch.get('cu_seqlens')
        sequence_ids = gpu_batch.get('sequence_ids', [])

        if cu_seqlens is not None and len(sequence_ids) > 0:
            # Packed format: representations shape is [total_tokens, hidden_dim]
            # NOT [batch, seq, hidden]
            # Each sequence in cu_seqlens includes BOS and EOS tokens:
            #   [BOS, aa_1, aa_2, ..., aa_L, EOS]
            # We mean-pool over aa_1..aa_L only (exclude BOS at start, EOS at position end-1 (cu_seqlens[i+1]-1))
            # This matches v1.0 behavior in features.py:224-231
            embeddings = []
            for i in range(len(sequence_ids)):
                seq_id = sequence_ids[i]
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_len = end - start  # Total including BOS + EOS
                if seq_len > 2:
                    # Mean pool positions start+1 to end-1 (exclude BOS and EOS)
                    seq_repr = representations[start + 1:end - 1].mean(dim=0)
                elif seq_len == 2:
                    # Only BOS + EOS, no actual sequence tokens - use EOS as fallback
                    logger.warning(
                        f"Sequence {seq_id} has only BOS+EOS tokens (seq_len={seq_len}), "
                        "falling back to EOS embedding"
                    )
                    seq_repr = representations[start + 1:end].mean(dim=0)
                else:
                    # Single token - use as-is
                    logger.warning(
                        f"Sequence {seq_id} has insufficient tokens (seq_len={seq_len}), "
                        "falling back to available token"
                    )
                    seq_repr = representations[start:end].mean(dim=0)
                embeddings.append(seq_repr)

            result = torch.stack(embeddings)
        else:
            # Single sequence or non-packed: mean pool (skip BOS)
            result = representations[0, 1:].mean(dim=0, keepdim=True)

        # FIX 7: Convert to FP32 for numerical stability
        # Even if model runs in FP16, embeddings stored/compared in FP32
        return result.float()

    def process_batch(self, batch: Dict[str, Any]) -> InferenceResult:
        """
        Process single batch through async pipeline.

        Args:
            batch: Batch from DataLoader (collated by VarlenCollator)
                MUST contain 'sequence_ids' key for traceability

        Returns:
            InferenceResult with sequence IDs and embeddings

        Raises:
            ValueError: If batch missing required 'sequence_ids' key

        Note:
            FIX 4: This method uses StreamProcessor.process_batch_async which
            must have signature: (batch, transfer_fn, compute_fn, retrieve_fn)
            Verify StreamProcessor interface matches before execution.
        """
        # FIX 8: Require sequence_ids for traceability (don't generate synthetic IDs)
        if 'sequence_ids' not in batch:
            raise ValueError(
                "Batch missing 'sequence_ids'. VarlenCollator must include "
                "sequence IDs in output for traceability."
            )

        sequence_ids = batch['sequence_ids']

        # FIX 2 & FIX 4: Single GPU transfer via StreamProcessor
        # StreamProcessor handles: transfer → compute → (optionally retrieve)
        # We keep representations on GPU to avoid D2H for cu_seqlens

        # Store gpu_batch reference from transfer_fn for _extract_embeddings
        gpu_batch_ref = {}

        def transfer_fn(b):
            gpu_b = self._transfer_to_gpu(b)
            gpu_batch_ref.update(gpu_b)  # Save for extraction step
            return gpu_b

        def compute_fn(gpu_b):
            return self._run_inference(gpu_b)

        # FIX 4: Verify StreamProcessor.process_batch_async signature
        # Expected: (batch, transfer_fn, compute_fn, retrieve_fn) -> result
        representations = self.stream_processor.process_batch_async(
            batch,
            transfer_fn=transfer_fn,
            compute_fn=compute_fn,
            retrieve_fn=None  # Keep on GPU for embedding extraction
        )

        # FIX 8: Synchronize compute stream before extracting embeddings
        # When retrieve_fn=None, process_batch_async returns without syncing.
        # Without this, _extract_embeddings may run on default stream before
        # representations are fully computed, causing race conditions.
        self.stream_processor.synchronize()

        # FIX 2: Use gpu_batch_ref (already on GPU) instead of re-transferring
        # Extract embeddings (representations and gpu_batch_ref both on GPU)
        embeddings = self._extract_embeddings(representations, gpu_batch_ref)

        # FIX 7: Embeddings already converted to FP32 in _extract_embeddings
        # Move to CPU for storage
        embeddings_cpu = embeddings.cpu()

        self._batch_count += 1
        self._total_sequences += len(sequence_ids)

        return InferenceResult(
            sequence_ids=sequence_ids,
            embeddings=embeddings_cpu,
            batch_idx=self._batch_count - 1
        )

    def _get_collator(self, dataloader: DataLoader):
        """Get the real collator from the DataLoader.

        create_async_dataloader stores the collator as dataloader.collator
        to prevent it from being pickled to worker subprocesses.

        Falls back to dataloader.collate_fn for backward compatibility
        with DataLoaders not created by create_async_dataloader.

        Args:
            dataloader: DataLoader to extract collator from

        Returns:
            Collator object, or None if no stateful collator found
        """
        # Prefer the explicitly stored collator (set by create_async_dataloader)
        collator = getattr(dataloader, 'collator', None)
        if collator is not None:
            return collator

        # Backward compatibility: check collate_fn directly
        if hasattr(dataloader, 'collate_fn') and hasattr(dataloader.collate_fn, 'flush'):
            return dataloader.collate_fn

        return None

    def _is_main_process_collation(self, dataloader: DataLoader) -> bool:
        """Check if collation should happen in the main process.

        Returns True when the DataLoader uses a passthrough collate_fn
        and has a stateful collator stored separately (the fix for the
        worker buffer data loss bug).

        Args:
            dataloader: DataLoader to check

        Returns:
            True if main process collation is needed
        """
        collator = getattr(dataloader, 'collator', None)
        if collator is None:
            return False

        # Check if collator is stateful (has buffer-based packing)
        return getattr(collator, 'enable_packing', False)

    def _resume_checkpoints(self, force_restart: bool) -> Optional[InferenceResult]:
        """Resume from checkpoints if available.

        Args:
            force_restart: If True, ignore checkpoints and start fresh

        Returns:
            InferenceResult with resumed data if available, None otherwise

        Side effects:
            Updates self._ckpt_embeddings, self._ckpt_ids, self._ckpt_batch_idx
        """
        if not self._checkpointing_enabled or force_restart:
            return None

        resumed_ids, resumed_embs, resume_batch_idx, corrupted_sequence_ids = resume_from_checkpoints(
            self.checkpoint_dir,
            self.rank,
            force_restart,
            self.manifest
        )

        # Handle corrupted sequences
        if corrupted_sequence_ids:
            logger.warning(
                f"Checkpoint corruption detected: {len(corrupted_sequence_ids)} sequences need reprocessing "
                f"(from batches after corruption point)\n"
            )
            logger.debug(f"Corrupted sequence IDs (first 10): {corrupted_sequence_ids[:10]}")

        # Yield resumed data if available
        if resumed_ids:
            # Store resumed data in accumulators
            self._ckpt_embeddings = [resumed_embs]
            self._ckpt_ids = resumed_ids
            self._ckpt_batch_idx = resume_batch_idx

            logger.info(
                f"Resuming shard {self.rank}: {len(resumed_ids)} sequences "
                f"from {resume_batch_idx} checkpoints"
            )

            # Yield resumed data as InferenceResult (batch_idx=-1 as marker)
            model_dtype = next(self.model.parameters()).dtype
            resumed_embeddings_tensor = torch.from_numpy(resumed_embs).to(model_dtype)
            return InferenceResult(
                sequence_ids=resumed_ids,
                embeddings=resumed_embeddings_tensor,
                batch_idx=-1
            )

        return None

    def _process_raw_item(self, raw_item: Any, collator: Any, main_process_collation: bool) -> Optional[Dict]:
        """Process raw item from DataLoader through collation.

        Args:
            raw_item: Raw item from DataLoader iterator
            collator: Collator instance (or None for legacy path)
            main_process_collation: Whether main-process collation is enabled

        Returns:
            Collated batch dict, or None if batch not ready (buffer not full)
        """
        if main_process_collation:
            # Main-process collation: raw_item is a dict from the dataset.
            # Pass through the collator which buffers and returns packed
            # batches when ready (or {} when buffer not full).
            batch = collator(raw_item)
        else:
            # Legacy path: collation already happened in DataLoader workers
            batch = raw_item

        # Skip empty batches (collator returns {} when buffer not full)
        if not batch or 'input_ids' not in batch:
            return None

        return batch

    def _record_batch_metrics(
        self,
        batch: Dict[str, Any],
        result: InferenceResult,
        batch_idx: int,
        fetch_time_ms: float
    ) -> None:
        """Record batch composition and packing metrics.

        Args:
            batch: Collated batch dict with tensors
            result: InferenceResult from process_batch
            batch_idx: Current batch index
            fetch_time_ms: Time to fetch batch from DataLoader (ms)
        """
        # Calculate batch composition metrics
        num_sequences = len(result.sequence_ids)
        tokens_in_batch = batch.get('input_ids', torch.tensor([])).numel()
        avg_seq_len = tokens_in_batch / num_sequences if num_sequences > 0 else 0
        max_seq_len = batch.get('max_seqlen', 0)

        # Compute packing efficiency for this batch
        packing_efficiency = None
        if 'cu_seqlens' in batch and 'max_seqlen' in batch:
            from virnucpro.data.packing import compute_batch_efficiency
            efficiency_stats = compute_batch_efficiency(
                num_tokens=tokens_in_batch,
                num_sequences=num_sequences,
                max_seqlen=batch['max_seqlen'],
                max_tokens_per_batch=batch.get('token_budget', 4096)
            )
            packing_efficiency = efficiency_stats['token_utilization']

        # Record DataLoader metrics
        self.monitor.record_dataloader_wait(
            wait_time_ms=fetch_time_ms,
            batch_idx=batch_idx,
            sequences_in_batch=num_sequences,
            tokens_in_batch=tokens_in_batch,
            avg_sequence_length=avg_seq_len,
            max_sequence_length=max_seq_len,
            packing_efficiency=packing_efficiency
        )

    def _log_progress(
        self,
        batch_idx: int,
        total_sequences_processed: int,
        dataloader: DataLoader,
        processing_start_time: float
    ) -> None:
        """Adaptive progress logging with ETA calculation.

        Logs at different frequencies based on progress:
        - First 10 batches: every batch
        - Batches 10-100: every 10 batches
        - After batch 100: every 100 batches

        Args:
            batch_idx: Current batch index
            total_sequences_processed: Cumulative sequences processed so far
            dataloader: DataLoader (used for dataset size estimation)
            processing_start_time: Time when processing started (from perf_counter)
        """
        # Adaptive logging frequency
        should_log_progress = False
        if batch_idx < 10:
            should_log_progress = True  # Log every batch during startup
        elif batch_idx < 100:
            should_log_progress = (batch_idx % 10 == 0)  # Every 10 batches
        else:
            should_log_progress = (batch_idx % 100 == 0)  # Every 100 batches

        if not should_log_progress:
            return

        dl_stats = self.monitor.get_dataloader_statistics()
        throughput = self.monitor.get_throughput()

        # Calculate elapsed time and ETA
        elapsed_time = time.perf_counter() - processing_start_time
        seq_per_sec = throughput.get('sequences_per_sec', 0)

        # Try to estimate total sequences from dataset if available
        estimated_total = None
        eta_str = "unknown"
        progress_pct = ""

        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
            try:
                estimated_total = len(dataloader.dataset)
                if estimated_total > 0:
                    progress_pct = f"{total_sequences_processed}/{estimated_total} ({100*total_sequences_processed/estimated_total:.1f}%) | "
                    remaining = estimated_total - total_sequences_processed
                    if seq_per_sec > 0:
                        eta_seconds = remaining / seq_per_sec
                        eta_minutes = eta_seconds / 60
                        eta_str = f"{eta_minutes:.1f}m" if eta_minutes < 60 else f"{eta_minutes/60:.1f}h"
            except:
                pass

        if not progress_pct:
            progress_pct = f"{total_sequences_processed:,} sequences | "

        logger.info(
            f"Batch {batch_idx}: {progress_pct}"
            f"{dl_stats.get('avg_packing_efficiency', 0):.1%} pack eff, "
            f"{throughput.get('tokens_per_sec', 0):,.0f} tok/s, "
            f"{seq_per_sec:.1f} seq/s, "
            f"ETA: {eta_str}"
        )

    def _accumulate_and_checkpoint(self, result: InferenceResult) -> None:
        """Accumulate embeddings and write checkpoint if triggered.

        Args:
            result: InferenceResult to accumulate
        """
        if not self._checkpointing_enabled:
            return

        # Accumulate embeddings (transfer to CPU before storing)
        self._ckpt_embeddings.append(result.embeddings.cpu().numpy())
        self._ckpt_ids.extend(result.sequence_ids)

        # Capture packing stats if available
        if hasattr(result, 'packing_stats') and result.packing_stats:
            self._last_packing_stats = result.packing_stats
        elif hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            # Extract from metadata if result carries it there
            packing_info = {}
            for key in ('packing_efficiency', 'token_count', 'buffer_size', 'token_budget'):
                if key in result.metadata:
                    packing_info[key] = result.metadata[key]
            if packing_info:
                self._last_packing_stats = packing_info

        # Check trigger
        should_checkpoint, reason = self.trigger.should_checkpoint(len(result.sequence_ids))
        if should_checkpoint and reason:
            try:
                self._write_checkpoint(reason)
            except Exception as e:
                logger.error(f"Checkpoint write failed: {e}. Continuing without checkpoint.")
            else:
                self.trigger.reset()

    def _flush_collator(self, collator: Any) -> Iterator[InferenceResult]:
        """Flush collator buffer and yield remaining batches.

        VarlenCollator accumulates sequences in buffer. This ensures no data loss
        for the last <buffer_size sequences.

        Args:
            collator: Collator instance (or None)

        Yields:
            InferenceResult for each flushed batch
        """
        if collator is None:
            logger.warning("No collator found on DataLoader - data may be lost if sequences were buffered!")
            return

        if not hasattr(collator, 'flush'):
            return

        logger.info("Flushing collator buffer for remaining sequences")
        flushed_batches = collator.flush()
        logger.info(f"Flush returned {len(flushed_batches)} batches")

        for batch in flushed_batches:
            if not batch or 'input_ids' not in batch:
                logger.warning("Skipping empty batch from flush")
                continue
            result = self.process_batch(batch)
            yield result

    def _finalize(self, processing_start_time: float) -> None:
        """Finalize inference run: checkpoint, shutdown, sync, and log stats.

        Called in finally block to ensure cleanup happens even on error.

        Args:
            processing_start_time: Time when processing started (from perf_counter)
        """
        # Final checkpoint (after loop completion)
        if self._checkpointing_enabled and self._ckpt_embeddings:
            try:
                self._write_checkpoint("final")
            except Exception as e:
                logger.error(f"Final checkpoint write failed: {e}. Data may be lost.")

        # Wait for all async checkpoint writes to complete
        if self._checkpointing_enabled and self.writer is not None:
            try:
                self.writer.wait_all(timeout=300)
                self.writer.shutdown()
                logger.info("All checkpoint writes completed")
            except Exception as e:
                logger.error(f"Error during checkpoint writer shutdown: {e}")

        # Synchronize streams before stopping
        self.stream_processor.synchronize()
        stats = self.monitor.stop_monitoring()

        throughput = self.monitor.get_throughput()
        dl_stats = self.monitor.get_dataloader_statistics()

        logger.info(
            f"Async inference complete: {self._total_sequences} sequences, "
            f"{throughput.get('sequences_per_sec', 0):.1f} seq/s, "
            f"{throughput.get('tokens_per_sec', 0):.0f} tokens/s"
        )
        if dl_stats:
            logger.info(
                f"DataLoader stats: avg_wait={dl_stats.get('avg_wait_time_ms', 0):.1f}ms, "
                f"max_wait={dl_stats.get('max_wait_time_ms', 0):.1f}ms, "
                f"packing_efficiency={dl_stats.get('packing_efficiency', 0):.2%}"
            )

    def run(
        self,
        dataloader: DataLoader,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        force_restart: bool = False
    ) -> Iterator[InferenceResult]:
        """
        Run inference on all batches from DataLoader.

        When the DataLoader has a stateful collator (stored as dataloader.collator
        by create_async_dataloader), raw items from workers are passed through
        the collator in the main process. This prevents data loss from PyTorch
        pickling stateful collators to worker subprocesses.

        Args:
            dataloader: Async DataLoader with prefetched batches.
                If created by create_async_dataloader with a stateful collator,
                raw items are collated in the main process.
            progress_callback: Optional callback(batch_idx, num_sequences)
            force_restart: If True, ignore checkpoints and start fresh (default: False)

        Yields:
            InferenceResult for each batch

        Raises:
            RuntimeError: If DataLoader fails or worker crashes
        """
        self.model.eval()
        self.monitor.start_monitoring()
        self.monitor.start_inference_timer()
        self.monitor.set_stage('inference')

        logger.info("Starting async inference loop")

        # Detect main-process collation mode
        main_process_collation = self._is_main_process_collation(dataloader)
        collator = self._get_collator(dataloader)

        if main_process_collation:
            logger.info(
                "Main-process collation enabled: raw items from DataLoader workers "
                "will be collated in main process to prevent buffer data loss"
            )

        # Resume from checkpoints if available
        resumed_result = self._resume_checkpoints(force_restart)
        if resumed_result:
            yield resumed_result

        # FIX 1: Track inter-batch arrival time (not processing time)
        last_batch_time = time.perf_counter()

        # Progress tracking
        total_sequences_processed = 0
        processing_start_time = time.perf_counter()

        # Log dataset info if available
        if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
            dataset_size = len(dataloader.dataset)
            collator_ref = collator if collator is not None else getattr(dataloader, 'collate_fn', None)
            logger.info(
                f"Starting inference on {dataset_size:,} sequences "
                f"(buffer size: {getattr(collator_ref, 'buffer_size', 'unknown')})"
            )
        else:
            logger.info("Starting inference (dataset size unknown)")

        try:
            # FIX 6: Wrap DataLoader iteration with exception handling
            dataloader_iter = iter(dataloader)
            batch_idx = 0

            while True:
                try:
                    # FIX 1: Measure time BEFORE fetching (inter-batch arrival)
                    fetch_start = time.perf_counter()
                    raw_item = next(dataloader_iter)
                    fetch_time_ms = (time.perf_counter() - fetch_start) * 1000

                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"DataLoader failed at batch {batch_idx}: {e}")
                    # Ensure monitor stops before re-raising
                    self.stream_processor.synchronize()
                    self.monitor.stop_monitoring()
                    raise RuntimeError(
                        f"DataLoader failed at batch {batch_idx}. "
                        f"Check worker logs and CUDA isolation."
                    ) from e

                # Process raw item through collation
                batch = self._process_raw_item(raw_item, collator, main_process_collation)
                if batch is None:
                    continue

                # FIX 5: Validate memory pinning (critical for performance)
                if batch_idx == 0:
                    if not main_process_collation:
                        # Only validate pinning when DataLoader handles collation
                        # (main-process collation produces non-pinned tensors)
                        self._validate_pinned_memory(batch)
                    # Calculate how long buffering took
                    buffering_time = time.perf_counter() - processing_start_time
                    logger.info(
                        f"First batch ready after {buffering_time:.1f}s buffering "
                        f"({len(batch['sequence_ids'])} sequences, "
                        f"{batch['input_ids'].numel()} tokens). Processing starting..."
                    )

                # Process batch
                result = self.process_batch(batch)

                # Record batch metrics
                self._record_batch_metrics(batch, result, batch_idx, fetch_time_ms)

                # Check for bottleneck every 10 batches
                if batch_idx % 10 == 0 and batch_idx > 0:
                    is_bottleneck, severity, avg_util = self.monitor.check_bottleneck()

                # Track cumulative progress
                total_sequences_processed += len(result.sequence_ids)

                # Adaptive progress logging
                self._log_progress(batch_idx, total_sequences_processed, dataloader, processing_start_time)

                # Progress callback
                if progress_callback:
                    progress_callback(batch_idx, len(result.sequence_ids))

                yield result

                # Checkpoint trigger (after yield, at batch boundaries)
                self._accumulate_and_checkpoint(result)

                batch_idx += 1

            # Flush collator buffer (handles last <buffer_size sequences)
            yield from self._flush_collator(collator)

        finally:
            self._finalize(processing_start_time)

    def _write_checkpoint(self, reason: str) -> None:
        """Write checkpoint to disk asynchronously.

        Args:
            reason: Trigger reason (e.g., "sequence_threshold", "time_threshold", "final")
        """
        # Return early if no accumulated data
        if not self._ckpt_embeddings:
            return

        # Concatenate accumulated embeddings
        embeddings = np.concatenate(self._ckpt_embeddings, axis=0)

        # Build checkpoint path
        checkpoint_path = self.shard_checkpoint_dir / f"batch_{self._ckpt_batch_idx:05d}.pt"

        # Determine GPU device from model
        device = next(self.model.parameters()).device

        # Synchronize CUDA before capturing memory metrics
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        # Build metadata dict
        env = get_env_config()
        metadata = {
            'batch_idx': self._ckpt_batch_idx,
            'num_sequences': len(self._ckpt_ids),
            'timestamp': datetime.utcnow().isoformat(),
            'trigger_reason': reason,
            'model_dtype': str(next(self.model.parameters()).dtype),
            'packing_enabled': not env.disable_packing,
            'gpu_memory_allocated_bytes': torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0,
            'gpu_memory_peak_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
            'input_fingerprint': self._input_fingerprint,
            'model_config_hash': self._model_config_hash,
            'packing_stats': self._last_packing_stats.copy() if self._last_packing_stats else {}
        }

        # Submit async write
        self.writer.write_checkpoint_async(
            checkpoint_path,
            embeddings,
            self._ckpt_ids,
            metadata
        )

        logger.info(
            f"Checkpoint {self._ckpt_batch_idx} queued: {len(self._ckpt_ids)} sequences, reason={reason}"
        )

        # Reset accumulators before increment to avoid data loss on error
        self._ckpt_ids = []
        self._ckpt_embeddings = []
        self._ckpt_batch_idx += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            'total_batches': self._batch_count,
            'total_sequences': self._total_sequences,
            'gpu_stats': self.monitor.get_statistics(),
            'dataloader_stats': self.monitor.get_dataloader_statistics(),
            'throughput': self.monitor.get_throughput(),
        }


def run_async_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    enable_streams: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[InferenceResult]:
    """
    Convenience function to run async inference on all batches.

    This is a simple wrapper around AsyncInferenceRunner for common use cases.
    For more control, use AsyncInferenceRunner directly.

    Args:
        model: Model for inference (will be moved to device)
        dataloader: Async DataLoader with prefetched batches
        device: CUDA device for inference
        enable_streams: Use CUDA streams for I/O overlap
        progress_callback: Optional callback(batch_idx, num_sequences)

    Returns:
        List of InferenceResult for all batches

    Example:
        >>> from virnucpro.data import SequenceDataset, VarlenCollator, create_async_dataloader
        >>> from virnucpro.models.esm2_flash import load_esm2_model
        >>>
        >>> model, batch_converter = load_esm2_model(device='cuda:0')
        >>> dataset = SequenceDataset(fasta_files)
        >>> collator = VarlenCollator(batch_converter)
        >>> loader = create_async_dataloader(dataset, collator)
        >>>
        >>> results = run_async_inference(model, loader, torch.device('cuda:0'))
        >>> for result in results:
        ...     print(f'Batch {result.batch_idx}: {len(result.sequence_ids)} sequences')
    """
    runner = AsyncInferenceRunner(
        model=model,
        device=device,
        enable_streams=enable_streams,
    )

    results = list(runner.run(dataloader, progress_callback))

    # Log final statistics
    stats = runner.get_statistics()
    logger.info(f"Inference complete: {stats}")

    return results
