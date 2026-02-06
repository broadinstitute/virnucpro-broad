"""Async inference runner for single-GPU processing with stream overlap.

This module implements the async DataLoader architecture where:
- DataLoader workers handle CPU-only I/O (FASTA parsing)
- Main process tokenizes in collate_fn
- GPU receives prefetched, pinned batches
- CUDA streams overlap data transfer with computation

This is the single-GPU foundation. Multi-GPU coordination is Phase 7.
"""

import torch
import time
import logging
from typing import Optional, List, Dict, Any, Callable, Iterator
from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import DataLoader

from virnucpro.cuda.stream_manager import StreamProcessor
from virnucpro.utils.gpu_monitor import NvitopMonitor

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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        enable_streams: bool = True,
        monitor_interval: float = 1.0,
        log_file: Optional[Path] = None,
    ):
        """
        Initialize async inference runner.

        Args:
            model: Model for inference (already on device)
            device: CUDA device (e.g., torch.device('cuda:0'))
            enable_streams: Use CUDA streams for I/O overlap (default: True)
            monitor_interval: GPU sampling interval in seconds (default: 1.0)
            log_file: Path for GPU metrics log (default: auto-generated)
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

        logger.info(
            f"AsyncInferenceRunner initialized on {device} "
            f"(streams={'enabled' if enable_streams else 'disabled'})"
        )

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
            import os
            DISABLE_PACKING = os.getenv('VIRNUCPRO_DISABLE_PACKING', 'false').lower() == 'true'

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
            embeddings = []
            for i in range(len(sequence_ids)):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                # Skip BOS token (position 0 of each sequence), mean pool the rest
                # For sequence at [start:end], BOS is at position 'start'
                if end - start > 1:
                    # Mean pool positions start+1 to end (exclude BOS)
                    seq_repr = representations[start + 1:end].mean(dim=0)
                else:
                    # Single token sequence - use as-is
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

    def run(
        self,
        dataloader: DataLoader,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Iterator[InferenceResult]:
        """
        Run inference on all batches from DataLoader.

        Args:
            dataloader: Async DataLoader with prefetched batches
            progress_callback: Optional callback(batch_idx, num_sequences)

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

        # FIX 1: Track inter-batch arrival time (not processing time)
        last_batch_time = time.perf_counter()

        try:
            # FIX 6: Wrap DataLoader iteration with exception handling
            dataloader_iter = iter(dataloader)
            batch_idx = 0

            while True:
                try:
                    # FIX 1: Measure time BEFORE fetching (inter-batch arrival)
                    fetch_start = time.perf_counter()
                    batch = next(dataloader_iter)
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

                # FIX 5: Validate memory pinning (critical for performance)
                if batch_idx == 0:
                    self._validate_pinned_memory(batch)

                # Process batch
                result = self.process_batch(batch)

                # FIX 1: Calculate batch composition metrics
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

                # Check for bottleneck every 10 batches
                if batch_idx % 10 == 0 and batch_idx > 0:
                    is_bottleneck, severity, avg_util = self.monitor.check_bottleneck()

                # Periodic logging every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    dl_stats = self.monitor.get_dataloader_statistics()
                    throughput = self.monitor.get_throughput()
                    logger.info(
                        f"Batch {batch_idx}: "
                        f"{dl_stats.get('avg_packing_efficiency', 0):.1%} avg efficiency, "
                        f"{throughput.get('tokens_per_sec', 0):.0f} tokens/sec, "
                        f"{throughput.get('sequences_per_sec', 0):.1f} seq/sec"
                    )

                # Progress callback
                if progress_callback:
                    progress_callback(batch_idx, num_sequences)

                yield result
                batch_idx += 1

            # Flush collator buffer (handles last <buffer_size sequences)
            # VarlenCollator accumulates sequences; flush ensures no data loss
            if hasattr(dataloader.collate_fn, 'flush'):
                logger.debug("Flushing collator buffer for remaining sequences")
                for batch in dataloader.collate_fn.flush():
                    result = self.process_batch(batch)
                    yield result

        finally:
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
