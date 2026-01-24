"""Main prediction pipeline orchestration with checkpointing"""

from pathlib import Path
from typing import Optional
import logging
import time
import os
import torch

from virnucpro.core.checkpoint import CheckpointManager, PipelineStage, atomic_save, has_done_marker, remove_done_marker
from virnucpro.core.checkpoint_validation import CheckpointError, CHECKPOINT_EXIT_CODE, load_failed_checkpoints
from virnucpro.core.config import Config
from virnucpro.pipeline.parallel import detect_cuda_devices
from virnucpro.pipeline.parallel_esm import assign_files_round_robin, process_esm_files_worker
from virnucpro.pipeline.parallel_dnabert import process_dnabert_files_worker, assign_files_by_sequences
from virnucpro.pipeline.parallel_merge import parallel_merge_with_progress
from virnucpro.pipeline.work_queue import BatchQueueManager
from virnucpro.cuda.memory_manager import MemoryManager, configure_memory_optimization
from virnucpro.data.dataloader_utils import create_optimized_dataloader, get_optimal_workers
from virnucpro.models.esm2_flash import load_esm2_model
# Import other pipeline components as they're refactored

logger = logging.getLogger('virnucpro.pipeline.prediction')


def run_prediction(
    input_file: Path,
    model_path: Path,
    expected_length: int,
    output_dir: Path,
    device: torch.device,
    dnabert_batch_size: int,
    parallel: bool,
    batch_size: int,
    num_workers: int,
    cleanup_intermediate: bool,
    resume: bool,
    show_progress: bool,
    config: Config,
    toks_per_batch: int = None,
    translation_threads: int = None,
    merge_threads: int = None,
    quiet: bool = False,
    gpus: str = None,
    skip_checkpoint_validation: bool = False,
    force_resume: bool = False,
    dataloader_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    expandable_segments: bool = False,
    cache_clear_interval: int = 100,
    cuda_streams: bool = True
) -> int:
    """
    Main prediction pipeline orchestration.

    Args:
        input_file: Input FASTA file
        model_path: Path to trained model
        expected_length: Expected sequence length
        output_dir: Output directory
        device: PyTorch device
        dnabert_batch_size: Batch size for DNABERT-S extraction (tokens per batch, default: 2048)
        parallel: Enable multi-GPU parallel processing (auto-enabled by CLI on multi-GPU systems)
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        cleanup_intermediate: Whether to clean intermediate files
        resume: Whether to resume from checkpoint
        show_progress: Whether to show progress bars
        config: Configuration object
        toks_per_batch: Tokens per batch for ESM-2 processing (optional)
        translation_threads: Number of CPU threads for six-frame translation (optional, default: all cores)
        merge_threads: Number of CPU threads for parallel embedding merge (optional, default: auto-detect)
        quiet: Disable dashboard and verbose logging (optional)
        gpus: GPU IDs to use (comma-separated, optional)
        skip_checkpoint_validation: Skip checkpoint validation (optional)
        force_resume: Force resume even if checkpoints corrupted (optional)
        dataloader_workers: Number of DataLoader workers (optional, auto-detect)
        pin_memory: Pin memory for faster GPU transfer (optional, auto-detect)
        expandable_segments: Enable expandable CUDA memory segments (optional, default: False)
        cache_clear_interval: Clear CUDA cache every N batches (optional, default: 100)
        cuda_streams: Use CUDA streams for I/O overlap (optional, default: True)

    Returns:
        Exit code: 0 for success, 1 for total failure, 2 for partial success, 4 for OOM
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress reporter
    from virnucpro.utils.progress import ProgressReporter
    progress = ProgressReporter(disable=not show_progress)

    # Initialize checkpointing
    checkpoint_dir = output_dir / config.get('checkpointing.checkpoint_dir', '.checkpoints')

    pipeline_config = {
        'expected_length': expected_length,
        'model_path': str(model_path),
        'batch_size': batch_size,
        'num_workers': num_workers
    }

    checkpoint_manager = CheckpointManager(checkpoint_dir, pipeline_config)

    # Initialize memory management (must be before any CUDA operations)
    memory_manager = None
    if torch.cuda.is_available():
        try:
            memory_manager = configure_memory_optimization(
                enable_expandable=expandable_segments,
                cache_interval=cache_clear_interval,
                verbose=not quiet
            )
            logger.info("Memory optimization initialized")

            # Log initial memory status
            if memory_manager and not quiet:
                stats = memory_manager.get_memory_stats()
                logger.info(f"  Initial GPU memory: {stats['allocated']:.2f}GB allocated, "
                           f"{stats['free']:.2f}GB free")
        except Exception as e:
            logger.warning(f"Memory optimization initialization failed: {e}")
            logger.warning("Continuing without memory optimization")
    else:
        logger.info("CUDA not available, skipping memory optimization")

    # Load state (or create new)
    if resume:
        state = checkpoint_manager.load_state()
        start_stage = checkpoint_manager.find_resume_stage(state)

        if start_stage is None:
            logger.info("All stages already completed!")
            return

        # Log resume summary with checkpoint status
        completed_stages = sum(
            1 for stage in PipelineStage
            if state['stages'][stage.name]['status'] == 'completed'
        )
        total_stages = len(PipelineStage)
        logger.info(f"=== Resuming Pipeline ===")
        logger.info(f"Progress: {completed_stages}/{total_stages} stages complete")
        logger.info(f"Resuming from stage: {start_stage.name}")

        # Show any failed checkpoints from previous runs
        failed_checkpoints = load_failed_checkpoints(checkpoint_dir)
        if failed_checkpoints:
            logger.warning(f"Found {len(failed_checkpoints)} failed checkpoints from previous runs:")
            for path, reason, timestamp in failed_checkpoints[:5]:  # Show first 5
                logger.warning(f"  {Path(path).name}: {reason}")
            if len(failed_checkpoints) > 5:
                logger.warning(f"  ... and {len(failed_checkpoints) - 5} more")
    else:
        state = checkpoint_manager._create_initial_state()
        start_stage = PipelineStage.CHUNKING

    # Define intermediate paths
    chunked_file = output_dir / f"{input_file.stem}_chunked{expected_length}.fa"
    nucleotide_file = output_dir / f"{input_file.stem}_nucleotide.fa"
    protein_file = output_dir / f"{input_file.stem}_protein.faa"

    try:
        # Wrap GPU operations for OOM handling
        # Stage 1: Chunking
        if start_stage == PipelineStage.CHUNKING or not checkpoint_manager.can_skip_stage(state, PipelineStage.CHUNKING):
            logger.info("=== Stage 1: Sequence Chunking ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.CHUNKING)

            # Chunk sequences with progress bar
            from virnucpro.utils.sequence import split_fasta_chunk
            from Bio import SeqIO

            # Count sequences for progress bar
            num_sequences = sum(1 for _ in SeqIO.parse(input_file, 'fasta'))

            # Show progress during chunking
            with progress.create_sequence_bar(num_sequences, desc="Chunking sequences") as pbar:
                split_fasta_chunk(input_file, chunked_file, expected_length)
                pbar.update(num_sequences)

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.CHUNKING,
                {'files': [str(chunked_file)]}
            )

        # Stage 2: Translation (Six-Frame Translation)
        if start_stage <= PipelineStage.TRANSLATION or not checkpoint_manager.can_skip_stage(state, PipelineStage.TRANSLATION):
            logger.info("=== Stage 2: Six-Frame Translation ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.TRANSLATION)

            # Determine whether to use parallel translation
            import os
            cpu_count = os.cpu_count() or 1
            use_parallel = False
            num_workers = translation_threads if translation_threads else cpu_count

            # Use parallel if explicitly requested or if multiple cores available
            if num_workers > 1:
                use_parallel = True
                logger.info(f"Using parallel translation with {num_workers} workers")
            else:
                logger.info("Using sequential translation (single core)")

            # Record start time for performance metrics
            start_time = time.time()
            sequences_processed = 0
            sequences_with_orfs = 0

            if use_parallel:
                from virnucpro.pipeline.parallel_translate import parallel_translate_with_progress
                try:
                    sequences_processed, sequences_with_orfs = parallel_translate_with_progress(
                        chunked_file,
                        nucleotide_file,
                        protein_file,
                        num_workers=num_workers,
                        show_progress=show_progress
                    )
                except Exception as e:
                    logger.error(f"Parallel translation failed: {e}")
                    logger.info("Falling back to sequential translation")
                    use_parallel = False

            if not use_parallel:
                # Sequential fallback
                from virnucpro.utils.sequence import identify_seq
                from Bio import SeqIO

                # Count sequences for progress bar
                records = list(SeqIO.parse(chunked_file, 'fasta'))
                num_sequences = len(records)

                with progress.create_sequence_bar(num_sequences, desc="Translating sequences") as pbar:
                    with open(nucleotide_file, 'w') as dna_out, open(protein_file, 'w') as protein_out:
                        for record in records:
                            sequences_processed += 1
                            sequence = str(record.seq).upper()
                            seqid = record.id
                            result = identify_seq(seqid, sequence)

                            if result:
                                sequences_with_orfs += 1
                                for item in result:
                                    if item.get('protein', '') != '':
                                        sequence_name = item['seqid']
                                        dna_sequence = item['nucleotide']
                                        protein_sequence = item['protein']

                                        dna_out.write(f'>{sequence_name}\n')
                                        dna_out.write(f'{dna_sequence}\n')

                                        protein_out.write(f'>{sequence_name}\n')
                                        protein_out.write(f'{protein_sequence}\n')

                            pbar.update(1)

            # Calculate performance metrics
            elapsed_time = time.time() - start_time
            sequences_per_sec = sequences_processed / elapsed_time if elapsed_time > 0 else 0

            logger.info(f"Translation complete: processed {sequences_processed:,} sequences in {elapsed_time:.1f}s ({sequences_per_sec:,.0f} seq/s)")
            logger.info(f"Found {sequences_with_orfs:,} sequences with valid ORFs across 6 frames")

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.TRANSLATION,
                {'nucleotide_file': str(nucleotide_file), 'protein_file': str(protein_file)}
            )

        # Stage 3: Nucleotide File Splitting
        nucleotide_split_dir = output_dir / f"{input_file.stem}_nucleotide"
        protein_split_dir = output_dir / f"{input_file.stem}_protein"

        if start_stage <= PipelineStage.NUCLEOTIDE_SPLITTING or not checkpoint_manager.can_skip_stage(state, PipelineStage.NUCLEOTIDE_SPLITTING):
            logger.info("=== Stage 3: Nucleotide File Splitting ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.NUCLEOTIDE_SPLITTING)

            from virnucpro.utils.file_utils import split_fasta_file
            from Bio import SeqIO

            sequences_per_file = config.get('prediction.sequences_per_file', 10000)

            # For multi-GPU, ensure we create enough files for parallelization
            if parallel:
                available_gpus = detect_cuda_devices()
                num_gpus = len(available_gpus) if available_gpus else 1
                if num_gpus > 1:
                    # Count sequences in nucleotide file
                    total_sequences = sum(1 for _ in SeqIO.parse(nucleotide_file, 'fasta'))
                    # Calculate sequences per file to create at least num_gpus * 2 files
                    # This ensures good load balancing via bin-packing
                    min_files = num_gpus * 2
                    adjusted_sequences_per_file = max(100, total_sequences // min_files)
                    if adjusted_sequences_per_file < sequences_per_file:
                        sequences_per_file = adjusted_sequences_per_file
                        logger.info(f"Adjusted sequences_per_file to {sequences_per_file} for multi-GPU load balancing ({total_sequences} sequences / {min_files} files)")

            # Split nucleotide file
            logger.info(f"Splitting nucleotide sequences into batches of {sequences_per_file}")
            nucleotide_files = split_fasta_file(
                nucleotide_file,
                nucleotide_split_dir,
                sequences_per_file=sequences_per_file,
                prefix="output"
            )

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.NUCLEOTIDE_SPLITTING,
                {'nucleotide_files': [str(f) for f in nucleotide_files]}
            )
        else:
            # Load from checkpoint
            nucleotide_files = [Path(f) for f in state['stages'][PipelineStage.NUCLEOTIDE_SPLITTING.name]['outputs']['nucleotide_files']]

        # Stage 4: Protein File Splitting
        if start_stage <= PipelineStage.PROTEIN_SPLITTING or not checkpoint_manager.can_skip_stage(state, PipelineStage.PROTEIN_SPLITTING):
            logger.info("=== Stage 4: Protein File Splitting ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.PROTEIN_SPLITTING)

            from virnucpro.utils.file_utils import split_fasta_file
            from Bio import SeqIO

            sequences_per_file = config.get('prediction.sequences_per_file', 10000)

            # For multi-GPU, ensure we create enough files for parallelization
            if parallel:
                available_gpus = detect_cuda_devices()
                num_gpus = len(available_gpus) if available_gpus else 1
                if num_gpus > 1:
                    # Count sequences in protein file
                    total_sequences = sum(1 for _ in SeqIO.parse(protein_file, 'fasta'))
                    # Calculate sequences per file to create at least num_gpus * 2 files
                    min_files = num_gpus * 2
                    adjusted_sequences_per_file = max(100, total_sequences // min_files)
                    if adjusted_sequences_per_file < sequences_per_file:
                        sequences_per_file = adjusted_sequences_per_file
                        logger.info(f"Adjusted sequences_per_file to {sequences_per_file} for multi-GPU load balancing ({total_sequences} sequences / {min_files} files)")

            # Split protein file
            logger.info(f"Splitting protein sequences into batches of {sequences_per_file}")
            protein_files = split_fasta_file(
                protein_file,
                protein_split_dir,
                sequences_per_file=sequences_per_file,
                prefix="output"
            )

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.PROTEIN_SPLITTING,
                {'protein_files': [str(f) for f in protein_files]}
            )
        else:
            # Load from checkpoint
            protein_files = [Path(f) for f in state['stages'][PipelineStage.PROTEIN_SPLITTING.name]['outputs']['protein_files']]

        # Stage 5: Nucleotide Feature Extraction (DNABERT-S)
        if start_stage <= PipelineStage.NUCLEOTIDE_FEATURES or not checkpoint_manager.can_skip_stage(state, PipelineStage.NUCLEOTIDE_FEATURES):
            logger.info("=== Stage 5: Nucleotide Feature Extraction (DNABERT-S) ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.NUCLEOTIDE_FEATURES)

            nucleotide_feature_files = []

            # Detect available GPUs for parallel processing
            available_gpus = detect_cuda_devices()
            num_gpus = len(available_gpus) if available_gpus else 1
            # Use parallel processing if multiple GPUs available and parallel mode enabled
            # Note: Works with single files - sequences are distributed across GPUs
            use_parallel = num_gpus > 1 and parallel

            # Log GPU capabilities and adjust batch size for BF16
            effective_dnabert_batch = dnabert_batch_size
            if torch.cuda.is_available() and available_gpus:
                for gpu_id in available_gpus:
                    device_name = torch.cuda.get_device_name(gpu_id)
                    capability = torch.cuda.get_device_capability(gpu_id)
                    compute_version = f"{capability[0]}.{capability[1]}"
                    bf16_enabled = capability[0] >= 8
                    logger.info(f"GPU {gpu_id} ({device_name}): Compute {compute_version}, BF16 {'enabled' if bf16_enabled else 'disabled'}")

                # Adjust batch size for BF16 (Ampere+ GPUs)
                if available_gpus and torch.cuda.get_device_capability(available_gpus[0])[0] >= 8:
                    # Auto-increase batch size for BF16 if using default
                    if dnabert_batch_size == 2048:
                        effective_dnabert_batch = 3072
                    logger.info(f"BF16 mixed precision enabled, using batch size {effective_dnabert_batch}")
                else:
                    logger.info(f"Using FP32 precision, batch size {effective_dnabert_batch}")

            if use_parallel:
                logger.info(f"Using {num_gpus} GPUs for DNABERT-S extraction")

                # Filter out files with existing outputs for checkpoint resume
                # Use .done markers for quick completion check without loading multi-GB checkpoints
                files_to_process = []
                complete_count = 0
                for nuc_file in nucleotide_files:
                    output_file = nuc_file.parent / f"{nuc_file.stem}_DNABERT_S.pt"

                    # Quick check: .done marker indicates completed checkpoint
                    if output_file.exists() and has_done_marker(output_file):
                        nucleotide_feature_files.append(output_file)
                        complete_count += 1
                        logger.debug(f"Skipping {nuc_file.name} (checkpoint complete with .done marker)")
                    elif output_file.exists() and not has_done_marker(output_file):
                        # Checkpoint exists but no .done marker - may be incomplete
                        logger.warning(f"Re-processing {nuc_file.name} (checkpoint missing .done marker)")
                        remove_done_marker(output_file)  # Defensive cleanup
                        files_to_process.append(nuc_file)
                    else:
                        files_to_process.append(nuc_file)

                if complete_count > 0:
                    logger.info(f"Resuming: {complete_count} DNABERT-S checkpoints complete, {len(files_to_process)} to process")

                # Skip parallel processing if no files to process
                if not files_to_process:
                    logger.info("All nucleotide feature files already exist, skipping extraction")
                else:
                    # Use bin-packing assignment by sequence count for balanced GPU utilization
                    file_assignments = assign_files_by_sequences(files_to_process, num_gpus)

                    # Create progress queue for live updates
                    import multiprocessing
                    import threading
                    ctx = multiprocessing.get_context('spawn')
                    progress_queue = ctx.Queue()

                    # Create and start dashboard
                    from virnucpro.pipeline.dashboard import monitor_progress, MultiGPUDashboard
                    total_files_per_gpu = {i: len(file_assignments[i]) for i in range(num_gpus)}
                    dashboard = MultiGPUDashboard(num_gpus, total_files_per_gpu)
                    dashboard.start()

                    # Start progress monitor thread
                    stop_event = threading.Event()
                    monitor_thread = threading.Thread(
                        target=monitor_progress,
                        args=(progress_queue, dashboard, stop_event),
                        daemon=True
                    )
                    monitor_thread.start()

                    try:
                        # Process with queue manager using DNABERT-S worker
                        queue_manager = BatchQueueManager(num_gpus, process_dnabert_files_worker, progress_queue=progress_queue)
                        processed, failed = queue_manager.process_files(
                            file_assignments,
                            toks_per_batch=effective_dnabert_batch,
                            output_dir=nucleotide_split_dir,
                            enable_streams=cuda_streams and torch.cuda.is_available()
                        )
                        nucleotide_feature_files.extend(processed)

                        # Clear cache after DNABERT-S stage if memory manager active
                        if memory_manager and memory_manager.should_clear_cache():
                            memory_manager.clear_cache()
                            if not quiet:
                                stats = memory_manager.get_memory_stats()
                                logger.info(f"  Post-DNABERT memory: {stats['allocated']:.2f}GB allocated, "
                                           f"{stats['free']:.2f}GB free")

                        # Log any failures
                        if failed:
                            logger.warning(f"Failed to process {len(failed)} DNABERT-S files")
                            for file_path, error in failed:
                                logger.error(f"  {file_path}: {error}")
                    finally:
                        # Stop monitor thread and complete dashboard
                        stop_event.set()
                        monitor_thread.join(timeout=1.0)
                        dashboard.complete_all()

                    # Validate that we have feature files
                    if not nucleotide_feature_files:
                        raise RuntimeError(
                            f"No nucleotide feature files produced or found. Expected {len(nucleotide_files)} files. "
                            f"Check that input files are not empty and feature extraction completed successfully."
                        )
            else:
                from virnucpro.pipeline.features import extract_dnabert_features

                logger.info("Using single GPU for DNABERT-S extraction")

                # Filter out files with existing outputs for checkpoint resume
                # Use .done markers for quick completion check without loading multi-GB checkpoints
                files_to_process = []
                complete_count = 0
                for nuc_file in nucleotide_files:
                    output_file = nuc_file.parent / f"{nuc_file.stem}_DNABERT_S.pt"

                    # Quick check: .done marker indicates completed checkpoint
                    if output_file.exists() and has_done_marker(output_file):
                        nucleotide_feature_files.append(output_file)
                        complete_count += 1
                        logger.debug(f"Skipping {nuc_file.name} (checkpoint complete with .done marker)")
                    elif output_file.exists() and not has_done_marker(output_file):
                        # Checkpoint exists but no .done marker - may be incomplete or 0 bytes
                        logger.warning(f"Re-processing {nuc_file.name} (checkpoint missing .done marker or invalid)")
                        remove_done_marker(output_file)  # Defensive cleanup
                        files_to_process.append(nuc_file)
                    else:
                        files_to_process.append(nuc_file)

                if complete_count > 0:
                    logger.info(f"Resuming: {complete_count} DNABERT-S checkpoints complete, {len(files_to_process)} to process")

                with progress.create_file_bar(len(files_to_process), desc="DNABERT-S extraction") as pbar:
                    for nuc_file in files_to_process:
                        output_file = nuc_file.parent / f"{nuc_file.stem}_DNABERT_S.pt"
                        extract_dnabert_features(
                            nuc_file,
                            output_file,
                            device,
                            batch_size=effective_dnabert_batch
                        )
                        nucleotide_feature_files.append(output_file)
                        pbar.update(1)
                        pbar.set_postfix_str(f"Current: {nuc_file.name}")

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.NUCLEOTIDE_FEATURES,
                {'nucleotide_features': [str(f) for f in nucleotide_feature_files]}
            )
        else:
            # Load from checkpoint
            nucleotide_feature_files = [Path(f) for f in state['stages'][PipelineStage.NUCLEOTIDE_FEATURES.name]['outputs']['nucleotide_features']]

        # Stage 6: Protein Feature Extraction
        if start_stage <= PipelineStage.PROTEIN_FEATURES or not checkpoint_manager.can_skip_stage(state, PipelineStage.PROTEIN_FEATURES):
            logger.info("=== Stage 6: Protein Feature Extraction ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.PROTEIN_FEATURES)

            from virnucpro.pipeline.features import extract_esm_features

            protein_feature_files = []
            failed_files = []

            # Extract ESM-2 features with multi-GPU support
            logger.info("Extracting ESM-2 features from protein sequences")
            truncation_length = config.get('features.esm.truncation_seq_length', 1024)
            if toks_per_batch is None:
                toks_per_batch = config.get('features.esm.toks_per_batch', 2048)

            # Detect available GPUs
            cuda_devices = detect_cuda_devices()
            num_gpus = len(cuda_devices) if cuda_devices else 1
            # Use parallel processing if multiple GPUs available and parallel mode enabled
            # Note: Works with single files - sequences are distributed across GPUs
            use_parallel = num_gpus > 1 and parallel

            # Log GPU capabilities and BF16 status
            if torch.cuda.is_available() and cuda_devices:
                for gpu_id in cuda_devices:
                    device_name = torch.cuda.get_device_name(gpu_id)
                    capability = torch.cuda.get_device_capability(gpu_id)
                    compute_version = f"{capability[0]}.{capability[1]}"
                    bf16_enabled = capability[0] >= 8
                    logger.info(f"GPU {gpu_id} ({device_name}): Compute {compute_version}, BF16 {'enabled' if bf16_enabled else 'disabled'}")

                # Log batch size based on BF16 status
                if cuda_devices and torch.cuda.get_device_capability(cuda_devices[0])[0] >= 8:
                    effective_batch = 3072 if toks_per_batch == 2048 else toks_per_batch
                    logger.info(f"BF16 mixed precision available, using batch size {effective_batch}")
                else:
                    logger.info(f"Using FP32 precision, batch size {toks_per_batch}")

            if use_parallel:
                logger.info(f"Using {num_gpus} GPUs for ESM-2 extraction")

                # Filter out files with existing outputs for checkpoint resume
                # Use .done markers for quick completion check without loading multi-GB checkpoints
                files_to_process = []
                complete_count = 0
                for pro_file in protein_files:
                    output_file = pro_file.parent / f"{pro_file.stem}_ESM.pt"

                    # Quick check: .done marker indicates completed checkpoint
                    if output_file.exists() and has_done_marker(output_file):
                        protein_feature_files.append(output_file)
                        complete_count += 1
                        logger.debug(f"Skipping {pro_file.name} (checkpoint complete with .done marker)")
                    elif output_file.exists() and not has_done_marker(output_file):
                        # Checkpoint exists but no .done marker - may be incomplete
                        logger.warning(f"Re-processing {pro_file.name} (checkpoint missing .done marker)")
                        remove_done_marker(output_file)  # Defensive cleanup
                        files_to_process.append(pro_file)
                    else:
                        files_to_process.append(pro_file)

                if complete_count > 0:
                    logger.info(f"Resuming: {complete_count} ESM-2 checkpoints complete, {len(files_to_process)} to process")

                # Skip parallel processing if no files to process
                if not files_to_process:
                    logger.info("All ESM-2 feature files already exist, skipping extraction")
                else:
                    # Assign files round-robin across GPUs
                    file_assignments = assign_files_round_robin(files_to_process, num_gpus)

                    # Create progress queue for live updates
                    import multiprocessing
                    import threading
                    ctx = multiprocessing.get_context('spawn')
                    progress_queue = ctx.Queue()

                    # Create and start dashboard
                    from virnucpro.pipeline.dashboard import monitor_progress, MultiGPUDashboard
                    total_files_per_gpu = {i: len(file_assignments[i]) for i in range(num_gpus)}
                    dashboard = MultiGPUDashboard(num_gpus, total_files_per_gpu)
                    dashboard.start()

                    # Start progress monitor thread
                    stop_event = threading.Event()
                    monitor_thread = threading.Thread(
                        target=monitor_progress,
                        args=(progress_queue, dashboard, stop_event),
                        daemon=True
                    )
                    monitor_thread.start()

                    try:
                        # Process with queue manager
                        queue_manager = BatchQueueManager(num_gpus, process_esm_files_worker, progress_queue=progress_queue)
                        processed, failed = queue_manager.process_files(
                            file_assignments,
                            toks_per_batch=toks_per_batch,
                            output_dir=protein_split_dir,
                            enable_streams=cuda_streams and torch.cuda.is_available()
                        )

                        protein_feature_files.extend(processed)
                        failed_files.extend(failed)

                        # Clear cache after ESM-2 stage if memory manager active
                        if memory_manager and memory_manager.should_clear_cache():
                            memory_manager.clear_cache()
                            if not quiet:
                                stats = memory_manager.get_memory_stats()
                                logger.info(f"  Post-ESM-2 memory: {stats['allocated']:.2f}GB allocated, "
                                           f"{stats['free']:.2f}GB free")
                    finally:
                        # Stop monitor thread and complete dashboard
                        stop_event.set()
                        monitor_thread.join(timeout=1.0)
                        dashboard.complete_all()

                    # Log failures
                    if failed:
                        failed_file_path = output_dir / "failed_files.txt"
                        with open(failed_file_path, 'w') as f:
                            for file_path, error in failed:
                                f.write(f"{file_path}|ESM-2|{error}\n")
                        logger.warning(f"Failed to process {len(failed)} files, see {failed_file_path}")
            else:
                logger.info("Using single GPU for ESM-2 extraction")
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                # Filter out files with existing outputs for checkpoint resume
                # Use .done markers for quick completion check without loading multi-GB checkpoints
                files_to_process = []
                complete_count = 0
                for pro_file in protein_files:
                    output_file = pro_file.parent / f"{pro_file.stem}_ESM.pt"

                    # Quick check: .done marker indicates completed checkpoint
                    if output_file.exists() and has_done_marker(output_file):
                        protein_feature_files.append(output_file)
                        complete_count += 1
                        logger.debug(f"Skipping {pro_file.name} (checkpoint complete with .done marker)")
                    elif output_file.exists() and not has_done_marker(output_file):
                        # Checkpoint exists but no .done marker - may be incomplete
                        logger.warning(f"Re-processing {pro_file.name} (checkpoint missing .done marker)")
                        remove_done_marker(output_file)  # Defensive cleanup
                        files_to_process.append(pro_file)
                    else:
                        files_to_process.append(pro_file)

                if complete_count > 0:
                    logger.info(f"Resuming: {complete_count} ESM-2 checkpoints complete, {len(files_to_process)} to process")

                with progress.create_file_bar(len(files_to_process), desc="ESM-2 extraction") as pbar:
                    for pro_file in files_to_process:
                        # Construct output filename: output_0.fa -> output_0_ESM.pt
                        output_file = pro_file.parent / f"{pro_file.stem}_ESM.pt"
                        try:
                            extract_esm_features(
                                pro_file,
                                output_file,
                                device,
                                truncation_length=truncation_length,
                                toks_per_batch=toks_per_batch
                            )
                            protein_feature_files.append(output_file)
                        except Exception as e:
                            logger.error(f"Failed to process {pro_file.name}: {e}")
                            failed_files.append((pro_file, str(e)))
                        pbar.update(1)
                        pbar.set_postfix_str(f"Current: {pro_file.name}")

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.PROTEIN_FEATURES,
                {'protein_features': [str(f) for f in protein_feature_files]}
            )
        else:
            # Load from checkpoint
            protein_feature_files = [Path(f) for f in state['stages'][PipelineStage.PROTEIN_FEATURES.name]['outputs']['protein_features']]
            failed_files = []

        # Stage 7: Feature Merging
        merged_dir = output_dir / f"{input_file.stem}_merged"

        if start_stage <= PipelineStage.FEATURE_MERGING or not checkpoint_manager.can_skip_stage(state, PipelineStage.FEATURE_MERGING):
            logger.info("=== Stage 7: Feature Merging ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.FEATURE_MERGING)

            from virnucpro.pipeline.features import merge_features

            merged_dir.mkdir(parents=True, exist_ok=True)

            # Determine if we should use parallel merge
            # Use parallel merge when workload benefits: multiple files to merge (including auto-split
            # files from Phase 2) and multiple cores available. Auto-splitting creates multiple files
            # from single input for GPU load balancing, and these benefit from parallel merge.
            num_merge_threads = merge_threads if merge_threads else os.cpu_count()
            use_parallel = num_merge_threads > 1 and len(nucleotide_feature_files) > 1

            if use_parallel:
                logger.info(f"Using parallel merge with {num_merge_threads} workers ({len(nucleotide_feature_files)} files to merge)")
                merged_feature_files, failed_pairs = parallel_merge_with_progress(
                    nucleotide_feature_files,
                    protein_feature_files,
                    merged_dir,
                    num_workers=num_merge_threads,
                    show_progress=show_progress
                )
                if failed_pairs:
                    logger.warning(f"Failed to merge {len(failed_pairs)} file pairs")
            else:
                # Fallback to sequential merge for single file or single core
                logger.info(f"Using sequential merge (single file or single core system)")
                merged_feature_files = []
                with progress.create_file_bar(len(nucleotide_feature_files), desc="Merging features") as pbar:
                    for nuc_feat, pro_feat in zip(nucleotide_feature_files, protein_feature_files):
                        # Generate output filename: output_0_DNABERT_S.pt -> output_0_merged.pt
                        base_name = nuc_feat.stem.replace('_DNABERT_S', '')
                        merged_file = merged_dir / f"{base_name}_merged.pt"

                        merge_features(nuc_feat, pro_feat, merged_file)
                        merged_feature_files.append(merged_file)
                        pbar.update(1)

            logger.info(f"Merged {len(merged_feature_files)} feature files")

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.FEATURE_MERGING,
                {'merged_files': [str(f) for f in merged_feature_files]}
            )
        else:
            # Load from checkpoint
            merged_feature_files = [Path(f) for f in state['stages'][PipelineStage.FEATURE_MERGING.name]['outputs']['merged_files']]

        # Stage 8: Prediction
        prediction_results_file = merged_dir / 'prediction_results.txt'

        if start_stage <= PipelineStage.PREDICTION or not checkpoint_manager.can_skip_stage(state, PipelineStage.PREDICTION):
            logger.info("=== Stage 8: Model Prediction ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.PREDICTION)

            from virnucpro.pipeline.predictor import predict_sequences

            # Run prediction
            predictions = predict_sequences(
                merged_feature_files,
                model_path,
                device,
                batch_size=batch_size,
                num_workers=num_workers
            )

            # Save results
            logger.info(f"Saving prediction results to {prediction_results_file}")
            with open(prediction_results_file, 'w') as f:
                f.write("Sequence_ID\tPrediction\tscore1\tscore2\n")
                for seq_id, prediction, score_0, score_1 in predictions:
                    f.write(f"{seq_id}\t{prediction}\t{score_0}\t{score_1}\n")

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.PREDICTION,
                {'prediction_file': str(prediction_results_file)}
            )

        # Stage 9: Consensus Scoring
        consensus_results_file = merged_dir / 'prediction_results_highestscore.csv'

        if start_stage <= PipelineStage.CONSENSUS or not checkpoint_manager.can_skip_stage(state, PipelineStage.CONSENSUS):
            logger.info("=== Stage 9: Consensus Scoring ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.CONSENSUS)

            from virnucpro.pipeline.predictor import compute_consensus
            import pandas as pd

            # Load predictions
            df = pd.read_csv(prediction_results_file, sep='\t')

            # Extract predictions
            predictions = [
                (row['Sequence_ID'], row['Prediction'], row['score1'], row['score2'])
                for _, row in df.iterrows()
            ]

            # Compute consensus
            consensus = compute_consensus(predictions)

            # Save consensus results
            logger.info(f"Saving consensus results to {consensus_results_file}")
            consensus_df = pd.DataFrame([
                {'Modified_ID': seq_id, 'Is_Virus': pred == 'virus',
                 'max_score_0': score_0, 'max_score_1': score_1}
                for seq_id, (pred, score_0, score_1) in consensus.items()
            ])
            consensus_df.to_csv(consensus_results_file, sep='\t', index=False)

            checkpoint_manager.mark_stage_completed(
                state,
                PipelineStage.CONSENSUS,
                {'consensus_file': str(consensus_results_file)}
            )

        # Final: Cleanup if requested
        if cleanup_intermediate:
            logger.info("Cleaning up intermediate files...")

            # Keep only final results
            keep_files = config.get('files.keep_files', [
                'prediction_results.txt',
                'prediction_results_highestscore.csv'
            ])

            # Remove intermediate directories
            import shutil
            for dir_path in [nucleotide_split_dir, protein_split_dir]:
                if dir_path.exists():
                    logger.info(f"Removing {dir_path}")
                    shutil.rmtree(dir_path)

            # Remove intermediate files
            for file_path in [chunked_file, nucleotide_file, protein_file]:
                if file_path.exists():
                    logger.info(f"Removing {file_path}")
                    file_path.unlink()

        logger.info("Pipeline completed successfully!")

        # Determine exit code based on failures
        if failed_files:
            logger.warning(f"Pipeline completed with {len(failed_files)} failures")
            return 2  # Partial success
        else:
            return 0  # Complete success

    except RuntimeError as e:
        # Handle OOM errors specifically
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            logger.error("=" * 60)
            logger.error("OUT OF MEMORY ERROR")
            logger.error("=" * 60)

            # Log memory diagnostics if manager available
            if memory_manager and torch.cuda.is_available():
                try:
                    stats = memory_manager.get_memory_stats()
                    logger.error(f"Memory at failure: {stats['allocated']:.2f}GB allocated, "
                               f"{stats['reserved']:.2f}GB reserved, {stats['free']:.2f}GB free")
                    frag_ratio = memory_manager.get_fragmentation_ratio()
                    logger.error(f"Fragmentation ratio: {frag_ratio:.2%}")
                except Exception:
                    pass

            logger.error("\nSuggestions:")
            logger.error("  1. Reduce batch size: --esm-batch-size 1024 or --dnabert-batch-size 1024")
            logger.error("  2. Enable expandable segments: --expandable-segments")
            logger.error("  3. Increase cache clearing: --cache-clear-interval 50")
            logger.error("  4. Use fewer GPUs or process files sequentially")
            logger.error("=" * 60)
            return 4  # OOM exit code
        else:
            # Other runtime errors
            logger.exception("Pipeline failed with runtime error")
            return 1
    except Exception as e:
        # Mark current stage as failed
        # (determine current stage from state)
        logger.exception("Pipeline failed")
        return 1  # Total failure
