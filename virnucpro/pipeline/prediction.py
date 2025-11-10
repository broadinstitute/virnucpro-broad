"""Main prediction pipeline orchestration with checkpointing"""

from pathlib import Path
from typing import Optional
import logging
import torch

from virnucpro.core.checkpoint import CheckpointManager, PipelineStage
from virnucpro.core.config import Config
# Import other pipeline components as they're refactored

logger = logging.getLogger('virnucpro.pipeline.prediction')


def run_prediction(
    input_file: Path,
    model_path: Path,
    expected_length: int,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cleanup_intermediate: bool,
    resume: bool,
    show_progress: bool,
    config: Config
):
    """
    Main prediction pipeline orchestration.

    Args:
        input_file: Input FASTA file
        model_path: Path to trained model
        expected_length: Expected sequence length
        output_dir: Output directory
        device: PyTorch device
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        cleanup_intermediate: Whether to clean intermediate files
        resume: Whether to resume from checkpoint
        show_progress: Whether to show progress bars
        config: Configuration object
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

    # Load state (or create new)
    if resume:
        state = checkpoint_manager.load_state()
        start_stage = checkpoint_manager.find_resume_stage(state)

        if start_stage is None:
            logger.info("All stages already completed!")
            return
    else:
        state = checkpoint_manager._create_initial_state()
        start_stage = PipelineStage.CHUNKING

    # Define intermediate paths
    chunked_file = output_dir / f"{input_file.stem}_chunked{expected_length}.fa"
    nucleotide_file = output_dir / f"{input_file.stem}_nucleotide.fa"
    protein_file = output_dir / f"{input_file.stem}_protein.faa"

    try:
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

            from virnucpro.utils.sequence import identify_seq
            from Bio import SeqIO

            # Count sequences for progress bar
            records = list(SeqIO.parse(chunked_file, 'fasta'))
            num_sequences = len(records)

            with progress.create_sequence_bar(num_sequences, desc="Translating sequences") as pbar:
                with open(nucleotide_file, 'w') as dna_out, open(protein_file, 'w') as protein_out:
                    for record in records:
                        sequence = str(record.seq).upper()
                        seqid = record.id
                        result = identify_seq(seqid, sequence)

                        if result:
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

            sequences_per_file = config.get('prediction.sequences_per_file', 10000)

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

            sequences_per_file = config.get('prediction.sequences_per_file', 10000)

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

        # Stage 5: Nucleotide Feature Extraction
        if start_stage <= PipelineStage.NUCLEOTIDE_FEATURES or not checkpoint_manager.can_skip_stage(state, PipelineStage.NUCLEOTIDE_FEATURES):
            logger.info("=== Stage 5: Nucleotide Feature Extraction ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.NUCLEOTIDE_FEATURES)

            from virnucpro.pipeline.features import extract_dnabert_features

            nucleotide_feature_files = []

            # Extract DNABERT-S features
            logger.info("Extracting DNABERT-S features from nucleotide sequences")
            with progress.create_file_bar(len(nucleotide_files), desc="DNABERT-S extraction") as pbar:
                for nuc_file in nucleotide_files:
                    output_file = nuc_file.with_suffix('').with_suffix('_DNABERT_S.pt')
                    extract_dnabert_features(
                        nuc_file,
                        output_file,
                        device,
                        batch_size=batch_size
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

            # Extract ESM-2 features
            logger.info("Extracting ESM-2 features from protein sequences")
            truncation_length = config.get('features.esm.truncation_seq_length', 1024)
            toks_per_batch = config.get('features.esm.toks_per_batch', 2048)

            with progress.create_file_bar(len(protein_files), desc="ESM-2 extraction") as pbar:
                for pro_file in protein_files:
                    output_file = pro_file.with_suffix('').with_suffix('_ESM.pt')
                    extract_esm_features(
                        pro_file,
                        output_file,
                        device,
                        truncation_length=truncation_length,
                        toks_per_batch=toks_per_batch
                    )
                    protein_feature_files.append(output_file)
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

        # Stage 7: Feature Merging
        merged_dir = output_dir / f"{input_file.stem}_merged"

        if start_stage <= PipelineStage.FEATURE_MERGING or not checkpoint_manager.can_skip_stage(state, PipelineStage.FEATURE_MERGING):
            logger.info("=== Stage 7: Feature Merging ===")
            checkpoint_manager.mark_stage_started(state, PipelineStage.FEATURE_MERGING)

            from virnucpro.pipeline.features import merge_features

            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_feature_files = []

            with progress.create_file_bar(len(nucleotide_feature_files), desc="Merging features") as pbar:
                for nuc_feat, pro_feat in zip(nucleotide_feature_files, protein_feature_files):
                    # Generate output filename
                    merged_file = merged_dir / nuc_feat.name.replace('_DNABERT_S.pt', '_merged.pt')

                    merge_features(nuc_feat, pro_feat, merged_file)
                    merged_feature_files.append(merged_file)
                    pbar.update(1)

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

    except Exception as e:
        # Mark current stage as failed
        # (determine current stage from state)
        logger.exception("Pipeline failed")
        raise
