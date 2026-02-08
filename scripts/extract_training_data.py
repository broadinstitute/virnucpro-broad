#!/usr/bin/env python3
"""Re-extract all training data with FastESM2_650 embeddings.

This script replaces the manual multi-step extraction workflow in features_extract.py
with a clean, resumable, single-command approach that produces FastESM2_650 protein
embeddings for all training data.

Usage:
    python scripts/extract_training_data.py

Prerequisites:
    - DNABERT-S embeddings must already be extracted (*.pt files exist)
    - CUDA-enabled GPU available
    - Running inside Docker container with FastESM2_650 model accessible
"""

import os
import sys
import json
import logging
import random
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from units import extract_fast_esm, extract_DNABERT_S, merge_data, validate_merge_inputs, PROTEIN_DIM, MERGED_DIM, split_fasta_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when extraction fails."""
    pass


def patch_dnabert_s_triton():
    """
    Runtime patch for DNABERT-S flash attention Triton API compatibility.

    Patches deprecated trans_b parameter in tl.dot() calls.
    Required for PyTorch 2.9 + Triton compatibility.

    Background: DNABERT-S's flash_attn_triton.py uses `tl.dot(q, k, trans_b=True)`
    which was deprecated in newer Triton versions. PyTorch 2.9 ships with Triton that
    removed this parameter. The fix replaces it with `tl.dot(q, tl.trans(k))` which
    is mathematically equivalent.

    TODO: Replace with forked model in Phase 5 maintenance cycle.

    Returns:
        bool: True if patch was applied or already present, False if patch failed

    Raises:
        ExtractionError: if model cannot be downloaded or patched
    """
    import glob
    import re

    logger.info("Applying DNABERT-S Triton compatibility patch...")

    # Try multiple cache locations (host vs Docker, transformers_modules vs models cache)
    cache_patterns = [
        # Transformers modules cache (older behavior)
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/*/flash_attn_triton.py"),
        "/root/.cache/huggingface/modules/transformers_modules/*/flash_attn_triton.py",
        # Model snapshots cache (current behavior with trust_remote_code)
        os.path.expanduser("~/.cache/huggingface/models--zhihan1996--DNABERT-S/snapshots/*/flash_attn_triton.py"),
        "/root/.cache/huggingface/models--zhihan1996--DNABERT-S/snapshots/*/flash_attn_triton.py",
    ]

    files = []
    for pattern in cache_patterns:
        files.extend(glob.glob(pattern))

    if not files:
        # Model not downloaded yet - trigger download then patch
        logger.info("DNABERT-S not in cache, downloading model...")
        from transformers import AutoConfig

        try:
            AutoConfig.from_pretrained(
                "zhihan1996/DNABERT-S",
                trust_remote_code=True
            )
        except Exception as e:
            raise ExtractionError(f"Failed to download DNABERT-S model: {str(e)}")

        # Try finding files again
        for pattern in cache_patterns:
            files.extend(glob.glob(pattern))

    if not files:
        # Debug: show what's actually in the cache
        for base in ["~/.cache/huggingface/modules/transformers_modules", "/root/.cache/huggingface/modules/transformers_modules"]:
            expanded = os.path.expanduser(base)
            if os.path.exists(expanded):
                logger.error(f"Cache directory exists: {expanded}")
                logger.error(f"Contents: {os.listdir(expanded)}")

        raise ExtractionError(
            f"Cannot find DNABERT-S flash_attn_triton.py in cache. Tried: {cache_patterns}"
        )

    patched_count = 0
    for filepath in files:
        # Only patch DNABERT-S files (not other models that might use flash attention)
        if "DNABERT-S" not in filepath and "dnabert" not in filepath.lower():
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        # Check if already patched
        if 'tl.trans(k)' in content and 'trans_b=True' not in content:
            logger.info(f"Already patched: {filepath}")
            patched_count += 1
            continue

        # Apply patch: replace trans_b parameter with tl.trans()
        original = content

        # Pattern 1: tl.dot(q, k, trans_b=True) - lowercase variables
        content = re.sub(
            r'tl\.dot\(([qQ]),\s*([kK]),\s*trans_b=True\)',
            r'tl.dot(\1, tl.trans(\2))',
            content
        )

        if content != original:
            try:
                with open(filepath, 'w') as f:
                    f.write(content)
                patched_count += 1
                logger.info(f"Patched: {filepath}")
            except Exception as e:
                raise ExtractionError(f"Failed to write patched file {filepath}: {str(e)}")
        else:
            logger.warning(f"No trans_b=True pattern found in {filepath}")

    if patched_count == 0:
        raise ExtractionError("Failed to apply Triton patch - no files were patched")

    logger.info(f"Triton patch applied to {patched_count} file(s)")
    return True


def split_identified_files(data_dir='./data/', sequences_per_file=10000):
    """
    Split monolithic identified FASTA files into 10,000-sequence chunks.

    This step is required before extraction. The make_train_dataset_300.py script
    creates monolithic *identified_nucleotide.fa and *identified_protein.fa files,
    but the extraction pipeline needs them split into chunks for parallel processing.

    Args:
        data_dir: directory containing identified FASTA files
        sequences_per_file: number of sequences per split file (default: 10000)
    """
    logger.info(f"Splitting identified FASTA files into {sequences_per_file}-sequence chunks")

    # Find all identified FASTA files
    identified_files = []
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('identified_nucleotide.fa') or filename.endswith('identified_protein.fa'):
                identified_files.append(os.path.join(root, filename))

    logger.info(f"Found {len(identified_files)} identified FASTA files to split")

    # Split each file
    for input_file in tqdm(identified_files, desc="Splitting files"):
        # Create output directory by replacing .fa with /
        output_dir = input_file.replace('.fa', '/')

        # Skip if already split (directory exists and has files)
        if os.path.exists(output_dir):
            split_files = [f for f in os.listdir(output_dir) if f.endswith('.fa')]
            if len(split_files) > 0:
                logger.debug(f"Skipping {input_file} - already split")
                continue

        # Split the file
        split_fasta_file(input_file, output_dir, sequences_per_file)
        logger.info(f"Split {input_file} into {output_dir}")

    logger.info("File splitting complete")


def discover_training_data(data_dir='./data/'):
    """
    Auto-discover all training FASTA files and return categorized lists.

    Mirrors logic from features_extract.py:
    - Walks data_dir to find *identified_nucleotide.fa files
    - Categorizes into viral vs host (vertebrate, protozoa, plant, invertebrate, fungi, bacteria, archaea)
    - Returns split FASTA sub-files from identified_nucleotide/ and identified_protein/ subdirectories

    Returns:
        tuple: (viral_files_dict, host_files_dict) where each dict contains:
            - 'nucleotide': list of nucleotide FASTA file paths
            - 'protein': list of protein FASTA file paths
    """
    logger.info(f"Discovering training data in {data_dir}")

    # Discover all *identified_nucleotide.fa files
    nucleotide_input_file_list = []
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith('identified_nucleotide.fa'):
                nucleotide_input_file_list.append(os.path.join(root, filename))

    logger.info(f"Found {len(nucleotide_input_file_list)} identified_nucleotide.fa files")

    # Categorize files
    viral_files = {'nucleotide': [], 'protein': []}
    host_files = {
        'vertebrate': {'nucleotide': [], 'protein': []},
        'protozoa': {'nucleotide': [], 'protein': []},
        'plant': {'nucleotide': [], 'protein': []},
        'invertebrate': {'nucleotide': [], 'protein': []},
        'fungi': {'nucleotide': [], 'protein': []},
        'bacteria': {'nucleotide': [], 'protein': []},
        'archaea': {'nucleotide': [], 'protein': []}
    }

    sequences_per_file = 10000

    # Process viral files
    for nucleotide_input_file in nucleotide_input_file_list:
        if nucleotide_input_file.startswith(f'{data_dir}viral'):
            nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
            protein_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein/')

            # Collect nucleotide split files
            if os.path.exists(nucleotide_output_dir):
                viral_nucleotide_files = [
                    os.path.join(nucleotide_output_dir, f)
                    for f in os.listdir(nucleotide_output_dir)
                    if os.path.isfile(os.path.join(nucleotide_output_dir, f))
                    and f.endswith('.fa')
                    and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
                ]
                viral_files['nucleotide'].extend(viral_nucleotide_files)

            # Collect protein split files
            if os.path.exists(protein_output_dir):
                viral_protein_files = [
                    os.path.join(protein_output_dir, f)
                    for f in os.listdir(protein_output_dir)
                    if os.path.isfile(os.path.join(protein_output_dir, f))
                    and f.endswith('.fa')
                    and sum(1 for _ in SeqIO.parse(os.path.join(protein_output_dir, f), "fasta")) == sequences_per_file
                ]
                viral_files['protein'].extend(viral_protein_files)

    # Process host files by category
    category_prefixes = {
        'vertebrate': f'{data_dir}vertebrate',
        'protozoa': f'{data_dir}protozoa',
        'plant': f'{data_dir}plant',
        'invertebrate': f'{data_dir}invertebrate',
        'fungi': f'{data_dir}fungi',
        'bacteria': f'{data_dir}bacteria',
        'archaea': f'{data_dir}archaea'
    }

    for category, prefix in category_prefixes.items():
        for nucleotide_input_file in nucleotide_input_file_list:
            if nucleotide_input_file.startswith(prefix):
                nucleotide_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_nucleotide/')
                protein_output_dir = nucleotide_input_file.replace('identified_nucleotide.fa', 'identified_protein/')

                # Collect nucleotide split files
                if os.path.exists(nucleotide_output_dir):
                    nucleotide_files = [
                        os.path.join(nucleotide_output_dir, f)
                        for f in os.listdir(nucleotide_output_dir)
                        if os.path.isfile(os.path.join(nucleotide_output_dir, f))
                        and f.endswith('.fa')
                        and sum(1 for _ in SeqIO.parse(os.path.join(nucleotide_output_dir, f), "fasta")) == sequences_per_file
                    ]
                    host_files[category]['nucleotide'].extend(nucleotide_files)

                # Collect protein split files
                if os.path.exists(protein_output_dir):
                    protein_files = [
                        os.path.join(protein_output_dir, f)
                        for f in os.listdir(protein_output_dir)
                        if os.path.isfile(os.path.join(protein_output_dir, f))
                        and f.endswith('.fa')
                        and sum(1 for _ in SeqIO.parse(os.path.join(protein_output_dir, f), "fasta")) == sequences_per_file
                    ]
                    host_files[category]['protein'].extend(protein_files)

    # Log discovery summary
    logger.info(f"Viral files: {len(viral_files['nucleotide'])} nucleotide, {len(viral_files['protein'])} protein")
    for category in category_prefixes.keys():
        logger.info(f"{category.capitalize()} files: {len(host_files[category]['nucleotide'])} nucleotide, {len(host_files[category]['protein'])} protein")

    return viral_files, host_files


def sample_host_files(viral_files, host_files):
    """
    Sample host files to match viral file count.

    Mirrors sampling logic from features_extract.py:
    - Select ceil(len(viral_files)/7) from each category
    - Downsample to len(viral_files) total
    - Uses random.seed(42) for consistency

    Args:
        viral_files: dict with 'nucleotide' and 'protein' lists
        host_files: dict of category dicts with 'nucleotide' and 'protein' lists

    Returns:
        dict: {'nucleotide': [...], 'protein': [...]} with sampled host files
    """
    random.seed(42)

    viral_count = len(viral_files['nucleotide'])
    sample_per_category = math.ceil(viral_count / 7)

    logger.info(f"Sampling {sample_per_category} files per host category (total target: {viral_count})")

    sampled_nucleotide = []
    sampled_protein = []

    categories = ['vertebrate', 'protozoa', 'plant', 'invertebrate', 'fungi', 'bacteria', 'archaea']

    for category in categories:
        category_nucleotide = host_files[category]['nucleotide']
        category_protein = host_files[category]['protein']

        if len(category_nucleotide) >= sample_per_category:
            selected_nucleotide = random.sample(category_nucleotide, sample_per_category)
            # Get corresponding protein files
            selected_protein = [f.replace('nucleotide', 'protein') for f in selected_nucleotide]

            sampled_nucleotide.extend(selected_nucleotide)
            sampled_protein.extend(selected_protein)
        else:
            logger.warning(f"Category {category} has only {len(category_nucleotide)} files, using all")
            sampled_nucleotide.extend(category_nucleotide)
            sampled_protein.extend(category_protein)

    # Downsample to match viral count exactly
    if len(sampled_nucleotide) > viral_count:
        indices = list(range(len(sampled_nucleotide)))
        random.shuffle(indices)
        selected_indices = indices[:viral_count]

        sampled_nucleotide = [sampled_nucleotide[i] for i in selected_indices]
        sampled_protein = [sampled_protein[i] for i in selected_indices]

    logger.info(f"Sampled {len(sampled_nucleotide)} host nucleotide files, {len(sampled_protein)} protein files")

    return {'nucleotide': sampled_nucleotide, 'protein': sampled_protein}


def load_checkpoint(checkpoint_path):
    """Load extraction checkpoint for resume capability."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint: {len(checkpoint.get('completed_files', []))} files completed")
        return checkpoint
    return {'completed_files': [], 'started_at': None, 'last_update': None}


def save_checkpoint(checkpoint_path, completed_files, started_at):
    """Save extraction checkpoint."""
    checkpoint = {
        'completed_files': completed_files,
        'started_at': started_at,
        'last_update': datetime.utcnow().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def extract_dnabert_all(viral_nucleotide_files, host_nucleotide_files):
    """
    Extract DNABERT-S embeddings for all nucleotide files.

    Args:
        viral_nucleotide_files: list of viral nucleotide FASTA paths
        host_nucleotide_files: list of host nucleotide FASTA paths

    Returns:
        dict: extraction statistics
    """
    all_nucleotide_files = viral_nucleotide_files + host_nucleotide_files
    total_files = len(all_nucleotide_files)

    logger.info(f"Extracting DNABERT-S embeddings for {total_files} nucleotide files")

    # Apply Triton compatibility patch before loading DNABERT-S
    patch_dnabert_s_triton()

    # Load DNABERT-S model
    from transformers import AutoTokenizer

    logger.info("Loading DNABERT-S model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-S",
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-S",
        trust_remote_code=True
    )
    model.cuda()
    model.eval()

    logger.info("DNABERT-S model loaded successfully")

    # Track statistics
    stats = {
        'total_files': total_files,
        'files_processed': 0,
        'files_skipped': 0,
        'total_time': 0
    }

    # Process each file with progress bar
    with tqdm(total=total_files, desc="Extracting DNABERT-S") as pbar:
        for nucleotide_file in all_nucleotide_files:
            output_file = f'{nucleotide_file.split(".fa")[0]}_DNABERT_S.pt'

            # Skip if already exists
            if os.path.exists(output_file):
                stats['files_skipped'] += 1
                pbar.update(1)
                continue

            # Extract embeddings
            try:
                start_time = datetime.now()

                extract_DNABERT_S(
                    input_file=nucleotide_file,
                    out_file=output_file,
                    model_loaded=True,
                    tokenizer=tokenizer,
                    model=model
                )

                elapsed = (datetime.now() - start_time).total_seconds()

                # Update statistics
                stats['files_processed'] += 1
                stats['total_time'] += elapsed

                logger.info(
                    f"[{stats['files_processed'] + stats['files_skipped']}/{total_files}] "
                    f"Extracted {nucleotide_file} ({elapsed:.1f}s)"
                )

                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to extract {nucleotide_file}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise ExtractionError(
                    f"DNABERT-S extraction failed for {nucleotide_file}. "
                    f"Error: {str(e)}"
                )

    # Calculate averages
    if stats['files_processed'] > 0:
        stats['avg_time_per_file'] = stats['total_time'] / stats['files_processed']
    else:
        stats['avg_time_per_file'] = 0

    logger.info(
        f"DNABERT-S extraction complete: {stats['files_processed']} processed, "
        f"{stats['files_skipped']} skipped"
    )

    return stats


def validate_prerequisites(viral_files, sampled_host_files):
    """
    Verify CUDA is available.

    Args:
        viral_files: dict with 'nucleotide' list
        sampled_host_files: dict with 'nucleotide' list

    Raises:
        ExtractionError: if prerequisites not met
    """
    logger.info("Validating prerequisites...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise ExtractionError("CUDA is not available. This script requires GPU support.")

    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")


def extract_all(viral_protein_files, host_protein_files, model, tokenizer, checkpoint_path):
    """
    Extract FastESM2_650 embeddings for all protein files.

    Args:
        viral_protein_files: list of viral protein FASTA paths
        host_protein_files: list of host protein FASTA paths
        model: loaded FastESM2_650 model
        tokenizer: loaded tokenizer
        checkpoint_path: path to checkpoint file

    Returns:
        dict: extraction statistics
    """
    all_protein_files = viral_protein_files + host_protein_files
    total_files = len(all_protein_files)

    logger.info(f"Extracting FastESM2_650 embeddings for {total_files} protein files")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed_files = set(checkpoint.get('completed_files', []))
    started_at = checkpoint.get('started_at') or datetime.utcnow().isoformat()

    # Track statistics
    stats = {
        'total_files': total_files,
        'total_sequences': 0,
        'total_time': 0,
        'files_processed': 0,
        'files_skipped': 0
    }

    # Process each file with progress bar
    with tqdm(total=total_files, desc="Extracting proteins") as pbar:
        for protein_file in all_protein_files:
            output_file = f'{protein_file.split(".fa")[0]}_ESM.pt'

            # Skip if already completed (checkpoint or file exists)
            if output_file in completed_files or os.path.exists(output_file):
                stats['files_skipped'] += 1
                pbar.update(1)
                continue

            # Extract embeddings
            try:
                start_time = datetime.now()

                proteins, data = extract_fast_esm(
                    fasta_file=protein_file,
                    out_file=output_file,
                    model=model,
                    tokenizer=tokenizer
                )

                elapsed = (datetime.now() - start_time).total_seconds()

                # Update statistics
                stats['files_processed'] += 1
                stats['total_sequences'] += len(proteins)
                stats['total_time'] += elapsed

                # Update checkpoint
                completed_files.add(output_file)
                save_checkpoint(checkpoint_path, list(completed_files), started_at)

                logger.info(
                    f"[{stats['files_processed'] + stats['files_skipped']}/{total_files}] "
                    f"Extracted {protein_file} ({len(proteins)} sequences, {elapsed:.1f}s)"
                )

                pbar.update(1)

            except Exception as e:
                logger.error(f"Failed to extract {protein_file}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise ExtractionError(
                    f"Extraction failed for {protein_file}. "
                    f"Error: {str(e)}"
                )

    # Calculate averages
    if stats['files_processed'] > 0:
        stats['avg_time_per_file'] = stats['total_time'] / stats['files_processed']
    else:
        stats['avg_time_per_file'] = 0

    if stats['total_sequences'] > 0:
        stats['avg_time_per_sequence'] = stats['total_time'] / stats['total_sequences']
    else:
        stats['avg_time_per_sequence'] = 0

    return stats


def merge_all(viral_files, sampled_host_files):
    """
    Merge DNABERT-S and protein embeddings for all files.

    Args:
        viral_files: dict with 'nucleotide' list
        sampled_host_files: dict with 'nucleotide' list

    Returns:
        dict: merge statistics
    """
    logger.info("Merging DNABERT-S and protein embeddings...")

    stats = {'viral_merged': 0, 'host_merged': 0, 'total_merged': 0}

    # Merge viral files
    viral_nucleotide_folder = None
    viral_protein_folder = None

    if viral_files['nucleotide']:
        # Get folders from first viral file
        sample_nucleotide = viral_files['nucleotide'][0]
        viral_nucleotide_folder = os.path.dirname(sample_nucleotide)
        viral_protein_folder = viral_nucleotide_folder.replace('identified_nucleotide/', 'identified_protein/')

        dnabert_files = [
            f for f in os.listdir(viral_nucleotide_folder)
            if os.path.isfile(os.path.join(viral_nucleotide_folder, f)) and f.endswith('.pt')
        ]

        for dnabert_file in tqdm(dnabert_files, desc="Merging viral files"):
            DNABERT_S_infile = os.path.join(viral_nucleotide_folder, dnabert_file)
            ESM_infile = os.path.join(
                viral_protein_folder,
                dnabert_file.split('_')[0] + '_' + dnabert_file.split('_')[1] + '_ESM.pt'
            )
            outfile = os.path.join(
                './data/data_merge/viral.1.1_merged/',
                dnabert_file.split('_')[0] + '_' + dnabert_file.split('_')[1] + '_merged.pt'
            )

            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            if not os.path.exists(outfile):
                try:
                    merge_data(DNABERT_S_infile, ESM_infile, outfile, 'viral')
                    stats['viral_merged'] += 1
                except Exception as e:
                    logger.error(f"Failed to merge {dnabert_file}: {str(e)}")
                    raise ExtractionError(f"Merge failed for viral file {dnabert_file}")

    # Merge host files
    host_nucleotide_files = sampled_host_files['nucleotide']
    host_protein_files = sampled_host_files['protein']

    # Create DNABERT-S and ESM file lists
    dnabert_features = [f'{f.split(".fa")[0]}_DNABERT_S.pt' for f in host_nucleotide_files]
    esm_features = [f'{f.split(".fa")[0]}_ESM.pt' for f in host_protein_files]

    # Create merged file paths
    merged_list = [
        item.replace('./data/', './data/data_merge/')
            .replace('.identified_nucleotide', '_merged')
            .replace('.fa', '_merged.pt')
        for item in host_nucleotide_files
    ]

    for dnabert_file, esm_file, merged_file in tqdm(
        zip(dnabert_features, esm_features, merged_list),
        total=len(merged_list),
        desc="Merging host files"
    ):
        output_folder = os.path.dirname(merged_file)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(merged_file):
            try:
                merge_data(dnabert_file, esm_file, merged_file, 'host')
                stats['host_merged'] += 1
            except Exception as e:
                logger.error(f"Failed to merge {merged_file}: {str(e)}")
                raise ExtractionError(f"Merge failed for host file {merged_file}")

    stats['total_merged'] = stats['viral_merged'] + stats['host_merged']
    logger.info(f"Merged {stats['viral_merged']} viral files, {stats['host_merged']} host files")

    return stats


def validate_all(data_dir='./data/'):
    """
    Post-extraction validation suite.

    Validates:
    1. All *_ESM.pt files have 1280-dim embeddings
    2. All *_merged.pt files have 2048-dim features
    3. No dimension mismatches

    Raises:
        ExtractionError: if validation fails
    """
    logger.info("Running post-extraction validation...")

    errors = []
    esm_files_checked = 0
    merged_files_checked = 0
    total_sequences = 0

    # Find all ESM files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Check _ESM.pt files
            if file.endswith('_ESM.pt'):
                try:
                    data = torch.load(file_path)
                    proteins = data['proteins']
                    embeddings = data['data']

                    for protein, embedding in zip(proteins, embeddings):
                        if embedding.shape != (PROTEIN_DIM,):
                            errors.append(
                                f"Dimension mismatch in {file_path}: "
                                f"protein {protein} has shape {embedding.shape}, expected ({PROTEIN_DIM},)"
                            )

                    esm_files_checked += 1
                    total_sequences += len(proteins)

                except Exception as e:
                    errors.append(f"Failed to load {file_path}: {str(e)}")

            # Check _merged.pt files
            elif file.endswith('_merged.pt'):
                try:
                    data = torch.load(file_path)
                    merged_tensor = data['data']

                    # Check second dimension is MERGED_DIM
                    if merged_tensor.dim() != 2:
                        errors.append(
                            f"Merged tensor in {file_path} has {merged_tensor.dim()} dimensions, expected 2"
                        )
                    elif merged_tensor.shape[1] != MERGED_DIM:
                        errors.append(
                            f"Merged tensor in {file_path} has shape {merged_tensor.shape}, "
                            f"expected second dimension to be {MERGED_DIM}"
                        )

                    merged_files_checked += 1

                except Exception as e:
                    errors.append(f"Failed to load {file_path}: {str(e)}")

    # Report results
    if errors:
        logger.error(f"Validation FAILED with {len(errors)} errors:")
        for error in errors[:20]:  # Show first 20
            logger.error(f"  - {error}")
        if len(errors) > 20:
            logger.error(f"  ... and {len(errors) - 20} more errors")
        raise ExtractionError(f"Validation failed with {len(errors)} dimension mismatches")

    logger.info(
        f"Validation PASSED: {esm_files_checked} ESM files, {merged_files_checked} merged files, "
        f"{total_sequences} sequences, all dimensions correct"
    )

    return {
        'esm_files': esm_files_checked,
        'merged_files': merged_files_checked,
        'total_sequences': total_sequences
    }


def main():
    """Main execution entry point."""
    logger.info("=" * 80)
    logger.info("FastESM2_650 Training Data Extraction")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        # Step 1: Split identified FASTA files into chunks
        logger.info("Step 1: Splitting identified FASTA files...")
        split_identified_files('./data/', sequences_per_file=10000)

        # Step 2: Discover training data
        logger.info("Step 2: Discovering training data...")
        viral_files, host_files = discover_training_data('./data/')

        # Step 3: Sample host files to match viral count
        logger.info("Step 3: Sampling host files...")
        sampled_host_files = sample_host_files(viral_files, host_files)

        # Step 4: Validate prerequisites
        logger.info("Step 4: Validating prerequisites...")
        validate_prerequisites(viral_files, sampled_host_files)

        # Step 5: Extract DNABERT-S embeddings
        logger.info("Step 5: Extracting DNABERT-S embeddings...")
        dnabert_stats = extract_dnabert_all(
            viral_files['nucleotide'],
            sampled_host_files['nucleotide']
        )

        # Step 6: Load FastESM2_650 model
        logger.info("Step 6: Loading FastESM2_650 model...")
        model = AutoModel.from_pretrained(
            "Synthyra/FastESM2_650",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).eval().cuda()
        tokenizer = model.tokenizer
        logger.info("Model loaded successfully")

        # Step 7: Extract protein embeddings
        logger.info("Step 7: Extracting protein embeddings...")
        checkpoint_path = './data/.extraction_checkpoint.json'
        extraction_stats = extract_all(
            viral_files['protein'],
            sampled_host_files['protein'],
            model,
            tokenizer,
            checkpoint_path
        )

        # Step 8: Merge embeddings
        logger.info("Step 8: Merging DNABERT-S and protein embeddings...")
        merge_stats = merge_all(viral_files, sampled_host_files)

        # Step 9: Validate all outputs
        logger.info("Step 9: Validating extracted embeddings...")
        validation_stats = validate_all('./data/')

        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()

        # Print summary
        logger.info("=" * 80)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info("")
        logger.info("DNABERT-S Extraction:")
        logger.info(f"  Files processed: {dnabert_stats['files_processed']}")
        logger.info(f"  Files skipped: {dnabert_stats['files_skipped']}")
        logger.info(f"  Avg time per file: {dnabert_stats['avg_time_per_file']:.1f}s")
        logger.info("")
        logger.info("FastESM2_650 Extraction:")
        logger.info(f"  Files processed: {extraction_stats['files_processed']}")
        logger.info(f"  Files skipped: {extraction_stats['files_skipped']}")
        logger.info(f"  Total sequences: {extraction_stats['total_sequences']}")
        logger.info(f"  Avg time per file: {extraction_stats['avg_time_per_file']:.1f}s")
        logger.info(f"  Avg time per sequence: {extraction_stats['avg_time_per_sequence']:.3f}s")
        logger.info("")
        logger.info("Merging:")
        logger.info(f"  Viral files merged: {merge_stats['viral_merged']}")
        logger.info(f"  Host files merged: {merge_stats['host_merged']}")
        logger.info("")
        logger.info("Validation:")
        logger.info(f"  {validation_stats['esm_files']} ESM files, {validation_stats['merged_files']} merged files")
        logger.info(f"  {validation_stats['total_sequences']} total sequences, all dimensions correct")
        logger.info("=" * 80)

        # Clean up checkpoint on success
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Checkpoint file removed")

    except ExtractionError as e:
        logger.error(f"Extraction failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
