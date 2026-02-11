#!/usr/bin/env python3
"""
Extract FastESM2-650M embeddings for 150bp training data.

This is the same pipeline as the test data extraction, but for training data.

Usage:
    python scripts/extract_150bp_training_embeddings.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
import time
from transformers import AutoTokenizer, AutoModel

from units import (
    extract_DNABERT_S,
    extract_fast_esm,
    merge_data,
    MERGED_DIM
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def patch_dnabert_s_triton():
    """Apply runtime patch for DNABERT-S Triton compatibility."""
    import glob
    import re

    logger.info("Checking for DNABERT-S Triton patch...")

    cache_patterns = [
        Path.home() / ".cache/huggingface/models--zhihan1996--DNABERT-S/snapshots/*/flash_attn_triton.py",
        Path("/root/.cache/huggingface/models--zhihan1996--DNABERT-S/snapshots/*/flash_attn_triton.py"),
        Path("/workspace/.cache/huggingface/models--zhihan1996--DNABERT-S/snapshots/*/flash_attn_triton.py"),
    ]

    files = []
    for pattern in cache_patterns:
        files.extend(glob.glob(str(pattern)))

    if not files:
        logger.info("DNABERT-S not in cache, downloading...")
        from transformers import AutoConfig
        AutoConfig.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)

        for pattern in cache_patterns:
            files.extend(glob.glob(str(pattern)))

    if not files:
        logger.warning("Could not find flash_attn_triton.py to patch")
        return False

    patched_count = 0
    for file_path in files:
        with open(file_path, 'r') as f:
            content = f.read()

        if 'tl.trans(k)' in content:
            patched_count += 1
            continue

        old_pattern = r'tl\.dot\(([^,]+),\s*([^,]+),\s*trans_b\s*=\s*True\)'
        new_code = r'tl.dot(\1, tl.trans(\2))'

        patched_content = re.sub(old_pattern, new_code, content)

        if patched_content != content:
            with open(file_path, 'w') as f:
                f.write(patched_content)
            logger.info(f"Patched: {file_path}")
            patched_count += 1

    return patched_count > 0


def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("150bp Training Data Embedding Extraction")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    patch_dnabert_s_triton()

    # Input/output paths
    viral_fasta = "train_150bp_viral.fasta"
    nonviral_fasta = "train_150bp_nonviral.fasta"

    output_dir = Path("data/train_150bp")
    output_dir.mkdir(parents=True, exist_ok=True)

    viral_dna_pt = output_dir / "viral_dna.pt"
    viral_protein_pt = output_dir / "viral_protein.pt"
    viral_merged_pt = output_dir / "viral_merged.pt"

    nonviral_dna_pt = output_dir / "nonviral_dna.pt"
    nonviral_protein_pt = output_dir / "nonviral_protein.pt"
    nonviral_merged_pt = output_dir / "nonviral_merged.pt"

    # Load models
    logger.info("\n" + "=" * 60)
    logger.info("Loading DNABERT-S model...")
    logger.info("=" * 60)
    dna_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    dna_model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)

    logger.info("\n" + "=" * 60)
    logger.info("Loading FastESM2-650M model...")
    logger.info("=" * 60)
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    protein_model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    protein_model = protein_model.to(device)
    protein_model.eval()

    # Process viral sequences
    logger.info("\n" + "=" * 60)
    logger.info("Processing VIRAL 150bp training data")
    logger.info("=" * 60)

    viral_start = time.time()

    logger.info("\n[1/3] Extracting DNABERT-S embeddings...")
    extract_DNABERT_S(
        viral_fasta,
        str(viral_dna_pt),
        model_loaded=True,
        tokenizer=dna_tokenizer,
        model=dna_model
    )
    logger.info(f"Saved: {viral_dna_pt}")

    logger.info("\n[2/3] Extracting FastESM2-650M protein embeddings...")
    extract_fast_esm(
        viral_fasta,
        out_file=str(viral_protein_pt),
        model=protein_model,
        tokenizer=protein_tokenizer,
        truncation_seq_length=1024,
        toks_per_batch=20480
    )
    logger.info(f"Saved: {viral_protein_pt}")

    logger.info("\n[3/3] Merging DNA + protein embeddings...")
    merge_data(
        str(viral_dna_pt),
        str(viral_protein_pt),
        str(viral_merged_pt),
        data_type="viral"
    )
    logger.info(f"Saved: {viral_merged_pt}")

    viral_elapsed = time.time() - viral_start

    # Validate
    viral_data = torch.load(viral_merged_pt, weights_only=False)
    logger.info(f"Viral merged shape: {viral_data['data'].shape}")
    assert viral_data['data'].shape[1] == MERGED_DIM, "Dimension mismatch!"

    # Process non-viral sequences
    logger.info("\n" + "=" * 60)
    logger.info("Processing NON-VIRAL 150bp training data")
    logger.info("=" * 60)

    nonviral_start = time.time()

    logger.info("\n[1/3] Extracting DNABERT-S embeddings...")
    extract_DNABERT_S(
        nonviral_fasta,
        str(nonviral_dna_pt),
        model_loaded=True,
        tokenizer=dna_tokenizer,
        model=dna_model
    )
    logger.info(f"Saved: {nonviral_dna_pt}")

    logger.info("\n[2/3] Extracting FastESM2-650M protein embeddings...")
    extract_fast_esm(
        nonviral_fasta,
        out_file=str(nonviral_protein_pt),
        model=protein_model,
        tokenizer=protein_tokenizer,
        truncation_seq_length=1024,
        toks_per_batch=20480
    )
    logger.info(f"Saved: {nonviral_protein_pt}")

    logger.info("\n[3/3] Merging DNA + protein embeddings...")
    merge_data(
        str(nonviral_dna_pt),
        str(nonviral_protein_pt),
        str(nonviral_merged_pt),
        data_type="host"
    )
    logger.info(f"Saved: {nonviral_merged_pt}")

    nonviral_elapsed = time.time() - nonviral_start

    # Validate
    nonviral_data = torch.load(nonviral_merged_pt, weights_only=False)
    logger.info(f"Non-viral merged shape: {nonviral_data['data'].shape}")
    assert nonviral_data['data'].shape[1] == MERGED_DIM, "Dimension mismatch!"

    total_elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Viral: {len(viral_data['ids'])} sequences")
    logger.info(f"Non-viral: {len(nonviral_data['ids'])} sequences")
    logger.info(f"Total: {len(viral_data['ids']) + len(nonviral_data['ids'])}")
    logger.info(f"Feature dimension: {MERGED_DIM}")
    logger.info(f"\nOutput directory: {output_dir}/")

    logger.info("\n" + "=" * 60)
    logger.info("RUNTIME")
    logger.info("=" * 60)
    logger.info(f"Viral: {viral_elapsed/60:.1f} minutes")
    logger.info(f"Non-viral: {nonviral_elapsed/60:.1f} minutes")
    logger.info(f"Total: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")

    logger.info("\nNext step: Fine-tune model with finetune_150bp.py")


if __name__ == '__main__':
    main()
