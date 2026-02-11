#!/usr/bin/env python3
"""
Extract ESM2-3B embeddings for 150bp reads (50K sample) for comparison.

This extracts the larger ESM2-3B protein embeddings to compare performance
against FastESM2-650M on short reads.

Usage:
    python scripts/extract_150bp_embeddings_esm2_3b.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
import time
from transformers import AutoTokenizer, AutoModel

from units import extract_DNABERT_S

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


def merge_data_esm2_3b(dna_file, protein_file, output_file, data_type):
    """
    Custom merge function for ESM2-3B (768 DNA + 2560 protein = 3328).
    Bypasses PROTEIN_DIM validation in units.py.
    """
    logger.info(f"Merging {dna_file} + {protein_file} -> {output_file}")

    dna_data = torch.load(dna_file, weights_only=False)
    protein_data = torch.load(protein_file, weights_only=False)

    # Build lookup dicts
    nucleotide_dict = {}
    for nucleotide, data in zip(dna_data['nucleotide'], dna_data['data']):
        nucleotide_dict[nucleotide] = torch.tensor(data['mean_representation'])

    protein_dict = {}
    for protein, data in zip(protein_data['proteins'], protein_data['data']):
        protein_dict[protein] = data

    # Merge
    merged_data = []
    for item in dna_data['nucleotide']:
        if item in protein_dict and item in nucleotide_dict:
            nucleotide_tensor = nucleotide_dict[item]
            protein_tensor = protein_dict[item]

            # Validate dimensions
            assert nucleotide_tensor.shape == (768,), f"Expected 768-dim DNA, got {nucleotide_tensor.shape}"
            assert protein_tensor.shape == (2560,), f"Expected 2560-dim protein, got {protein_tensor.shape}"

            merged_feature = torch.cat((nucleotide_tensor, protein_tensor), dim=-1)
            assert merged_feature.shape == (3328,), f"Expected 3328-dim merged, got {merged_feature.shape}"

            merged_data.append(merged_feature)
        else:
            logger.warning(f"Skipping {item}: not found in both datasets")

    # Stack and save
    merged_data = torch.stack(merged_data)

    if data_type == 'host':
        merged_torch = {'ids': dna_data['nucleotide'], 'data': merged_data, 'labels': [0]}
    elif data_type == 'viral':
        merged_torch = {'ids': dna_data['nucleotide'], 'data': merged_data, 'labels': [1]}
    else:
        merged_torch = {'ids': dna_data['nucleotide'], 'data': merged_data}

    torch.save(merged_torch, output_file)
    logger.info(f"Saved merged data: {merged_data.shape}")


def extract_esm2_3b(fasta_file, out_file, model, tokenizer, device, toks_per_batch=20480):
    """Extract ESM2-3B embeddings (2560-dim)."""
    from Bio import SeqIO
    from tqdm import tqdm

    if Path(out_file).exists():
        logger.info(f"Output file {out_file} already exists, skipping...")
        return

    logger.info(f"Extracting ESM2-3B embeddings from {fasta_file}")

    records = list(SeqIO.parse(fasta_file, 'fasta'))
    sequences = [str(record.seq) for record in records]
    labels = [record.id for record in records]

    proteins = []
    data = []

    # Process in batches
    batch_size = 32  # Conservative batch size for 150bp reads

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch_seqs = sequences[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding='longest',
                truncation=True,
                max_length=1024
            )

            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pool over sequence length
            for j, label in enumerate(batch_labels):
                # Get sequence length (excluding padding)
                seq_len = attention_mask[j].sum().item()
                # Mean pool positions 1 to seq_len-1 (excluding BOS/EOS)
                embedding = outputs.last_hidden_state[j, 1:seq_len-1].mean(0)
                embedding = embedding.float().cpu()

                proteins.append(label)
                data.append(embedding)

    # Save
    torch.save({
        'proteins': proteins,
        'data': data
    }, out_file)

    logger.info(f"Saved {len(proteins)} embeddings to {out_file}")


def main():
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("150bp ESM2-3B Embedding Extraction (50K Sample)")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    patch_dnabert_s_triton()

    # Input/output paths
    viral_fasta = "test_150bp_viral_50k.fasta"
    nonviral_fasta = "test_150bp_nonviral_50k.fasta"

    output_dir = Path("data/test_150bp_50k_esm2_3b")
    output_dir.mkdir(parents=True, exist_ok=True)

    # We can reuse DNA embeddings from FastESM2-650M run
    viral_dna_pt = "data/test_150bp_50k/viral_dna.pt"
    nonviral_dna_pt = "data/test_150bp_50k/nonviral_dna.pt"

    viral_protein_pt = output_dir / "viral_protein.pt"
    viral_merged_pt = output_dir / "viral_merged.pt"

    nonviral_protein_pt = output_dir / "nonviral_protein.pt"
    nonviral_merged_pt = output_dir / "nonviral_merged.pt"

    # Load ESM2-3B model
    logger.info("\n" + "=" * 60)
    logger.info("Loading ESM2-3B model (2560-dim)...")
    logger.info("=" * 60)
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    protein_model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
    protein_model = protein_model.to(device)
    protein_model.eval()

    # Process viral sequences
    logger.info("\n" + "=" * 60)
    logger.info("Processing VIRAL 150bp reads")
    logger.info("=" * 60)

    viral_start = time.time()

    logger.info("\n[1/2] Extracting ESM2-3B protein embeddings...")
    extract_esm2_3b(
        viral_fasta,
        str(viral_protein_pt),
        protein_model,
        protein_tokenizer,
        device
    )

    logger.info("\n[2/2] Merging DNA + protein embeddings (768 + 2560 = 3328)...")
    merge_data_esm2_3b(
        viral_dna_pt,
        str(viral_protein_pt),
        str(viral_merged_pt),
        data_type="viral"
    )
    logger.info(f"Saved: {viral_merged_pt}")

    viral_elapsed = time.time() - viral_start

    # Validate
    viral_data = torch.load(viral_merged_pt, weights_only=False)
    logger.info(f"Viral merged shape: {viral_data['data'].shape}")
    logger.info(f"Expected: ({len(viral_data['ids'])}, 3328)")
    assert viral_data['data'].shape[1] == 3328, f"Expected 3328-dim, got {viral_data['data'].shape[1]}"

    # Process non-viral sequences
    logger.info("\n" + "=" * 60)
    logger.info("Processing NON-VIRAL 150bp reads")
    logger.info("=" * 60)

    nonviral_start = time.time()

    logger.info("\n[1/2] Extracting ESM2-3B protein embeddings...")
    extract_esm2_3b(
        nonviral_fasta,
        str(nonviral_protein_pt),
        protein_model,
        protein_tokenizer,
        device
    )

    logger.info("\n[2/2] Merging DNA + protein embeddings (768 + 2560 = 3328)...")
    merge_data_esm2_3b(
        nonviral_dna_pt,
        str(nonviral_protein_pt),
        str(nonviral_merged_pt),
        data_type="host"
    )
    logger.info(f"Saved: {nonviral_merged_pt}")

    nonviral_elapsed = time.time() - nonviral_start

    # Validate
    nonviral_data = torch.load(nonviral_merged_pt, weights_only=False)
    logger.info(f"Non-viral merged shape: {nonviral_data['data'].shape}")
    assert nonviral_data['data'].shape[1] == 3328, f"Expected 3328-dim, got {nonviral_data['data'].shape[1]}"

    total_elapsed = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Viral: {len(viral_data['ids'])} sequences")
    logger.info(f"Non-viral: {len(nonviral_data['ids'])} sequences")
    logger.info(f"Total: {len(viral_data['ids']) + len(nonviral_data['ids'])}")
    logger.info(f"Feature dimension: 3328 (768 DNA + 2560 protein)")
    logger.info(f"\nOutput directory: {output_dir}/")

    logger.info("\n" + "=" * 60)
    logger.info("RUNTIME")
    logger.info("=" * 60)
    logger.info(f"Viral: {viral_elapsed/60:.1f} minutes")
    logger.info(f"Non-viral: {nonviral_elapsed/60:.1f} minutes")
    logger.info(f"Total: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")


if __name__ == '__main__':
    main()
