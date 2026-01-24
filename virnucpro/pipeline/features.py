"""Feature extraction using DNABERT-S and ESM-2 transformers"""

import torch
from torch.cuda.amp import autocast
from pathlib import Path
from typing import Dict, List, Optional
from Bio import SeqIO
import logging

from virnucpro.core.checkpoint import atomic_save

logger = logging.getLogger('virnucpro.features')


def extract_dnabert_features(
    nucleotide_file: Path,
    output_file: Path,
    device: torch.device,
    batch_size: int = 256,
    model=None,
    tokenizer=None
) -> Path:
    """
    Extract DNA sequence embeddings using DNABERT-S.

    Based on units.py:160-201

    Args:
        nucleotide_file: Input FASTA file with nucleotide sequences
        output_file: Output .pt file for features
        device: PyTorch device for computation
        batch_size: Batch size for processing
        model: Optional pre-loaded model (for persistent workers)
        tokenizer: Optional pre-loaded tokenizer (for persistent workers)

    Returns:
        Path to saved feature file
    """
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Extracting DNABERT-S features from {nucleotide_file}")

    # Only load if not provided (backward compatibility)
    if model is None or tokenizer is None:
        model_name = "zhihan1996/DNABERT-S"
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model is None:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    else:
        # Ensure provided model is on correct device
        model = model.to(device)

    model.eval()

    # Load all sequences
    nucleotide = []
    data = []

    records = list(SeqIO.parse(nucleotide_file, 'fasta'))

    with torch.no_grad():
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch_seqs = [str(record.seq) for record in batch_records]
            batch_labels = [record.id for record in batch_records]

            # Tokenize batch with padding
            inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass - model returns tuple, take first element
            if attention_mask is not None:
                hidden_states = model(input_ids, attention_mask=attention_mask)[0]
            else:
                hidden_states = model(input_ids)[0]

            # Mean pool each sequence in batch, excluding padding tokens
            if attention_mask is not None:
                embedding_means = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                embedding_means = torch.mean(hidden_states, dim=1)

            for label, embedding_mean in zip(batch_labels, embedding_means):
                result = {"label": label, "mean_representation": embedding_mean.cpu().tolist()}
                nucleotide.append(label)
                data.append(result)

    # Save to file using atomic write
    checkpoint_dict = {'nucleotide': nucleotide, 'data': data}
    atomic_save(checkpoint_dict, output_file, validate_after_save=False)

    logger.info(f"Saved DNABERT-S features to {output_file} ({len(data)} sequences)")
    return output_file


def extract_esm_features(
    protein_file: Path,
    output_file: Path,
    device: torch.device,
    truncation_length: int = 1024,
    toks_per_batch: int = 2048,
    stream_processor=None,
    model=None,
    batch_converter=None
) -> Path:
    """
    Extract protein sequence embeddings using ESM-2.

    Based on units.py:204-265

    Supports optional CUDA stream-based processing for I/O-compute overlap.
    When stream_processor is provided, uses pipelined processing to hide
    data transfer latency.

    Args:
        protein_file: Input FASTA file with protein sequences
        output_file: Output .pt file for features
        device: PyTorch device for computation
        truncation_length: Maximum sequence length
        toks_per_batch: Tokens per batch for batching
        stream_processor: Optional StreamProcessor for I/O-compute overlap
        model: Optional pre-loaded model (for persistent workers)
        batch_converter: Optional pre-loaded batch_converter (for persistent workers)

    Returns:
        Path to saved feature file
    """
    from virnucpro.models.esm2_flash import load_esm2_model

    logger.info(f"Extracting ESM-2 features from {protein_file}")

    # Only load if not provided (backward compatibility)
    if model is None or batch_converter is None:
        # Load ESM-2 3B model with FlashAttention-2 support
        # The wrapper automatically detects and enables FlashAttention-2 on Ampere+ GPUs
        # and configures BF16 mixed precision
        model, batch_converter = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device=str(device),
            logger_instance=logger
        )
    else:
        # Ensure provided model is on correct device
        model = model.to(device)

    # Get BF16 status from wrapper (already configured)
    use_bf16 = getattr(model, 'use_bf16', False)

    # Increase batch size with BF16 if using default
    if use_bf16 and toks_per_batch == 2048:
        toks_per_batch = 3072  # Increase batch size with BF16
        logger.info(f"Increased toks_per_batch to {toks_per_batch} with BF16")

    # Load sequences
    sequences = []
    for record in SeqIO.parse(protein_file, 'fasta'):
        seq_str = str(record.seq)
        # Truncate if needed
        if len(seq_str) > truncation_length:
            seq_str = seq_str[:truncation_length]
        sequences.append((record.id, seq_str))

    # Batch sequences by token count
    batches = []
    current_batch = []
    current_tokens = 0

    for seq_id, seq_str in sequences:
        seq_len = len(seq_str)
        if current_tokens + seq_len > toks_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append((seq_id, seq_str))
        current_tokens += seq_len

    if current_batch:
        batches.append(current_batch)

    # Extract features batch by batch (matching original format)
    proteins = []
    data = []

    with torch.no_grad():
        with autocast(dtype=torch.bfloat16, enabled=use_bf16):
            for batch_idx, batch in enumerate(batches):
                if stream_processor is not None:
                    # Stream-based processing with I/O-compute overlap
                    def transfer_fn(b):
                        labels, strs, tokens = batch_converter(b)
                        return tokens.to(device, non_blocking=True)

                    def compute_fn(tokens):
                        results = model(tokens, repr_layers=[36])
                        return results["representations"][36]

                    def retrieve_fn(reps):
                        return reps.cpu()

                    # Convert batch
                    batch_labels, batch_strs, _ = batch_converter(batch)

                    # Process with streams
                    representations = stream_processor.process_batch_async(
                        batch, transfer_fn, compute_fn, retrieve_fn
                    )
                else:
                    # Standard synchronous processing
                    batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                    batch_tokens = batch_tokens.to(device)

                    # Forward pass
                    results = model(batch_tokens, repr_layers=[36])
                    representations = results["representations"][36]  # Layer 36

                # Mean pool over sequence (excluding special tokens)
                for i, (seq_id, seq_str) in enumerate(batch):
                    truncate_len = min(truncation_length, len(seq_str))
                    # Mean pool positions 1 to truncate_len+1 (skip BOS token)
                    # Convert to FP32 for storage compatibility
                    if stream_processor is not None:
                        # Already on CPU from retrieve_fn
                        embedding = representations[i, 1:truncate_len + 1].mean(dim=0).clone().float()
                    else:
                        embedding = representations[i, 1:truncate_len + 1].mean(dim=0).clone().float().to('cpu')
                    proteins.append(seq_id)
                    data.append(embedding)

                logger.debug(f"Processed batch {batch_idx + 1}/{len(batches)}")

    # Save to file using atomic write (list of tensors, not stacked)
    checkpoint_dict = {'proteins': proteins, 'data': data}
    atomic_save(checkpoint_dict, output_file, validate_after_save=False)

    logger.info(f"Saved ESM-2 features to {output_file} ({len(data)} sequences)")
    return output_file


def merge_features(
    nucleotide_feature_file: Path,
    protein_feature_file: Path,
    output_file: Path
) -> Path:
    """
    Merge DNABERT-S and ESM-2 features by concatenation.

    Based on units.py:290-324

    Args:
        nucleotide_feature_file: .pt file with DNABERT-S features (768-dim)
        protein_feature_file: .pt file with ESM-2 features (2560-dim)
        output_file: Output .pt file for merged features (3328-dim)

    Returns:
        Path to saved merged feature file

    Raises:
        ValueError: If sequence IDs don't match between files
    """
    # Skip if already merged (checkpoint support)
    if output_file.exists() and output_file.stat().st_size > 0:
        logger.info(f"Skipping merge - output exists: {output_file}")
        return output_file

    logger.info(f"Merging features: {nucleotide_feature_file.name} + {protein_feature_file.name}")

    # Load features
    nuc_data = torch.load(nucleotide_feature_file)
    pro_data = torch.load(protein_feature_file)

    # Create dictionaries for lookup (matching original implementation)
    nucleotide_data_dict = {}
    protein_data_dict = {}

    # Convert DNABERT-S data from list of dicts to tensors
    for nucleotide, data in zip(nuc_data['nucleotide'], nuc_data['data']):
        nucleotide_data_dict[nucleotide] = torch.tensor(data['mean_representation'])

    # ESM data is already tensors
    for protein, data in zip(pro_data['proteins'], pro_data['data']):
        protein_data_dict[protein] = data

    # Merge features
    merged_data = []
    for item in nuc_data['nucleotide']:
        if item in protein_data_dict and item in nucleotide_data_dict:
            protein_feature = protein_data_dict[item]
            nucleotide_feature = nucleotide_data_dict[item]

            merged_feature = torch.cat((nucleotide_feature, protein_feature), dim=-1)
            merged_data.append(merged_feature)
        else:
            logger.warning(f"Warning: {item} not found in both datasets")

    # Handle empty merged data
    if not merged_data:
        logger.warning(f"No matching sequences between {nucleotide_feature_file.name} and {protein_feature_file.name}")
        empty_dict = {'ids': nuc_data['nucleotide'], 'data': torch.empty((0, 3328))}
        atomic_save(empty_dict, output_file, validate_after_save=False)
        return output_file

    # Stack into tensor
    merged_data = torch.stack(merged_data)

    # Save merged features using atomic write (no labels for prediction mode)
    merged_dict = {
        'ids': nuc_data['nucleotide'],
        'data': merged_data
    }
    atomic_save(merged_dict, output_file, validate_after_save=False)

    logger.info(f"Saved merged features to {output_file} (shape: {merged_data.shape})")
    return output_file
