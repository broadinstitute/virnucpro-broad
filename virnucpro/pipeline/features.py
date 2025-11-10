"""Feature extraction using DNABERT-S and ESM-2 transformers"""

import torch
from pathlib import Path
from typing import Dict, List, Optional
from Bio import SeqIO
import logging

logger = logging.getLogger('virnucpro.features')


def extract_dnabert_features(
    nucleotide_file: Path,
    output_file: Path,
    device: torch.device,
    batch_size: int = 256
) -> Path:
    """
    Extract DNA sequence embeddings using DNABERT-S.

    Based on units.py:160-201

    Args:
        nucleotide_file: Input FASTA file with nucleotide sequences
        output_file: Output .pt file for features
        device: PyTorch device for computation
        batch_size: Batch size for processing

    Returns:
        Path to saved feature file
    """
    from transformers import AutoTokenizer, AutoModel

    logger.info(f"Extracting DNABERT-S features from {nucleotide_file}")

    # Load model and tokenizer
    model_name = "zhihan1996/DNABERT-S"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    # Load sequences
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(nucleotide_file, 'fasta'):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)

    # Extract features in batches
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Forward pass
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state

            # Mean pooling across sequence dimension
            embeddings = hidden_states.mean(dim=1)  # (batch, 768)
            all_embeddings.append(embeddings.cpu())

            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size}")

    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Save to file
    feature_dict = {
        'nucleotide': seq_ids,
        'data': all_embeddings
    }
    torch.save(feature_dict, output_file)

    logger.info(f"Saved DNABERT-S features to {output_file} (shape: {all_embeddings.shape})")
    return output_file


def extract_esm_features(
    protein_file: Path,
    output_file: Path,
    device: torch.device,
    truncation_length: int = 1024,
    toks_per_batch: int = 2048
) -> Path:
    """
    Extract protein sequence embeddings using ESM-2.

    Based on units.py:204-265

    Args:
        protein_file: Input FASTA file with protein sequences
        output_file: Output .pt file for features
        device: PyTorch device for computation
        truncation_length: Maximum sequence length
        toks_per_batch: Tokens per batch for batching

    Returns:
        Path to saved feature file
    """
    import esm

    logger.info(f"Extracting ESM-2 features from {protein_file}")

    # Load ESM-2 3B model
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

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

    # Extract features batch by batch
    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            # Convert batch
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)

            # Forward pass
            results = model(batch_tokens, repr_layers=[36])
            representations = results["representations"][36]  # Layer 36

            # Mean pool over sequence (excluding special tokens)
            for i, (seq_id, seq_str) in enumerate(batch):
                seq_len = len(seq_str)
                # Mean pool positions 1 to seq_len+1 (skip BOS token)
                embedding = representations[i, 1:seq_len + 1].mean(dim=0)  # (2560,)
                all_embeddings.append(embedding.cpu())
                all_ids.append(seq_id)

            logger.debug(f"Processed batch {batch_idx + 1}/{len(batches)}")

    # Stack all embeddings
    all_embeddings = torch.stack(all_embeddings)

    # Save to file
    feature_dict = {
        'proteins': all_ids,
        'data': all_embeddings
    }
    torch.save(feature_dict, output_file)

    logger.info(f"Saved ESM-2 features to {output_file} (shape: {all_embeddings.shape})")
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
    logger.info(f"Merging features: {nucleotide_feature_file.name} + {protein_feature_file.name}")

    # Load features
    nuc_data = torch.load(nucleotide_feature_file)
    pro_data = torch.load(protein_feature_file)

    # Extract IDs and features
    nuc_ids = nuc_data['nucleotide']
    nuc_features = nuc_data['data']  # (N, 768)

    pro_ids = pro_data['proteins']
    pro_features = pro_data['data']  # (N, 2560)

    # Verify IDs match
    if nuc_ids != pro_ids:
        raise ValueError(
            f"Sequence ID mismatch between nucleotide and protein features.\n"
            f"Nucleotide has {len(nuc_ids)} sequences, protein has {len(pro_ids)}"
        )

    # Concatenate features
    merged_features = torch.cat([nuc_features, pro_features], dim=1)  # (N, 3328)

    # Save merged features
    merged_dict = {
        'ids': nuc_ids,
        'data': merged_features,
        'labels': None  # No labels for prediction mode
    }
    torch.save(merged_dict, output_file)

    logger.info(f"Saved merged features to {output_file} (shape: {merged_features.shape})")
    return output_file
