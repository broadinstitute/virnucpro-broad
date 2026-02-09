#!/usr/bin/env python3
"""
Extract nucleotide sequences from test set .pt files to FASTA format.

The merged .pt files contain:
  - 'ids': List of sequence identifiers (e.g., 'NC_001234.1_chunk_1R2')
  - 'data': Tensor of merged embeddings (not the sequences themselves)
  - 'labels': [0] or [1] for non-viral/viral

This script:
1. Loads test .pt files to get sequence IDs
2. Maps IDs back to original identified_nucleotide.fa files
3. Extracts corresponding nucleotide sequences
4. Writes to FASTA files for baseline comparison

Output files:
- test_sequences_viral.fasta: All viral test sequences
- test_sequences_nonviral.fasta: All non-viral test sequences
- test_sequences_all.fasta: Combined (all test sequences)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
from tqdm import tqdm
from Bio import SeqIO
from collections import defaultdict


def load_fasta_sequences(fasta_path):
    """
    Load all sequences from a FASTA file into a dictionary.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        dict: {sequence_id: sequence_string}
    """
    sequences = {}
    try:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            sequences[record.id] = str(record.seq)
    except Exception as e:
        print(f"Warning: Could not load {fasta_path}: {e}")
    return sequences


def extract_ids_from_pt(pt_file_path):
    """
    Extract sequence IDs from a merged .pt file.

    Args:
        pt_file_path: Path to the .pt file

    Returns:
        tuple: (ids_list, label)
        where ids_list is list of sequence identifiers and label is 0 or 1
    """
    try:
        data = torch.load(pt_file_path, map_location='cpu')

        if isinstance(data, dict):
            ids = data.get('ids', [])
            labels = data.get('labels', [0])
            label = labels[0] if labels else 0
            return (ids, label)
        else:
            print(f"Warning: {pt_file_path} is not a dict, it's {type(data)}")
            return ([], 0)

    except Exception as e:
        print(f"Error loading {pt_file_path}: {e}")
        return ([], 0)


def determine_label_from_path(file_path):
    """
    Determine if a file is viral or non-viral from its path.

    Args:
        file_path: Path to the file

    Returns:
        int: 1 if viral, 0 if non-viral
    """
    path_str = str(file_path).lower()
    if 'viral.1.1' in path_str:
        return 1
    return 0


def get_source_fasta_from_pt_path(pt_path):
    """
    Determine the source FASTA file from a .pt file path.

    Args:
        pt_path: Path to .pt file (e.g., data/data_merge/viral.1.1_merged/output_2_merged.pt)

    Returns:
        str: Path to source identified_nucleotide.fa file
    """
    pt_path_str = str(pt_path)

    # Extract organism name from path (e.g., viral.1.1, bacteria.663.1)
    if '_merged/' in pt_path_str:
        org_part = pt_path_str.split('_merged/')[0].split('/')[-1]
        # Remove _merged suffix if present
        org_name = org_part.replace('_merged', '')

        # Construct path to identified_nucleotide.fa
        fasta_path = f"data/{org_name}.identified_nucleotide.fa"
        return fasta_path

    return None


def write_fasta(sequences, output_path, description=""):
    """
    Write sequences to FASTA format.

    Args:
        sequences: List of tuples (seq_id, seq, label, file_origin)
        output_path: Path to output FASTA file
        description: Optional description for file header
    """
    with open(output_path, 'w') as f:
        if description:
            f.write(f"# {description}\n")

        for idx, (seq_id, seq, label, file_origin) in enumerate(sequences):
            label_str = "viral" if label == 1 else "non-viral"
            file_basename = Path(file_origin).name
            header = f">{seq_id}|{label_str}|{file_basename}"
            f.write(f"{header}\n{seq}\n")

    print(f"Wrote {len(sequences)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract test sequences from .pt files to FASTA format"
    )
    parser.add_argument(
        '--test-metadata',
        default='data/test_set/test_metadata.json',
        help='Path to test metadata JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for FASTA files'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory containing identified_nucleotide.fa files'
    )

    args = parser.parse_args()

    # Load test metadata
    print(f"Loading test metadata from: {args.test_metadata}")
    with open(args.test_metadata, 'r') as f:
        metadata = json.load(f)

    test_files = metadata.get('test_files', [])
    print(f"Found {len(test_files)} test files")

    # Group test files by organism to minimize FASTA loading
    files_by_organism = defaultdict(list)
    for test_file in test_files:
        fasta_path = get_source_fasta_from_pt_path(test_file)
        if fasta_path:
            files_by_organism[fasta_path].append(test_file)

    print(f"\nTest files span {len(files_by_organism)} different organisms")

    # Extract sequences
    all_sequences = []
    viral_sequences = []
    nonviral_sequences = []

    print("\nExtracting sequence IDs and loading source FASTAs...")
    for fasta_path, pt_files in tqdm(files_by_organism.items()):
        # Load FASTA sequences once per organism
        print(f"  Loading {fasta_path}...")
        fasta_seqs = load_fasta_sequences(fasta_path)
        print(f"    Loaded {len(fasta_seqs)} sequences")

        # Process each .pt file for this organism
        for pt_file in pt_files:
            ids, label = extract_ids_from_pt(pt_file)

            for seq_id in ids:
                if seq_id in fasta_seqs:
                    seq = fasta_seqs[seq_id]
                    seq_tuple = (seq_id, seq, label, pt_file)
                    all_sequences.append(seq_tuple)

                    if label == 1:
                        viral_sequences.append(seq_tuple)
                    else:
                        nonviral_sequences.append(seq_tuple)
                else:
                    print(f"    Warning: {seq_id} not found in {fasta_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write FASTA files
    print("\nWriting FASTA files...")

    viral_files = [f for f in test_files if 'viral.1.1' in f]
    nonviral_files = [f for f in test_files if 'viral.1.1' not in f]

    write_fasta(
        viral_sequences,
        output_dir / 'test_sequences_viral.fasta',
        f"Viral test sequences extracted from {len(viral_files)} test files"
    )

    write_fasta(
        nonviral_sequences,
        output_dir / 'test_sequences_nonviral.fasta',
        f"Non-viral test sequences extracted from {len(nonviral_files)} test files"
    )

    write_fasta(
        all_sequences,
        output_dir / 'test_sequences_all.fasta',
        f"All test sequences extracted from {len(test_files)} test files"
    )

    # Print summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total test files processed: {len(test_files)}")
    print(f"Total sequences extracted: {len(all_sequences)}")
    print(f"  Viral sequences: {len(viral_sequences)}")
    print(f"  Non-viral sequences: {len(nonviral_sequences)}")
    print("\nOutput files:")
    print(f"  {output_dir / 'test_sequences_viral.fasta'}")
    print(f"  {output_dir / 'test_sequences_nonviral.fasta'}")
    print(f"  {output_dir / 'test_sequences_all.fasta'}")
    print("="*80)


if __name__ == '__main__':
    main()
