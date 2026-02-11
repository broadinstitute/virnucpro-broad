#!/usr/bin/env python3
"""
Sample 50K random 150bp windows for fast benchmarking.
Maintains viral/non-viral ratio from original test set.
"""
import random
from Bio import SeqIO
import json

random.seed(42)

def sample_fasta(input_file, output_file, n_samples):
    """Sample n random sequences from a FASTA file."""
    sequences = list(SeqIO.parse(input_file, "fasta"))

    if len(sequences) < n_samples:
        print(f"Warning: {input_file} has only {len(sequences)} sequences, less than requested {n_samples}")
        sampled = sequences
    else:
        sampled = random.sample(sequences, n_samples)

    with open(output_file, 'w') as f:
        for record in sampled:
            f.write(f">{record.id}\n{record.seq}\n")

    print(f"{input_file}: {len(sequences)} -> {len(sampled)} sequences")
    return len(sampled)

if __name__ == '__main__':
    # Load test set metadata to maintain viral/non-viral ratio
    with open('data/test_set/test_metadata.json') as f:
        metadata = json.load(f)

    total_test_seqs = metadata['distribution']['test_sequences']
    # Viral ratio based on sequences, not files
    # Test set has 110K viral, 100K non-viral = 52.4% viral
    viral_ratio = 110000 / 210000  # 0.524

    total_samples = 50000
    viral_samples = int(total_samples * viral_ratio)
    nonviral_samples = total_samples - viral_samples

    print(f"Sampling {total_samples} total windows:")
    print(f"  Viral: {viral_samples} ({viral_ratio*100:.1f}%)")
    print(f"  Non-viral: {nonviral_samples} ({(1-viral_ratio)*100:.1f}%)")
    print()

    viral_count = sample_fasta(
        "test_150bp_viral.fasta",
        "test_150bp_viral_50k.fasta",
        viral_samples
    )

    nonviral_count = sample_fasta(
        "test_150bp_nonviral.fasta",
        "test_150bp_nonviral_50k.fasta",
        nonviral_samples
    )

    # Save metadata
    stats = {
        'total_samples': viral_count + nonviral_count,
        'viral_samples': viral_count,
        'nonviral_samples': nonviral_count,
        'viral_ratio': viral_count / (viral_count + nonviral_count),
        'sampling_strategy': 'random',
        'seed': 42,
        'source': '150bp sliding windows'
    }

    with open('test_150bp_50k_metadata.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal sampled: {viral_count + nonviral_count} sequences")
    print("\nOutput files:")
    print("  test_150bp_viral_50k.fasta")
    print("  test_150bp_nonviral_50k.fasta")
    print("  test_150bp_50k_metadata.json")
