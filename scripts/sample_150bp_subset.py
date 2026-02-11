#!/usr/bin/env python3
"""
Sample 10% of 150bp windows for runtime testing.
"""
import random
from Bio import SeqIO

random.seed(42)

def sample_fasta(input_file, output_file, fraction=0.1):
    """Sample a fraction of sequences from a FASTA file."""
    sequences = list(SeqIO.parse(input_file, "fasta"))
    sample_size = int(len(sequences) * fraction)

    sampled = random.sample(sequences, sample_size)

    with open(output_file, 'w') as f:
        for record in sampled:
            f.write(f">{record.id}\n{record.seq}\n")

    print(f"{input_file}: {len(sequences)} -> {len(sampled)} sequences ({fraction*100:.0f}%)")
    return len(sampled)

if __name__ == '__main__':
    viral_count = sample_fasta("test_150bp_viral.fasta", "test_150bp_viral_10pct.fasta", 0.1)
    nonviral_count = sample_fasta("test_150bp_nonviral.fasta", "test_150bp_nonviral_10pct.fasta", 0.1)

    print(f"\nTotal sampled: {viral_count + nonviral_count} sequences")
    print("Output files:")
    print("  test_150bp_viral_10pct.fasta")
    print("  test_150bp_nonviral_10pct.fasta")
