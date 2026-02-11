#!/usr/bin/env python3
"""
Generate 150bp sliding windows from test sequences for short-read benchmarking.
Uses 50bp step size for comprehensive coverage.
"""
import sys
from pathlib import Path
from Bio import SeqIO
from collections import defaultdict
import json

def generate_sliding_windows(seq_record, window_size=150, step_size=50):
    """Generate sliding windows from a sequence."""
    seq = str(seq_record.seq)
    seq_id = seq_record.id
    windows = []

    # Skip sequences shorter than window size
    if len(seq) < window_size:
        return windows

    # Generate windows
    for i in range(0, len(seq) - window_size + 1, step_size):
        window_seq = seq[i:i + window_size]
        window_id = f"{seq_id}_window_{i}_{i+window_size}"
        windows.append((window_id, window_seq))

    return windows

def main():
    window_size = 150
    step_size = 50

    viral_input = "test_sequences_viral.fasta"
    nonviral_input = "test_sequences_nonviral.fasta"

    viral_output = "test_150bp_viral.fasta"
    nonviral_output = "test_150bp_nonviral.fasta"

    stats = {
        'window_size': window_size,
        'step_size': step_size,
        'viral': {'sequences': 0, 'windows': 0, 'skipped_short': 0},
        'nonviral': {'sequences': 0, 'windows': 0, 'skipped_short': 0}
    }

    # Process viral sequences
    print(f"Processing viral sequences from {viral_input}...")
    with open(viral_output, 'w') as out_f:
        for record in SeqIO.parse(viral_input, "fasta"):
            stats['viral']['sequences'] += 1

            if len(record.seq) < window_size:
                stats['viral']['skipped_short'] += 1
                continue

            windows = generate_sliding_windows(record, window_size, step_size)
            stats['viral']['windows'] += len(windows)

            for window_id, window_seq in windows:
                out_f.write(f">{window_id}\n{window_seq}\n")

            if stats['viral']['sequences'] % 1000 == 0:
                print(f"  Processed {stats['viral']['sequences']} viral sequences, "
                      f"{stats['viral']['windows']} windows generated")

    print(f"Viral: {stats['viral']['sequences']} sequences -> {stats['viral']['windows']} windows")
    print(f"       Skipped {stats['viral']['skipped_short']} sequences shorter than {window_size}bp\n")

    # Process non-viral sequences
    print(f"Processing non-viral sequences from {nonviral_input}...")
    with open(nonviral_output, 'w') as out_f:
        for record in SeqIO.parse(nonviral_input, "fasta"):
            stats['nonviral']['sequences'] += 1

            if len(record.seq) < window_size:
                stats['nonviral']['skipped_short'] += 1
                continue

            windows = generate_sliding_windows(record, window_size, step_size)
            stats['nonviral']['windows'] += len(windows)

            for window_id, window_seq in windows:
                out_f.write(f">{window_id}\n{window_seq}\n")

            if stats['nonviral']['sequences'] % 1000 == 0:
                print(f"  Processed {stats['nonviral']['sequences']} non-viral sequences, "
                      f"{stats['nonviral']['windows']} windows generated")

    print(f"Non-viral: {stats['nonviral']['sequences']} sequences -> {stats['nonviral']['windows']} windows")
    print(f"           Skipped {stats['nonviral']['skipped_short']} sequences shorter than {window_size}bp\n")

    # Save statistics
    stats['total_windows'] = stats['viral']['windows'] + stats['nonviral']['windows']
    stats['total_sequences'] = stats['viral']['sequences'] + stats['nonviral']['sequences']

    with open('test_150bp_metadata.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSummary:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Total 150bp windows: {stats['total_windows']}")
    print(f"  Average windows per sequence: {stats['total_windows'] / stats['total_sequences']:.1f}")
    print(f"  Viral windows: {stats['viral']['windows']}")
    print(f"  Non-viral windows: {stats['nonviral']['windows']}")
    print(f"\nOutputs:")
    print(f"  {viral_output}")
    print(f"  {nonviral_output}")
    print(f"  test_150bp_metadata.json")

if __name__ == '__main__':
    main()
