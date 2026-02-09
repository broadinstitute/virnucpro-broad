#!/usr/bin/env python3
"""
Create stratified test set from merged training data.

This script performs a reproducible 10% stratified split of the training data,
separating test files from training files to ensure unbiased evaluation.
The test set is used for both FastESM2 and baseline model evaluation.
"""

import os
import argparse
import json
import random
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_merged_files(data_dir):
    """
    Discover all merged .pt files from data_merge directory.

    Args:
        data_dir: Path to data_merge directory

    Returns:
        tuple of (file_paths, labels) where labels are 1 for viral, 0 for non-viral
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_paths = []
    labels = []

    # Discover subdirectories
    subdirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    print(f"Discovered {len(subdirs)} subdirectories in {data_dir}")

    for subdir in subdirs:
        # Classify as viral (label=1) or non-viral (label=0)
        is_viral = 'viral' in subdir.name.lower()
        label = 1 if is_viral else 0

        # Find all .pt files in this directory
        pt_files = sorted(subdir.glob('*_merged.pt'))

        if not pt_files:
            print(f"Warning: No *_merged.pt files found in {subdir.name}")
            continue

        # Verify label by loading first file
        first_file = pt_files[0]
        try:
            data_dict = torch.load(first_file)
            file_label = data_dict['labels'][0]

            if file_label != label:
                print(f"Warning: Directory {subdir.name} classification mismatch!")
                print(f"  Expected label={label}, file contains label={file_label}")
                # Trust the file label
                label = file_label

        except Exception as e:
            print(f"Warning: Could not verify label from {first_file}: {e}")

        # Add all files from this directory
        for pt_file in pt_files:
            file_paths.append(str(pt_file))
            labels.append(label)

        print(f"  {subdir.name}: {len(pt_files)} files, label={label}")

    return file_paths, labels


def create_test_set(data_dir, output_dir, split_ratio, seed):
    """
    Create stratified test set split.

    Args:
        data_dir: Path to data_merge directory
        output_dir: Path to output test_set directory
        split_ratio: Fraction of data for test set (e.g., 0.1)
        seed: Random seed for reproducibility

    Returns:
        dict with metadata about the split
    """
    # Set all seeds
    set_all_seeds(seed)

    # Discover all files
    print("\n=== Discovering files ===")
    file_paths, labels = discover_merged_files(data_dir)

    if not file_paths:
        raise ValueError(f"No merged .pt files found in {data_dir}")

    print(f"\nTotal files discovered: {len(file_paths)}")
    print(f"  Viral (label=1): {sum(labels)}")
    print(f"  Non-viral (label=0): {len(labels) - sum(labels)}")

    # Perform stratified split
    print(f"\n=== Performing stratified split (test_size={split_ratio}, seed={seed}) ===")
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=split_ratio,
        stratify=labels,
        random_state=seed
    )

    print(f"\nSplit results:")
    print(f"  Train files: {len(train_files)}")
    print(f"    Viral: {sum(train_labels)}")
    print(f"    Non-viral: {len(train_labels) - sum(train_labels)}")
    print(f"  Test files: {len(test_files)}")
    print(f"    Viral: {sum(test_labels)}")
    print(f"    Non-viral: {len(test_labels) - sum(test_labels)}")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viral_dir = output_path / 'viral'
    non_viral_dir = output_path / 'non_viral'
    viral_dir.mkdir(exist_ok=True)
    non_viral_dir.mkdir(exist_ok=True)

    # Copy test files (create symlinks to save disk space)
    print(f"\n=== Creating symlinks in {output_dir} ===")
    for test_file, label in zip(test_files, test_labels):
        source = Path(test_file).resolve()

        if label == 1:
            target_dir = viral_dir
        else:
            target_dir = non_viral_dir

        target = target_dir / source.name

        # Remove existing symlink if present
        if target.exists() or target.is_symlink():
            target.unlink()

        # Create symlink
        target.symlink_to(source)

    print(f"  Created {len(test_files)} symlinks")
    print(f"    Viral: {viral_dir} ({sum(test_labels)} files)")
    print(f"    Non-viral: {non_viral_dir} ({len(test_labels) - sum(test_labels)} files)")

    # Calculate distribution statistics
    train_viral_ratio = sum(train_labels) / len(train_labels) if train_labels else 0
    test_viral_ratio = sum(test_labels) / len(test_labels) if test_labels else 0

    print(f"\n=== Distribution statistics ===")
    print(f"  Train viral ratio: {train_viral_ratio:.4f}")
    print(f"  Test viral ratio: {test_viral_ratio:.4f}")
    print(f"  Ratio difference: {abs(train_viral_ratio - test_viral_ratio):.4f}")

    # Count sequences in test set
    print(f"\n=== Counting test sequences ===")
    total_test_sequences = 0
    for test_file in test_files:
        try:
            data_dict = torch.load(test_file)
            num_sequences = data_dict['data'].shape[0]
            total_test_sequences += num_sequences
        except Exception as e:
            print(f"Warning: Could not load {test_file}: {e}")

    print(f"  Total sequences in test set: {total_test_sequences:,}")

    # Create metadata
    metadata = {
        'seed': seed,
        'split_ratio': split_ratio,
        'train_files': train_files,
        'test_files': test_files,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'distribution': {
            'total_files': len(file_paths),
            'train_files': len(train_files),
            'test_files': len(test_files),
            'train_viral': sum(train_labels),
            'train_non_viral': len(train_labels) - sum(train_labels),
            'test_viral': sum(test_labels),
            'test_non_viral': len(test_labels) - sum(test_labels),
            'train_viral_ratio': train_viral_ratio,
            'test_viral_ratio': test_viral_ratio,
            'test_sequences': total_test_sequences
        },
        'timestamp': datetime.utcnow().isoformat()
    }

    # Save metadata
    metadata_file = output_path / 'test_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Metadata saved to {metadata_file} ===")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified test set from merged training data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/data_merge/',
        help='Path to data_merge directory containing merged .pt files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/test_set/',
        help='Path to output test_set directory'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.1,
        help='Fraction of data for test set (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Run test set creation
    metadata = create_test_set(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_ratio=args.split,
        seed=args.seed
    )

    print("\n=== Test set creation complete ===")
    print(f"Train files: {metadata['distribution']['train_files']}")
    print(f"Test files: {metadata['distribution']['test_files']}")
    print(f"Test sequences: {metadata['distribution']['test_sequences']:,}")


if __name__ == '__main__':
    main()
