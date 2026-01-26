"""Synthetic test data generator for controlled benchmarks.

Provides:
- generate_synthetic_fasta(): Generate viral-like DNA sequences
- Preset configurations for standard test sizes (TINY, SMALL, MEDIUM, LARGE)
- generate_benchmark_dataset(): Create multi-file benchmark datasets
- real_sample_loader(): Load real viral samples if available

Generated data is reproducible via seed parameter and gitignored but regeneratable.
"""

import random
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import json
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger('virnucpro.benchmarks.data_generator')


# ==================== Preset Configurations ====================

@dataclass
class DatasetConfig:
    """Configuration for benchmark dataset."""
    name: str
    num_sequences: int
    min_length: int
    max_length: int
    description: str


# Standard preset sizes for benchmarking
PRESETS = {
    'TINY': DatasetConfig(
        name='tiny',
        num_sequences=10,
        min_length=200,
        max_length=500,
        description='Quick smoke tests (10 sequences)'
    ),
    'SMALL': DatasetConfig(
        name='small',
        num_sequences=100,
        min_length=200,
        max_length=800,
        description='CI tests (100 sequences)'
    ),
    'MEDIUM': DatasetConfig(
        name='medium',
        num_sequences=1000,
        min_length=200,
        max_length=1500,
        description='Standard benchmarks (1000 sequences)'
    ),
    'LARGE': DatasetConfig(
        name='large',
        num_sequences=10000,
        min_length=100,
        max_length=2000,
        description='Stress tests (10000 sequences)'
    ),
}


# ==================== FASTA Generation ====================

def generate_synthetic_fasta(num_sequences: int,
                             min_length: int,
                             max_length: int,
                             output_path: Path,
                             seed: Optional[int] = 42,
                             gc_content: float = 0.5) -> Path:
    """
    Generate synthetic viral-like DNA sequences in FASTA format.

    Creates sequences with:
    - Realistic length distribution (uniform between min and max)
    - Configurable GC content (default: 50%)
    - Reproducible via seed parameter
    - BioPython SeqRecord format for compatibility

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length (bp)
        max_length: Maximum sequence length (bp)
        output_path: Path to output FASTA file
        seed: Random seed for reproducibility (None = random)
        gc_content: Target GC content ratio (0.0-1.0, default: 0.5)

    Returns:
        Path to generated FASTA file

    Example:
        >>> generate_synthetic_fasta(
        ...     num_sequences=100,
        ...     min_length=300,
        ...     max_length=800,
        ...     output_path=Path('tests/data/synthetic/test_100.fa')
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Create base pools weighted by GC content
    gc_bases = ['G', 'C']
    at_bases = ['A', 'T']

    records = []

    logger.info(f"Generating {num_sequences} sequences ({min_length}-{max_length} bp)")

    for i in range(num_sequences):
        # Random length in range
        seq_length = random.randint(min_length, max_length)

        # Generate sequence with target GC content
        sequence = []
        for _ in range(seq_length):
            if random.random() < gc_content:
                sequence.append(random.choice(gc_bases))
            else:
                sequence.append(random.choice(at_bases))

        seq_str = ''.join(sequence)

        # Create SeqRecord
        record = SeqRecord(
            Seq(seq_str),
            id=f"synthetic_{i:06d}",
            description=f"Synthetic viral sequence {i} (length={seq_length}, gc={gc_content:.2f})"
        )
        records.append(record)

    # Write FASTA file
    SeqIO.write(records, output_path, "fasta")

    logger.info(f"Generated {num_sequences} sequences to {output_path}")
    logger.info(f"Total size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


# ==================== Benchmark Dataset Generation ====================

def generate_benchmark_dataset(preset: str,
                               output_dir: Path,
                               num_files: int = 1,
                               seed: Optional[int] = 42) -> Path:
    """
    Generate complete benchmark dataset with metadata.

    Creates directory structure:
        output_dir/
        ├── {preset}/
        │   ├── file_001.fa
        │   ├── file_002.fa
        │   ├── ...
        │   └── metadata.json

    Args:
        preset: Preset name ('TINY', 'SMALL', 'MEDIUM', 'LARGE')
        output_dir: Base output directory (e.g., 'tests/data/synthetic')
        num_files: Number of FASTA files to generate (for multi-file benchmarks)
        seed: Random seed for reproducibility

    Returns:
        Path to dataset directory

    Example:
        >>> generate_benchmark_dataset(
        ...     preset='MEDIUM',
        ...     output_dir=Path('tests/data/synthetic'),
        ...     num_files=4
        ... )
        Path('tests/data/synthetic/medium')
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    config = PRESETS[preset]
    dataset_dir = Path(output_dir) / config.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {preset} dataset: {config.description}")
    logger.info(f"Directory: {dataset_dir}")

    # Calculate sequences per file
    sequences_per_file = config.num_sequences // num_files
    remainder = config.num_sequences % num_files

    metadata = {
        'preset': preset,
        'config': {
            'num_sequences': config.num_sequences,
            'min_length': config.min_length,
            'max_length': config.max_length,
            'gc_content': 0.5,
        },
        'num_files': num_files,
        'sequences_per_file': sequences_per_file,
        'seed': seed,
        'files': [],
    }

    # Generate each file
    sequence_offset = 0
    for file_idx in range(num_files):
        # Last file gets remainder sequences
        num_seqs = sequences_per_file + (remainder if file_idx == num_files - 1 else 0)

        file_path = dataset_dir / f"file_{file_idx+1:03d}.fa"

        # Generate with offset seed for variation between files
        file_seed = seed + file_idx if seed is not None else None

        generate_synthetic_fasta(
            num_sequences=num_seqs,
            min_length=config.min_length,
            max_length=config.max_length,
            output_path=file_path,
            seed=file_seed,
            gc_content=0.5
        )

        # Record metadata
        file_size = file_path.stat().st_size
        metadata['files'].append({
            'filename': file_path.name,
            'path': str(file_path),
            'num_sequences': num_seqs,
            'size_bytes': file_size,
        })

        sequence_offset += num_seqs

    # Calculate total size
    total_size = sum(f['size_bytes'] for f in metadata['files'])
    metadata['total_size_bytes'] = total_size
    metadata['total_size_mb'] = total_size / (1024 ** 2)

    # Save metadata
    metadata_path = dataset_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset complete:")
    logger.info(f"  Files: {num_files}")
    logger.info(f"  Total sequences: {config.num_sequences}")
    logger.info(f"  Total size: {metadata['total_size_mb']:.2f} MB")

    return dataset_dir


# ==================== Real Sample Loading ====================

def real_sample_loader(data_dir: Path,
                      max_sequences: Optional[int] = None) -> Optional[List[Tuple[str, str]]]:
    """
    Load real viral samples from test data directory.

    Searches for FASTA files in the specified directory and loads them.
    Returns None if no data available (allows benchmarks to run with synthetic only).

    Args:
        data_dir: Path to directory with real samples (e.g., 'tests/data/small_real')
        max_sequences: Maximum sequences to load (None = all)

    Returns:
        List of (sequence_id, sequence) tuples, or None if no data found

    Example:
        >>> samples = real_sample_loader(Path('tests/data/small_real'))
        >>> if samples:
        ...     print(f"Loaded {len(samples)} real samples")
        ... else:
        ...     print("No real samples available, using synthetic")
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        logger.info(f"Real sample directory not found: {data_dir}")
        return None

    # Find FASTA files
    fasta_files = list(data_dir.glob('*.fa')) + list(data_dir.glob('*.fasta'))

    if not fasta_files:
        logger.info(f"No FASTA files found in {data_dir}")
        return None

    logger.info(f"Loading real samples from {data_dir}")

    sequences = []
    loaded_count = 0

    for fasta_file in fasta_files:
        try:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                sequences.append((record.id, str(record.seq)))
                loaded_count += 1

                if max_sequences and loaded_count >= max_sequences:
                    break

            if max_sequences and loaded_count >= max_sequences:
                break

        except Exception as e:
            logger.warning(f"Failed to load {fasta_file}: {e}")
            continue

    if sequences:
        logger.info(f"Loaded {len(sequences)} real samples from {len(fasta_files)} files")
        return sequences
    else:
        logger.info("No sequences loaded from real samples")
        return None


# ==================== Convenience Functions ====================

def generate_test_dataset(num_sequences: int = 100,
                         output_path: Optional[Path] = None,
                         seed: int = 42) -> Path:
    """
    Quick helper to generate a test dataset with default parameters.

    Args:
        num_sequences: Number of sequences (default: 100)
        output_path: Output path (default: temp file in tests/data/synthetic)
        seed: Random seed

    Returns:
        Path to generated FASTA file
    """
    if output_path is None:
        output_path = Path('tests/data/synthetic') / f'test_{num_sequences}.fa'

    return generate_synthetic_fasta(
        num_sequences=num_sequences,
        min_length=200,
        max_length=800,
        output_path=output_path,
        seed=seed
    )


def get_preset_config(preset: str) -> DatasetConfig:
    """
    Get configuration for a preset dataset.

    Args:
        preset: Preset name ('TINY', 'SMALL', 'MEDIUM', 'LARGE')

    Returns:
        DatasetConfig object

    Raises:
        ValueError: If preset not found
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")

    return PRESETS[preset]


# ==================== CLI Support ====================

def main():
    """Command-line interface for data generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic benchmark datasets')
    parser.add_argument('preset', choices=list(PRESETS.keys()),
                       help='Dataset preset size')
    parser.add_argument('--output-dir', type=Path, default=Path('tests/data/synthetic'),
                       help='Output directory (default: tests/data/synthetic)')
    parser.add_argument('--num-files', type=int, default=1,
                       help='Number of FASTA files to generate (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataset_dir = generate_benchmark_dataset(
        preset=args.preset,
        output_dir=args.output_dir,
        num_files=args.num_files,
        seed=args.seed
    )

    print(f"\n✓ Dataset generated: {dataset_dir}")
    print(f"  Use in benchmarks: pytest tests/benchmarks/ --test-data={dataset_dir}")


if __name__ == '__main__':
    main()
