"""Vanilla baseline runner for correctness validation.

Provides:
- VanillaRunner: Run pipeline with all optimizations disabled
- Baseline comparison utilities for validating optimized pipeline
- Reference output generation for equivalence testing

Purpose: Establish ground truth for correctness validation
Strategy: Run pipeline in minimal configuration matching original 45-hour baseline
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger('virnucpro.benchmarks.vanilla_baseline')


# ==================== Vanilla Runner ====================

@dataclass
class VanillaConfig:
    """Configuration for vanilla baseline run."""
    use_bf16: bool = False
    use_flash_attention: bool = False
    use_cuda_streams: bool = False
    use_persistent_models: bool = False
    parallel_processing: bool = False
    num_gpus: int = 1
    gpu_id: int = 0

    def to_cli_args(self) -> List[str]:
        """Convert configuration to CLI arguments."""
        args = []

        # Force single GPU
        args.extend(['--gpus', str(self.gpu_id)])

        # Disable optimizations explicitly
        if not self.use_cuda_streams:
            args.append('--no-cuda-streams')

        if not self.use_persistent_models:
            # Default is False, so nothing needed
            pass
        else:
            args.append('--persistent-models')

        # Ensure single-threaded for determinism
        args.extend(['--threads', '1'])

        return args


class VanillaRunner:
    """
    Run pipeline with all optimizations disabled.

    Provides baseline "ground truth" for correctness validation.
    Matches original 45-hour baseline configuration.

    Example:
        >>> runner = VanillaRunner()
        >>> result = runner.run_pipeline(
        ...     input_dir=Path('tests/data/small'),
        ...     output_dir=Path('tests/output/vanilla'),
        ...     gpu_id=0
        ... )
        >>> print(f"Completed in {result['duration']:.1f}s")
    """

    def __init__(self, config: Optional[VanillaConfig] = None):
        """
        Initialize vanilla runner.

        Args:
            config: Vanilla configuration (default: all optimizations disabled)
        """
        self.config = config or VanillaConfig()

    def run_pipeline(self,
                    input_dir: Path,
                    output_dir: Path,
                    model_path: Optional[Path] = None,
                    expected_length: int = 1000,
                    gpu_id: Optional[int] = None,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run vanilla pipeline on input data.

        Args:
            input_dir: Directory containing input FASTA files
            output_dir: Output directory for results
            model_path: Path to trained model (optional, uses default)
            expected_length: Expected sequence length
            gpu_id: GPU device ID (optional, uses config default)
            timeout: Timeout in seconds (optional, default: None)

        Returns:
            Dictionary with:
            - duration: Execution time (seconds)
            - exit_code: Process exit code
            - stdout: Standard output
            - stderr: Standard error
            - output_files: Dictionary of generated files

        Raises:
            subprocess.TimeoutExpired: If timeout exceeded
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Override GPU ID if specified
        if gpu_id is not None:
            self.config.gpu_id = gpu_id

        # Build command
        cmd = self._build_command(
            input_dir=input_dir,
            output_dir=output_dir,
            model_path=model_path,
            expected_length=expected_length
        )

        logger.info(f"Running vanilla pipeline: {' '.join(map(str, cmd))}")

        # Run pipeline
        import time
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise on non-zero exit
            )

            duration = time.time() - start_time

            # Collect output files
            output_files = self._collect_output_files(output_dir)

            return {
                'duration': duration,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'output_files': output_files,
                'config': asdict(self.config),
            }

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            logger.error(f"Pipeline timed out after {duration:.1f}s")
            raise

    def _build_command(self,
                      input_dir: Path,
                      output_dir: Path,
                      model_path: Optional[Path],
                      expected_length: int) -> List[str]:
        """Build virnucpro CLI command with vanilla settings."""
        cmd = ['virnucpro', 'predict']

        # Input/output
        cmd.extend(['--input', str(input_dir)])
        cmd.extend(['--output', str(output_dir)])

        # Model
        if model_path:
            cmd.extend(['--model', str(model_path)])

        # Expected length
        cmd.extend(['--expected-length', str(expected_length)])

        # Add vanilla configuration flags
        cmd.extend(self.config.to_cli_args())

        return cmd

    def _collect_output_files(self, output_dir: Path) -> Dict[str, Path]:
        """Collect all output files from pipeline run."""
        output_files = {}

        # Prediction results
        pred_file = output_dir / 'prediction_results_highestscore.csv'
        if pred_file.exists():
            output_files['predictions'] = pred_file

        # DNABERT embeddings
        dnabert_dir = output_dir / 'features_dnabert'
        if dnabert_dir.exists():
            output_files['dnabert_embeddings'] = list(dnabert_dir.glob('*.pt'))

        # ESM-2 embeddings
        esm_dir = output_dir / 'features_esm'
        if esm_dir.exists():
            output_files['esm_embeddings'] = list(esm_dir.glob('*.pt'))

        # Consensus sequences
        consensus_file = output_dir / 'consensus_sequences.csv'
        if consensus_file.exists():
            output_files['consensus'] = consensus_file

        return output_files


# ==================== Reference Output Generation ====================

def generate_reference_outputs(input_dir: Path,
                               output_dir: Path,
                               gpu_id: int = 0,
                               timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate reference outputs for equivalence testing.

    Runs vanilla pipeline and saves outputs to reference directory.
    Generates checksums for validation.

    Args:
        input_dir: Input FASTA files
        output_dir: Reference output directory (e.g., 'tests/data/reference')
        gpu_id: GPU device ID
        timeout: Timeout in seconds

    Returns:
        Dictionary with:
        - output_files: Paths to generated files
        - checksums: SHA256 checksums of outputs
        - metadata: Run metadata

    Example:
        >>> generate_reference_outputs(
        ...     input_dir=Path('tests/data/small'),
        ...     output_dir=Path('tests/data/reference/small')
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run vanilla pipeline
    runner = VanillaRunner()
    result = runner.run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        gpu_id=gpu_id,
        timeout=timeout
    )

    if result['exit_code'] != 0:
        logger.error("Reference generation failed!")
        logger.error(f"stderr: {result['stderr']}")
        raise RuntimeError(f"Pipeline failed with exit code {result['exit_code']}")

    # Generate checksums
    checksums = {}
    for file_type, file_path in result['output_files'].items():
        if isinstance(file_path, list):
            # Multiple files (embeddings)
            checksums[file_type] = {}
            for f in file_path:
                checksums[file_type][f.name] = _compute_checksum(f)
        else:
            # Single file
            checksums[file_type] = _compute_checksum(file_path)

    # Save metadata
    metadata = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'duration': result['duration'],
        'gpu_id': gpu_id,
        'config': result['config'],
        'checksums': checksums,
    }

    metadata_path = output_dir / 'reference_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Reference outputs generated in {result['duration']:.1f}s")
    logger.info(f"Metadata saved to {metadata_path}")

    return {
        'output_files': result['output_files'],
        'checksums': checksums,
        'metadata': metadata,
    }


def _compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of file."""
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)

    return sha256.hexdigest()


# ==================== Baseline Comparison Utilities ====================

def load_predictions(csv_path: Path):
    """
    Load prediction results from CSV file.

    Args:
        csv_path: Path to prediction_results_highestscore.csv

    Returns:
        DataFrame with predictions
    """
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions not found: {csv_path}")

    df = pd.read_csv(csv_path, sep='\t')

    # Sort by file_path for consistent comparison
    if 'file_path' in df.columns:
        df = df.sort_values('file_path').reset_index(drop=True)

    return df


def load_embeddings(pt_path: Path):
    """
    Load embedding tensors from .pt file.

    Args:
        pt_path: Path to .pt embedding file

    Returns:
        Tensor with embeddings
    """
    import torch

    if not pt_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {pt_path}")

    return torch.load(pt_path, map_location='cpu')


def compare_files(reference_path: Path,
                 test_path: Path,
                 file_type: str = 'auto') -> Dict[str, Any]:
    """
    Basic file comparison for validation.

    Args:
        reference_path: Reference file path
        test_path: Test file path
        file_type: File type ('csv', 'pt', 'auto' for detection)

    Returns:
        Dictionary with comparison results:
        - match: True if files match
        - details: Comparison details
    """
    if not reference_path.exists():
        return {'match': False, 'error': f"Reference not found: {reference_path}"}

    if not test_path.exists():
        return {'match': False, 'error': f"Test file not found: {test_path}"}

    # Auto-detect file type
    if file_type == 'auto':
        if reference_path.suffix == '.csv':
            file_type = 'csv'
        elif reference_path.suffix == '.pt':
            file_type = 'pt'
        else:
            file_type = 'binary'

    # CSV comparison
    if file_type == 'csv':
        import pandas as pd

        try:
            ref_df = pd.read_csv(reference_path, sep='\t')
            test_df = pd.read_csv(test_path, sep='\t')

            if ref_df.shape != test_df.shape:
                return {
                    'match': False,
                    'error': f"Shape mismatch: {ref_df.shape} vs {test_df.shape}"
                }

            # Check if predictions column matches
            if 'Prediction' in ref_df.columns:
                matches = (ref_df['Prediction'] == test_df['Prediction']).all()
                return {
                    'match': matches,
                    'num_rows': len(ref_df),
                    'predictions_match': matches
                }

            return {'match': True, 'num_rows': len(ref_df)}

        except Exception as e:
            return {'match': False, 'error': f"CSV comparison failed: {e}"}

    # PyTorch tensor comparison
    elif file_type == 'pt':
        import torch

        try:
            ref_tensor = torch.load(reference_path, map_location='cpu')
            test_tensor = torch.load(test_path, map_location='cpu')

            if ref_tensor.shape != test_tensor.shape:
                return {
                    'match': False,
                    'error': f"Shape mismatch: {ref_tensor.shape} vs {test_tensor.shape}"
                }

            # Use torch.allclose with BF16 tolerance
            matches = torch.allclose(ref_tensor, test_tensor, rtol=1e-3, atol=1e-5)

            max_diff = torch.max(torch.abs(ref_tensor - test_tensor)).item()

            return {
                'match': matches,
                'shape': list(ref_tensor.shape),
                'max_diff': max_diff,
            }

        except Exception as e:
            return {'match': False, 'error': f"Tensor comparison failed: {e}"}

    # Binary comparison (checksums)
    else:
        ref_checksum = _compute_checksum(reference_path)
        test_checksum = _compute_checksum(test_path)

        return {
            'match': ref_checksum == test_checksum,
            'ref_checksum': ref_checksum,
            'test_checksum': test_checksum,
        }


def get_baseline_timings(reference_metadata_path: Path) -> Dict[str, float]:
    """
    Get baseline timing information from reference run.

    Args:
        reference_metadata_path: Path to reference_metadata.json

    Returns:
        Dictionary with timing information:
        - total_duration: Total pipeline duration (seconds)
        - sequences_per_second: Throughput
    """
    if not reference_metadata_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {reference_metadata_path}")

    with open(reference_metadata_path, 'r') as f:
        metadata = json.load(f)

    return {
        'total_duration': metadata.get('duration', 0.0),
        'config': metadata.get('config', {}),
    }


# ==================== Convenience Functions ====================

def run_vanilla_pipeline(input_dir: Path,
                        output_dir: Path,
                        gpu_id: int = 0,
                        timeout: Optional[int] = None) -> Dict[str, Path]:
    """
    Quick helper to run vanilla pipeline and return output file paths.

    Args:
        input_dir: Input FASTA directory
        output_dir: Output directory
        gpu_id: GPU device ID (default: 0)
        timeout: Timeout in seconds (optional)

    Returns:
        Dictionary of output file paths

    Example:
        >>> outputs = run_vanilla_pipeline(
        ...     input_dir=Path('tests/data/small'),
        ...     output_dir=Path('tests/output/vanilla')
        ... )
        >>> print(f"Predictions: {outputs['predictions']}")
    """
    runner = VanillaRunner()
    result = runner.run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        gpu_id=gpu_id,
        timeout=timeout
    )

    if result['exit_code'] != 0:
        logger.error(f"Pipeline failed with exit code {result['exit_code']}")
        logger.error(f"stderr: {result['stderr']}")
        raise RuntimeError(f"Pipeline failed: {result['stderr'][:200]}")

    return result['output_files']
