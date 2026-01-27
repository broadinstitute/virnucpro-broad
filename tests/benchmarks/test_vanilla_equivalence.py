"""Vanilla equivalence validation tests.

Tests validate that optimized pipeline produces equivalent results to vanilla
implementation within acceptable numerical tolerance.

Purpose: Ensure optimizations don't compromise prediction accuracy
Strategy: Compare vanilla vs optimized outputs at multiple pipeline stages
Tolerance: rtol=1e-3 for BF16/FP32 precision differences
"""

import pytest
import torch
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass, asdict

from tests.benchmarks.vanilla_baseline import (
    VanillaRunner,
    VanillaConfig,
    load_predictions,
    load_embeddings,
    compare_files
)
from tests.benchmarks.data_generator import generate_benchmark_dataset, PRESETS

logger = logging.getLogger('virnucpro.benchmarks.test_equivalence')


# ==================== Configuration ====================

# BF16/FP32 tolerance based on research (rtol=1e-3 recommended)
EQUIVALENCE_TOLERANCE = {
    'rtol': 1e-3,  # Relative tolerance
    'atol': 1e-5,  # Absolute tolerance
}


# ==================== Test Class ====================

@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.equivalence
class TestVanillaEquivalence:
    """
    Validate optimized pipeline produces equivalent results to vanilla.

    Tests compare vanilla (all optimizations disabled) vs optimized outputs
    at multiple pipeline stages with appropriate numerical tolerances for
    BF16/FP32 precision differences.
    """

    @pytest.fixture(autouse=True)
    def setup_datasets(self, benchmark_dir, tmp_path):
        """Setup test datasets for equivalence testing."""
        self.benchmark_dir = benchmark_dir
        self.data_dir = benchmark_dir / "data" / "equivalence"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Generate small dataset for faster comparison
        logger.info("Generating equivalence test dataset...")
        from tests.benchmarks.data_generator import generate_synthetic_fasta

        self.test_input = generate_synthetic_fasta(
            num_sequences=100,
            min_length=200,
            max_length=800,
            output_path=self.data_dir / "test_100.fa",
            seed=42
        )

        logger.info(f"Test input: {self.test_input}")

        # Output directories
        self.vanilla_output = benchmark_dir / "outputs" / "vanilla"
        self.optimized_output = benchmark_dir / "outputs" / "optimized"

    def test_prediction_equivalence(self, single_gpu):
        """
        Compare final prediction CSV files between vanilla and optimized pipelines.

        Validates:
        - Same file paths predicted
        - Same prediction labels
        - Confidence scores within tolerance (rtol=1e-3)

        This is the most important test - predictions must match exactly,
        scores can vary slightly due to BF16/FP32 precision.
        """
        logger.info("=" * 80)
        logger.info("TEST: Prediction equivalence (vanilla vs optimized)")
        logger.info("=" * 80)

        # Run vanilla pipeline
        logger.info("Running vanilla pipeline...")
        vanilla_runner = VanillaRunner()
        vanilla_result = vanilla_runner.run_pipeline(
            input_dir=self.data_dir,
            output_dir=self.vanilla_output,
            gpu_id=single_gpu['gpu_ids'][0],
            timeout=600  # 10 minutes
        )

        assert vanilla_result['exit_code'] == 0, f"Vanilla pipeline failed: {vanilla_result['stderr']}"
        logger.info(f"Vanilla completed in {vanilla_result['duration']:.1f}s")

        # Run optimized pipeline
        logger.info("Running optimized pipeline...")
        optimized_result = self._run_optimized_pipeline(
            input_dir=self.data_dir,
            output_dir=self.optimized_output,
            gpu_id=single_gpu['gpu_ids'][0],
            timeout=600
        )

        assert optimized_result['exit_code'] == 0, f"Optimized pipeline failed: {optimized_result['stderr']}"
        logger.info(f"Optimized completed in {optimized_result['duration']:.1f}s")

        # Compare prediction files
        vanilla_preds = self.vanilla_output / "prediction_results_highestscore.csv"
        optimized_preds = self.optimized_output / "prediction_results_highestscore.csv"

        assert vanilla_preds.exists(), "Vanilla predictions not found"
        assert optimized_preds.exists(), "Optimized predictions not found"

        # Load predictions
        vanilla_df = load_predictions(vanilla_preds)
        optimized_df = load_predictions(optimized_preds)

        # Compare structure
        assert vanilla_df.shape == optimized_df.shape, \
            f"Prediction shape mismatch: {vanilla_df.shape} vs {optimized_df.shape}"

        logger.info(f"Comparing {len(vanilla_df)} predictions...")

        # Compare file paths (must be identical)
        if 'file_path' in vanilla_df.columns:
            assert (vanilla_df['file_path'] == optimized_df['file_path']).all(), \
                "File paths don't match between vanilla and optimized"

        # Compare prediction labels (must be identical)
        if 'Prediction' in vanilla_df.columns:
            predictions_match = (vanilla_df['Prediction'] == optimized_df['Prediction']).all()
            if not predictions_match:
                mismatches = vanilla_df[vanilla_df['Prediction'] != optimized_df['Prediction']]
                logger.error(f"Prediction mismatches:\n{mismatches}")

            assert predictions_match, "Predictions don't match between vanilla and optimized"
            logger.info("✓ All predictions match exactly")

        # Compare confidence scores with tolerance
        import numpy as np

        score_cols = [col for col in vanilla_df.columns if 'score' in col.lower()]
        for col in score_cols:
            vanilla_scores = vanilla_df[col].astype(float).values
            optimized_scores = optimized_df[col].astype(float).values

            matches = np.allclose(
                vanilla_scores,
                optimized_scores,
                rtol=EQUIVALENCE_TOLERANCE['rtol'],
                atol=EQUIVALENCE_TOLERANCE['atol']
            )

            max_diff = np.max(np.abs(vanilla_scores - optimized_scores))
            mean_diff = np.mean(np.abs(vanilla_scores - optimized_scores))

            logger.info(f"Score column: {col}")
            logger.info(f"  Max diff: {max_diff:.6f}")
            logger.info(f"  Mean diff: {mean_diff:.6f}")

            assert matches, \
                f"Scores don't match within tolerance for {col} (max_diff={max_diff:.6f})"

        logger.info("✓ All confidence scores match within tolerance")

    def test_embedding_equivalence(self, single_gpu):
        """
        Compare DNABERT-S and ESM-2 embeddings between vanilla and optimized.

        Validates:
        - Embedding tensor shapes match
        - Values within BF16/FP32 tolerance (rtol=1e-3, atol=1e-5)
        - Reports max absolute and relative differences

        This test accounts for precision differences when BF16 is used in
        optimized pipeline vs FP32 in vanilla.
        """
        logger.info("=" * 80)
        logger.info("TEST: Embedding equivalence (vanilla vs optimized)")
        logger.info("=" * 80)

        # Run both pipelines (reuse if already run from previous test)
        if not (self.vanilla_output / "features_dnabert").exists():
            logger.info("Running vanilla pipeline for embeddings...")
            vanilla_runner = VanillaRunner()
            vanilla_result = vanilla_runner.run_pipeline(
                input_dir=self.data_dir,
                output_dir=self.vanilla_output,
                gpu_id=single_gpu['gpu_ids'][0],
                timeout=600
            )
            assert vanilla_result['exit_code'] == 0

        if not (self.optimized_output / "features_dnabert").exists():
            logger.info("Running optimized pipeline for embeddings...")
            optimized_result = self._run_optimized_pipeline(
                input_dir=self.data_dir,
                output_dir=self.optimized_output,
                gpu_id=single_gpu['gpu_ids'][0],
                timeout=600
            )
            assert optimized_result['exit_code'] == 0

        # Compare DNABERT-S embeddings
        logger.info("Comparing DNABERT-S embeddings...")
        dnabert_vanilla_dir = self.vanilla_output / "features_dnabert"
        dnabert_optimized_dir = self.optimized_output / "features_dnabert"

        assert dnabert_vanilla_dir.exists(), "Vanilla DNABERT embeddings not found"
        assert dnabert_optimized_dir.exists(), "Optimized DNABERT embeddings not found"

        vanilla_dnabert_files = sorted(dnabert_vanilla_dir.glob("*.pt"))
        optimized_dnabert_files = sorted(dnabert_optimized_dir.glob("*.pt"))

        assert len(vanilla_dnabert_files) == len(optimized_dnabert_files), \
            f"DNABERT file count mismatch: {len(vanilla_dnabert_files)} vs {len(optimized_dnabert_files)}"

        dnabert_diffs = self._compare_embedding_files(vanilla_dnabert_files, optimized_dnabert_files, "DNABERT-S")

        # Compare ESM-2 embeddings
        logger.info("Comparing ESM-2 embeddings...")
        esm_vanilla_dir = self.vanilla_output / "features_esm"
        esm_optimized_dir = self.optimized_output / "features_esm"

        assert esm_vanilla_dir.exists(), "Vanilla ESM-2 embeddings not found"
        assert esm_optimized_dir.exists(), "Optimized ESM-2 embeddings not found"

        vanilla_esm_files = sorted(esm_vanilla_dir.glob("*.pt"))
        optimized_esm_files = sorted(esm_optimized_dir.glob("*.pt"))

        assert len(vanilla_esm_files) == len(optimized_esm_files), \
            f"ESM-2 file count mismatch: {len(vanilla_esm_files)} vs {len(optimized_esm_files)}"

        esm_diffs = self._compare_embedding_files(vanilla_esm_files, optimized_esm_files, "ESM-2")

        # Report overall statistics
        logger.info("=" * 80)
        logger.info("EMBEDDING EQUIVALENCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"DNABERT-S: max_diff={dnabert_diffs['max_diff']:.6e}, mean_diff={dnabert_diffs['mean_diff']:.6e}")
        logger.info(f"ESM-2:     max_diff={esm_diffs['max_diff']:.6e}, mean_diff={esm_diffs['mean_diff']:.6e}")
        logger.info(f"Tolerance: rtol={EQUIVALENCE_TOLERANCE['rtol']}, atol={EQUIVALENCE_TOLERANCE['atol']}")

    def test_consensus_equivalence(self, single_gpu):
        """
        Compare consensus sequences between vanilla and optimized.

        Validates:
        - Consensus sequences match exactly
        - Translation consistency

        Consensus generation should be deterministic with no floating point
        operations, so this should match exactly (no tolerance).
        """
        logger.info("=" * 80)
        logger.info("TEST: Consensus equivalence (vanilla vs optimized)")
        logger.info("=" * 80)

        # Run both pipelines if not already done
        if not (self.vanilla_output / "consensus_sequences.csv").exists():
            logger.info("Running vanilla pipeline for consensus...")
            vanilla_runner = VanillaRunner()
            vanilla_result = vanilla_runner.run_pipeline(
                input_dir=self.data_dir,
                output_dir=self.vanilla_output,
                gpu_id=single_gpu['gpu_ids'][0],
                timeout=600
            )
            assert vanilla_result['exit_code'] == 0

        if not (self.optimized_output / "consensus_sequences.csv").exists():
            logger.info("Running optimized pipeline for consensus...")
            optimized_result = self._run_optimized_pipeline(
                input_dir=self.data_dir,
                output_dir=self.optimized_output,
                gpu_id=single_gpu['gpu_ids'][0],
                timeout=600
            )
            assert optimized_result['exit_code'] == 0

        # Compare consensus files
        vanilla_consensus = self.vanilla_output / "consensus_sequences.csv"
        optimized_consensus = self.optimized_output / "consensus_sequences.csv"

        # Consensus may not exist for all pipelines
        if not vanilla_consensus.exists():
            logger.warning("Vanilla consensus not found - skipping consensus comparison")
            pytest.skip("Consensus sequences not generated")

        assert optimized_consensus.exists(), "Optimized consensus not found"

        # Load and compare
        vanilla_df = load_predictions(vanilla_consensus)
        optimized_df = load_predictions(optimized_consensus)

        assert vanilla_df.shape == optimized_df.shape, \
            f"Consensus shape mismatch: {vanilla_df.shape} vs {optimized_df.shape}"

        # Compare all columns - should match exactly
        for col in vanilla_df.columns:
            if col not in optimized_df.columns:
                logger.error(f"Column {col} missing from optimized consensus")
                continue

            matches = (vanilla_df[col] == optimized_df[col]).all()
            if not matches:
                mismatches = vanilla_df[vanilla_df[col] != optimized_df[col]]
                logger.error(f"Consensus mismatches in column {col}:\n{mismatches}")

            assert matches, f"Consensus column {col} doesn't match"

        logger.info("✓ All consensus sequences match exactly")

    def test_incremental_equivalence(self, single_gpu):
        """
        Test each optimization individually to identify divergence sources.

        Runs 5 configurations:
        1. Vanilla (all optimizations disabled)
        2. Multi-GPU only (single GPU for test, but parallel codepath)
        3. BF16 only
        4. FlashAttention only
        5. All optimizations

        Identifies which specific optimization causes numerical divergence.
        """
        logger.info("=" * 80)
        logger.info("TEST: Incremental optimization equivalence")
        logger.info("=" * 80)

        configs = {
            'vanilla': VanillaConfig(
                use_bf16=False,
                use_flash_attention=False,
                use_cuda_streams=False,
                use_persistent_models=False,
                parallel_processing=False
            ),
            'bf16_only': VanillaConfig(
                use_bf16=True,
                use_flash_attention=False,
                use_cuda_streams=False,
                use_persistent_models=False,
                parallel_processing=False
            ),
            'flash_only': VanillaConfig(
                use_bf16=False,
                use_flash_attention=True,
                use_cuda_streams=False,
                use_persistent_models=False,
                parallel_processing=False
            ),
            'streams_only': VanillaConfig(
                use_bf16=False,
                use_flash_attention=False,
                use_cuda_streams=True,
                use_persistent_models=False,
                parallel_processing=False
            ),
            'all_optimizations': VanillaConfig(
                use_bf16=True,
                use_flash_attention=True,
                use_cuda_streams=True,
                use_persistent_models=True,
                parallel_processing=False
            ),
        }

        # Run each configuration
        results = {}
        vanilla_preds = None

        for config_name, config in configs.items():
            logger.info(f"Running configuration: {config_name}")

            output_dir = self.benchmark_dir / "outputs" / f"incremental_{config_name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            runner = VanillaRunner(config=config)
            result = runner.run_pipeline(
                input_dir=self.data_dir,
                output_dir=output_dir,
                gpu_id=single_gpu['gpu_ids'][0],
                timeout=600
            )

            assert result['exit_code'] == 0, f"{config_name} failed: {result['stderr']}"

            # Load predictions
            pred_file = output_dir / "prediction_results_highestscore.csv"
            assert pred_file.exists(), f"Predictions not found for {config_name}"

            preds = load_predictions(pred_file)

            if config_name == 'vanilla':
                vanilla_preds = preds
                results[config_name] = {
                    'duration': result['duration'],
                    'predictions': preds,
                    'vs_vanilla': {'match': True, 'max_diff': 0.0}
                }
            else:
                # Compare to vanilla
                comparison = self._compare_predictions(vanilla_preds, preds, config_name)
                results[config_name] = {
                    'duration': result['duration'],
                    'predictions': preds,
                    'vs_vanilla': comparison
                }

        # Generate comparison report
        self._generate_incremental_report(results)

        # All configurations should match vanilla within tolerance
        for config_name, result in results.items():
            if config_name == 'vanilla':
                continue

            assert result['vs_vanilla']['match'], \
                f"{config_name} diverges from vanilla (max_diff={result['vs_vanilla']['max_diff']:.6e})"

        logger.info("✓ All optimization configurations match vanilla within tolerance")

    # ==================== Helper Methods ====================

    def _run_optimized_pipeline(self,
                                input_dir: Path,
                                output_dir: Path,
                                gpu_id: int,
                                timeout: Optional[int] = None) -> Dict:
        """Run optimized pipeline with all optimizations enabled."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command with optimizations
        cmd = [
            'virnucpro', 'predict',
            '--input', str(input_dir),
            '--output', str(output_dir),
            '--gpus', str(gpu_id),
            '--cuda-streams',  # Enable CUDA streams
            # BF16 and FlashAttention auto-enabled on Ampere+ GPUs
        ]

        logger.info(f"Running optimized pipeline: {' '.join(cmd)}")

        import time
        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        duration = time.time() - start_time

        return {
            'duration': duration,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }

    def _compare_embedding_files(self,
                                 vanilla_files: List[Path],
                                 optimized_files: List[Path],
                                 model_name: str) -> Dict[str, float]:
        """Compare embedding files and return difference statistics."""
        import numpy as np

        all_max_diffs = []
        all_mean_diffs = []
        all_match = True

        for vanilla_file, optimized_file in zip(vanilla_files, optimized_files):
            # Load embeddings
            vanilla_emb = load_embeddings(vanilla_file)
            optimized_emb = load_embeddings(optimized_file)

            # Compare shapes
            if vanilla_emb.shape != optimized_emb.shape:
                logger.error(f"Shape mismatch in {vanilla_file.name}: {vanilla_emb.shape} vs {optimized_emb.shape}")
                all_match = False
                continue

            # Compare values
            matches = torch.allclose(
                vanilla_emb,
                optimized_emb,
                rtol=EQUIVALENCE_TOLERANCE['rtol'],
                atol=EQUIVALENCE_TOLERANCE['atol']
            )

            max_diff = torch.max(torch.abs(vanilla_emb - optimized_emb)).item()
            mean_diff = torch.mean(torch.abs(vanilla_emb - optimized_emb)).item()

            all_max_diffs.append(max_diff)
            all_mean_diffs.append(mean_diff)

            if not matches:
                logger.warning(f"{model_name} embeddings don't match in {vanilla_file.name} (max_diff={max_diff:.6e})")
                all_match = False
            else:
                logger.info(f"✓ {vanilla_file.name}: max_diff={max_diff:.6e}")

        assert all_match, f"{model_name} embeddings don't match within tolerance"

        return {
            'max_diff': np.max(all_max_diffs) if all_max_diffs else 0.0,
            'mean_diff': np.mean(all_mean_diffs) if all_mean_diffs else 0.0,
        }

    def _compare_predictions(self, vanilla_df, optimized_df, config_name: str) -> Dict:
        """Compare prediction DataFrames and return comparison results."""
        import numpy as np

        # Check predictions match
        predictions_match = (vanilla_df['Prediction'] == optimized_df['Prediction']).all()

        # Check scores
        score_cols = [col for col in vanilla_df.columns if 'score' in col.lower()]
        max_diffs = []

        for col in score_cols:
            vanilla_scores = vanilla_df[col].astype(float).values
            optimized_scores = optimized_df[col].astype(float).values

            matches = np.allclose(
                vanilla_scores,
                optimized_scores,
                rtol=EQUIVALENCE_TOLERANCE['rtol'],
                atol=EQUIVALENCE_TOLERANCE['atol']
            )

            max_diff = np.max(np.abs(vanilla_scores - optimized_scores))
            max_diffs.append(max_diff)

        overall_max_diff = np.max(max_diffs) if max_diffs else 0.0

        return {
            'match': predictions_match and (overall_max_diff <= EQUIVALENCE_TOLERANCE['rtol']),
            'predictions_match': predictions_match,
            'max_diff': overall_max_diff,
        }

    def _generate_incremental_report(self, results: Dict):
        """Generate comparison report for incremental optimizations."""
        logger.info("=" * 80)
        logger.info("INCREMENTAL OPTIMIZATION COMPARISON")
        logger.info("=" * 80)

        for config_name, result in results.items():
            logger.info(f"\n{config_name}:")
            logger.info(f"  Duration: {result['duration']:.1f}s")
            logger.info(f"  vs Vanilla: match={result['vs_vanilla']['match']}, max_diff={result['vs_vanilla']['max_diff']:.6e}")

        # Save report
        report_path = self.benchmark_dir / "reports" / "incremental_equivalence.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'tolerance': EQUIVALENCE_TOLERANCE,
            'configurations': {
                config_name: {
                    'duration': result['duration'],
                    'vs_vanilla': result['vs_vanilla']
                }
                for config_name, result in results.items()
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nReport saved to {report_path}")
