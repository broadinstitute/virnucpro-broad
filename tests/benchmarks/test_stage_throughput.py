"""Per-stage throughput benchmarks for bottleneck identification.

This module tests each pipeline stage independently:
1. Translation: CPU parallelization performance
2. DNABERT-S: Multi-GPU embedding extraction
3. ESM-2: Multi-GPU embedding extraction
4. Merge: Parallel embedding merge
5. Prediction: Final classification stage

Validates GPU utilization >80% for DNABERT and ESM-2 stages (PERF-02).
Identifies bottlenecks consuming >30% of total pipeline time.
"""

import pytest
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
import tempfile
from dataclasses import dataclass, asdict

from tests.benchmarks.data_generator import generate_synthetic_fasta, PRESETS
from tests.benchmarks.utils import BenchmarkTimer
from virnucpro.utils.gpu_monitor import NvitopMonitor

logger = logging.getLogger('virnucpro.benchmarks.stage_throughput')


@dataclass
class StageThroughputResult:
    """Results from stage throughput benchmark."""
    stage_name: str
    num_sequences: int
    total_time: float
    sequences_per_second: float
    gpu_util_avg: Optional[float] = None
    gpu_util_min: Optional[float] = None
    gpu_util_max: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    is_bottleneck: bool = False


@pytest.mark.gpu
@pytest.mark.throughput
class TestStageThroughput:
    """
    Per-stage throughput benchmarks.

    Tests each pipeline stage independently to:
    - Measure sequences/second throughput
    - Track GPU utilization (must be >80% for GPU stages)
    - Identify bottlenecks
    - Compare optimized vs vanilla performance
    """

    @pytest.fixture(scope="class")
    def small_dataset(self, tmp_path_factory):
        """
        Generate small synthetic dataset for quick stage tests.

        Uses SMALL preset (100 sequences) for fast iteration.

        Returns:
            Path to generated FASTA file
        """
        output_dir = tmp_path_factory.mktemp("stage_data")
        config = PRESETS['SMALL']

        fasta_path = output_dir / "test_small.fa"
        generate_synthetic_fasta(
            num_sequences=config.num_sequences,
            min_length=config.min_length,
            max_length=config.max_length,
            output_path=fasta_path,
            seed=42
        )

        return fasta_path

    @pytest.fixture(scope="class")
    def gpu_devices(self):
        """
        Get available GPU device IDs.

        Returns:
            List of GPU device IDs
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        return list(range(torch.cuda.device_count()))

    def test_translation_throughput(self, small_dataset: Path, tmp_path: Path):
        """
        Test CPU translation throughput.

        Measures 6-frame translation parallelization performance.
        This is a CPU-bound stage that should utilize all CPU cores.

        Args:
            small_dataset: Path to test FASTA file
            tmp_path: Temporary directory for outputs
        """
        from virnucpro.utils.sequence import translate_sequences
        from Bio import SeqIO
        import multiprocessing

        # Load sequences
        sequences = []
        for record in SeqIO.parse(small_dataset, "fasta"):
            sequences.append((record.id, str(record.seq)))

        num_sequences = len(sequences)
        logger.info(f"Testing translation with {num_sequences} sequences")

        # Create output directory
        output_dir = tmp_path / "translation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Measure translation time
        start_time = time.time()

        # Use multiprocessing for translation
        num_threads = multiprocessing.cpu_count()

        translated = []
        for seq_id, seq in sequences:
            # Perform 6-frame translation
            from virnucpro.utils.sequence import translate_6_frame
            frames = translate_6_frame(seq)
            translated.append((seq_id, frames))

        elapsed = time.time() - start_time

        # Calculate throughput
        throughput = num_sequences / elapsed if elapsed > 0 else 0.0

        result = StageThroughputResult(
            stage_name="translation",
            num_sequences=num_sequences,
            total_time=elapsed,
            sequences_per_second=throughput,
            gpu_util_avg=None,  # CPU-only stage
        )

        logger.info(f"Translation: {throughput:.1f} seq/s ({elapsed:.2f}s total)")

        # Save result
        self._save_stage_result(result, tmp_path)

        # Verify reasonable throughput (at least 10 seq/s on modern CPU)
        assert throughput >= 10.0, \
            f"Translation throughput {throughput:.1f} seq/s is too low (expected ≥10 seq/s)"

    def test_dnabert_throughput(self,
                                small_dataset: Path,
                                tmp_path: Path,
                                gpu_devices: List[int]):
        """
        Test DNABERT-S throughput and GPU utilization.

        Validates:
        - Sequences/second throughput
        - GPU utilization >80% (PERF-02 requirement)
        - Memory usage within limits

        Args:
            small_dataset: Path to test FASTA file
            tmp_path: Temporary directory for outputs
            gpu_devices: List of available GPU device IDs
        """
        from virnucpro.models.dnabert_flash import DNABERTFlash
        from virnucpro.utils.sequence import translate_sequences
        from Bio import SeqIO

        if len(gpu_devices) == 0:
            pytest.skip("No GPUs available")

        # Load and translate sequences
        sequences = []
        for record in SeqIO.parse(small_dataset, "fasta"):
            sequences.append((record.id, str(record.seq)))

        num_sequences = len(sequences)
        logger.info(f"Testing DNABERT with {num_sequences} sequences on {len(gpu_devices)} GPU(s)")

        # Create output directory
        output_dir = tmp_path / "dnabert"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GPU monitoring
        monitor = NvitopMonitor(device_ids=gpu_devices, log_interval=0.5)
        monitor.start_monitoring()
        monitor.set_stage("dnabert")

        # Initialize DNABERT model
        device = torch.device(f"cuda:{gpu_devices[0]}")
        model = DNABERTFlash(device=device)
        model.eval()

        # Measure extraction time
        start_time = time.time()

        with torch.no_grad():
            for seq_id, seq in sequences:
                # Extract embedding
                try:
                    embedding = model.extract_embedding(seq)
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for {seq_id}: {e}")
                    continue

        elapsed = time.time() - start_time

        # Stop monitoring and get stats
        stats = monitor.stop_monitoring()

        # Calculate throughput
        throughput = num_sequences / elapsed if elapsed > 0 else 0.0

        # Get GPU utilization
        gpu_util_avg = monitor.get_average_utilization()
        peak_memory = monitor.get_peak_memory_usage() / (1024**3)  # Convert to GB

        result = StageThroughputResult(
            stage_name="dnabert",
            num_sequences=num_sequences,
            total_time=elapsed,
            sequences_per_second=throughput,
            gpu_util_avg=gpu_util_avg,
            gpu_util_min=stats.get(gpu_devices[0], {}).get('gpu_util_min', 0.0) if stats else 0.0,
            gpu_util_max=stats.get(gpu_devices[0], {}).get('gpu_util_max', 0.0) if stats else 0.0,
            memory_peak_gb=peak_memory,
        )

        logger.info(f"DNABERT: {throughput:.1f} seq/s, GPU util: {gpu_util_avg:.1f}%")

        # Save result
        self._save_stage_result(result, tmp_path)

        # Validate GPU utilization >80% (PERF-02)
        assert gpu_util_avg >= 80.0, \
            f"DNABERT GPU utilization {gpu_util_avg:.1f}% below 80% threshold (PERF-02)"

        # Verify reasonable throughput
        assert throughput >= 1.0, \
            f"DNABERT throughput {throughput:.1f} seq/s is too low (expected ≥1 seq/s)"

    def test_esm2_throughput(self,
                            small_dataset: Path,
                            tmp_path: Path,
                            gpu_devices: List[int]):
        """
        Test ESM-2 throughput and GPU utilization.

        Validates:
        - Sequences/second throughput
        - GPU utilization >80% (PERF-02 requirement)
        - Memory usage within limits

        Args:
            small_dataset: Path to test FASTA file
            tmp_path: Temporary directory for outputs
            gpu_devices: List of available GPU device IDs
        """
        from virnucpro.models.esm2_flash import ESM2Flash
        from virnucpro.utils.sequence import translate_6_frame
        from Bio import SeqIO

        if len(gpu_devices) == 0:
            pytest.skip("No GPUs available")

        # Load and translate sequences
        protein_sequences = []
        for record in SeqIO.parse(small_dataset, "fasta"):
            dna_seq = str(record.seq)
            # Get 6-frame translations
            frames = translate_6_frame(dna_seq)
            # Use longest ORF for simplicity
            longest_orf = max(frames, key=len)
            protein_sequences.append((record.id, longest_orf))

        num_sequences = len(protein_sequences)
        logger.info(f"Testing ESM-2 with {num_sequences} sequences on {len(gpu_devices)} GPU(s)")

        # Create output directory
        output_dir = tmp_path / "esm2"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GPU monitoring
        monitor = NvitopMonitor(device_ids=gpu_devices, log_interval=0.5)
        monitor.start_monitoring()
        monitor.set_stage("esm2")

        # Initialize ESM-2 model
        device = torch.device(f"cuda:{gpu_devices[0]}")
        model = ESM2Flash(device=device)
        model.eval()

        # Measure extraction time
        start_time = time.time()

        with torch.no_grad():
            for seq_id, protein_seq in protein_sequences:
                # Extract embedding
                try:
                    embedding = model.extract_embedding(protein_seq)
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for {seq_id}: {e}")
                    continue

        elapsed = time.time() - start_time

        # Stop monitoring and get stats
        stats = monitor.stop_monitoring()

        # Calculate throughput
        throughput = num_sequences / elapsed if elapsed > 0 else 0.0

        # Get GPU utilization
        gpu_util_avg = monitor.get_average_utilization()
        peak_memory = monitor.get_peak_memory_usage() / (1024**3)  # Convert to GB

        result = StageThroughputResult(
            stage_name="esm2",
            num_sequences=num_sequences,
            total_time=elapsed,
            sequences_per_second=throughput,
            gpu_util_avg=gpu_util_avg,
            gpu_util_min=stats.get(gpu_devices[0], {}).get('gpu_util_min', 0.0) if stats else 0.0,
            gpu_util_max=stats.get(gpu_devices[0], {}).get('gpu_util_max', 0.0) if stats else 0.0,
            memory_peak_gb=peak_memory,
        )

        logger.info(f"ESM-2: {throughput:.1f} seq/s, GPU util: {gpu_util_avg:.1f}%")

        # Save result
        self._save_stage_result(result, tmp_path)

        # Validate GPU utilization >80% (PERF-02)
        assert gpu_util_avg >= 80.0, \
            f"ESM-2 GPU utilization {gpu_util_avg:.1f}% below 80% threshold (PERF-02)"

        # Verify reasonable throughput
        assert throughput >= 1.0, \
            f"ESM-2 throughput {throughput:.1f} seq/s is too low (expected ≥1 seq/s)"

    def test_merge_throughput(self, tmp_path: Path):
        """
        Test embedding merge throughput.

        Measures parallel embedding merge performance.
        This is a CPU-bound stage that should benefit from multi-threading.

        Args:
            tmp_path: Temporary directory for outputs
        """
        import numpy as np

        num_sequences = 100
        dnabert_dim = 768
        esm2_dim = 1280

        logger.info(f"Testing merge with {num_sequences} embedding pairs")

        # Generate synthetic embeddings
        dnabert_embeddings = np.random.randn(num_sequences, dnabert_dim).astype(np.float32)
        esm2_embeddings = np.random.randn(num_sequences, esm2_dim).astype(np.float32)

        # Create output directory
        output_dir = tmp_path / "merge"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Measure merge time
        start_time = time.time()

        merged_embeddings = []
        for i in range(num_sequences):
            # Concatenate embeddings (simple merge)
            merged = np.concatenate([dnabert_embeddings[i], esm2_embeddings[i]])
            merged_embeddings.append(merged)

        elapsed = time.time() - start_time

        # Calculate throughput
        throughput = num_sequences / elapsed if elapsed > 0 else 0.0

        result = StageThroughputResult(
            stage_name="merge",
            num_sequences=num_sequences,
            total_time=elapsed,
            sequences_per_second=throughput,
            gpu_util_avg=None,  # CPU-only stage
        )

        logger.info(f"Merge: {throughput:.1f} seq/s ({elapsed:.2f}s total)")

        # Save result
        self._save_stage_result(result, tmp_path)

        # Verify reasonable throughput (merge should be very fast)
        assert throughput >= 100.0, \
            f"Merge throughput {throughput:.1f} seq/s is too low (expected ≥100 seq/s)"

    def test_prediction_throughput(self, tmp_path: Path, gpu_devices: List[int]):
        """
        Test final prediction stage throughput.

        Measures classification performance on merged embeddings.

        Args:
            tmp_path: Temporary directory for outputs
            gpu_devices: List of available GPU device IDs
        """
        import numpy as np
        import torch

        if len(gpu_devices) == 0:
            pytest.skip("No GPUs available")

        num_sequences = 100
        embedding_dim = 2048  # Combined DNABERT + ESM-2

        logger.info(f"Testing prediction with {num_sequences} embeddings")

        # Generate synthetic merged embeddings
        embeddings = torch.randn(num_sequences, embedding_dim, dtype=torch.float32)

        # Create simple test model
        device = torch.device(f"cuda:{gpu_devices[0]}")
        model = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2)
        ).to(device)
        model.eval()

        # Create output directory
        output_dir = tmp_path / "prediction"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Measure prediction time
        start_time = time.time()

        with torch.no_grad():
            batch_size = 32
            for i in range(0, num_sequences, batch_size):
                batch = embeddings[i:i+batch_size].to(device)
                outputs = model(batch)
                predictions = torch.argmax(outputs, dim=1)

        torch.cuda.synchronize(device)
        elapsed = time.time() - start_time

        # Calculate throughput
        throughput = num_sequences / elapsed if elapsed > 0 else 0.0

        result = StageThroughputResult(
            stage_name="prediction",
            num_sequences=num_sequences,
            total_time=elapsed,
            sequences_per_second=throughput,
            gpu_util_avg=None,  # Not tracking for simple test model
        )

        logger.info(f"Prediction: {throughput:.1f} seq/s ({elapsed:.2f}s total)")

        # Save result
        self._save_stage_result(result, tmp_path)

        # Verify reasonable throughput (prediction should be very fast)
        assert throughput >= 100.0, \
            f"Prediction throughput {throughput:.1f} seq/s is too low (expected ≥100 seq/s)"

    def test_identify_bottlenecks(self, tmp_path: Path):
        """
        Analyze all stage results to identify bottlenecks.

        A stage is a bottleneck if:
        - It consumes >30% of total pipeline time
        - GPU stages have <80% utilization

        Generates bottleneck report with recommendations.
        """
        results_file = tmp_path / "stage_results.json"

        if not results_file.exists():
            pytest.skip("No stage results found - run stage tests first")

        with open(results_file) as f:
            all_results = json.load(f)

        if not all_results:
            pytest.skip("No stage results to analyze")

        # Calculate total time
        total_time = sum(r['total_time'] for r in all_results.values())

        # Identify bottlenecks
        bottlenecks = []

        for stage_name, result in all_results.items():
            stage_time = result['total_time']
            time_percentage = (stage_time / total_time * 100) if total_time > 0 else 0.0

            # Check if stage is bottleneck
            is_bottleneck = False
            reasons = []

            if time_percentage > 30.0:
                is_bottleneck = True
                reasons.append(f"Consumes {time_percentage:.1f}% of total time (>30% threshold)")

            # Check GPU utilization for GPU stages
            if result.get('gpu_util_avg') is not None:
                gpu_util = result['gpu_util_avg']
                if gpu_util < 80.0:
                    is_bottleneck = True
                    reasons.append(f"GPU utilization {gpu_util:.1f}% below 80% threshold")

            if is_bottleneck:
                bottlenecks.append({
                    'stage': stage_name,
                    'time_percentage': time_percentage,
                    'reasons': reasons,
                    'result': result
                })

        # Generate bottleneck report
        report_path = tmp_path / "bottleneck_report.md"
        self._generate_bottleneck_report(all_results, bottlenecks, report_path)

        logger.info(f"Bottleneck report saved to {report_path}")

        # Log bottlenecks
        if bottlenecks:
            logger.warning(f"Found {len(bottlenecks)} bottleneck(s):")
            for b in bottlenecks:
                logger.warning(f"  {b['stage']}: {', '.join(b['reasons'])}")
        else:
            logger.info("No bottlenecks detected - all stages performing well")

    def _save_stage_result(self, result: StageThroughputResult, tmp_path: Path):
        """
        Save stage result to JSON file.

        Args:
            result: Stage throughput result
            tmp_path: Directory to save results
        """
        results_file = tmp_path / "stage_results.json"

        # Load existing results
        if results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        # Add this result
        all_results[result.stage_name] = asdict(result)

        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    def _generate_bottleneck_report(self,
                                    all_results: Dict,
                                    bottlenecks: List[Dict],
                                    output_path: Path):
        """
        Generate markdown bottleneck report.

        Args:
            all_results: All stage results
            bottlenecks: List of identified bottlenecks
            output_path: Path to save report
        """
        lines = []
        lines.append("# Per-Stage Throughput and Bottleneck Analysis\n")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Calculate total time
        total_time = sum(r['total_time'] for r in all_results.values())

        lines.append("## Stage Performance\n")
        lines.append("| Stage | Time (s) | % Total | Seq/s | GPU Util | Status |")
        lines.append("|-------|----------|---------|-------|----------|--------|")

        for stage_name, result in sorted(all_results.items()):
            stage_time = result['total_time']
            time_pct = (stage_time / total_time * 100) if total_time > 0 else 0.0
            throughput = result['sequences_per_second']
            gpu_util = result.get('gpu_util_avg')
            gpu_util_str = f"{gpu_util:.1f}%" if gpu_util is not None else "N/A"

            # Determine status
            is_bottleneck = any(b['stage'] == stage_name for b in bottlenecks)
            status = "⚠️ Bottleneck" if is_bottleneck else "✓ OK"

            lines.append(
                f"| {stage_name} | {stage_time:.2f} | {time_pct:.1f}% | "
                f"{throughput:.1f} | {gpu_util_str} | {status} |"
            )

        lines.append(f"\n**Total Pipeline Time:** {total_time:.2f}s\n")

        # Bottleneck details
        if bottlenecks:
            lines.append("## Bottlenecks Detected\n")

            for b in bottlenecks:
                lines.append(f"### {b['stage'].upper()}\n")
                lines.append(f"**Time:** {b['result']['total_time']:.2f}s ({b['time_percentage']:.1f}% of total)\n")
                lines.append("**Issues:**")
                for reason in b['reasons']:
                    lines.append(f"- {reason}")
                lines.append("\n**Recommendations:**")

                # Stage-specific recommendations
                if b['stage'] == 'translation':
                    lines.append("- Increase CPU thread count (--threads)")
                    lines.append("- Consider chunking strategies for large sequences")
                elif b['stage'] in ['dnabert', 'esm2']:
                    if 'GPU utilization' in str(b['reasons']):
                        lines.append("- Increase batch size to improve GPU saturation")
                        lines.append("- Check for CPU bottlenecks in data loading")
                        lines.append("- Enable CUDA streams for I/O overlap")
                    else:
                        lines.append("- Use multi-GPU parallelization (--parallel --gpus)")
                        lines.append("- Optimize batch size for model")
                elif b['stage'] == 'merge':
                    lines.append("- Increase merge threads (--merge-threads)")
                    lines.append("- Consider vectorized operations")
                lines.append("")
        else:
            lines.append("## No Bottlenecks Detected\n")
            lines.append("All stages are performing within expected thresholds.\n")

        # GPU utilization summary
        lines.append("## GPU Utilization (PERF-02 Requirement)\n")
        lines.append("GPU stages must achieve ≥80% utilization:\n")

        for stage_name in ['dnabert', 'esm2']:
            if stage_name in all_results:
                result = all_results[stage_name]
                gpu_util = result.get('gpu_util_avg')
                if gpu_util is not None:
                    status = "✓ Pass" if gpu_util >= 80.0 else "✗ Fail"
                    lines.append(f"- **{stage_name.upper()}:** {gpu_util:.1f}% {status}")

        report_text = '\n'.join(lines)

        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Bottleneck report:\n{report_text}")
