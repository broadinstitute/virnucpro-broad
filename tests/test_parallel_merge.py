"""Unit tests for parallel merge module"""

import pytest
import torch
import multiprocessing
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from virnucpro.pipeline.parallel_merge import (
    merge_file_pair_worker,
    merge_batch_worker,
    get_optimal_settings,
    parallel_merge_features,
    parallel_merge_with_progress
)


@pytest.fixture
def mock_nucleotide_features(tmp_path):
    """Create mock DNABERT-S feature files"""
    features = []
    for i in range(10):
        feature_file = tmp_path / f"file_{i}_DNABERT_S.pt"

        # Create mock DNABERT-S data (768-dim)
        nucleotide_ids = [f"seq_{i}_{j}" for j in range(5)]  # 5 sequences per file
        data = []
        for seq_id in nucleotide_ids:
            data.append({
                'label': seq_id,
                'mean_representation': torch.randn(768).tolist()
            })

        torch.save({'nucleotide': nucleotide_ids, 'data': data}, feature_file)
        features.append(feature_file)

    return features


@pytest.fixture
def mock_protein_features(tmp_path):
    """Create mock ESM-2 feature files"""
    features = []
    for i in range(10):
        feature_file = tmp_path / f"file_{i}_ESM2.pt"

        # Create mock ESM-2 data (2560-dim)
        protein_ids = [f"seq_{i}_{j}" for j in range(5)]  # 5 sequences per file
        data = [torch.randn(2560) for _ in protein_ids]

        torch.save({'proteins': protein_ids, 'data': data}, feature_file)
        features.append(feature_file)

    return features


@pytest.fixture
def output_dir(tmp_path):
    """Create output directory for merged features"""
    output = tmp_path / "merged"
    output.mkdir()
    return output


class TestMergeFilePairWorker:
    """Test merge_file_pair_worker function"""

    def test_worker_valid_file_pair(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test worker with valid file pair creates merged output"""
        nuc_file = mock_nucleotide_features[0]
        pro_file = mock_protein_features[0]
        output_file = output_dir / "file_0_merged.pt"

        file_pair = (nuc_file, pro_file, output_file)
        result = merge_file_pair_worker(file_pair)

        # Should return output path
        assert result == output_file

        # Verify output file exists
        assert output_file.exists()

        # Load and verify shape
        merged_data = torch.load(output_file)
        assert 'ids' in merged_data
        assert 'data' in merged_data
        assert merged_data['data'].shape[1] == 3328  # 768 + 2560

    def test_worker_missing_nucleotide_file(self, mock_protein_features, output_dir, tmp_path):
        """Test worker with missing nucleotide file returns None"""
        nuc_file = tmp_path / "missing_nuc.pt"
        pro_file = mock_protein_features[0]
        output_file = output_dir / "file_missing_merged.pt"

        file_pair = (nuc_file, pro_file, output_file)
        result = merge_file_pair_worker(file_pair)

        # Should return None on error
        assert result is None

        # Output file should not exist
        assert not output_file.exists()

    def test_worker_checkpoint_skip(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test worker skips if output already exists"""
        nuc_file = mock_nucleotide_features[0]
        pro_file = mock_protein_features[0]
        output_file = output_dir / "file_0_merged.pt"

        # Pre-create output file
        torch.save({'ids': ['seq_0_0'], 'data': torch.randn(1, 3328)}, output_file)
        original_mtime = output_file.stat().st_mtime

        file_pair = (nuc_file, pro_file, output_file)
        result = merge_file_pair_worker(file_pair)

        # Should still return output path
        assert result == output_file

        # File should not be rewritten (mtime unchanged)
        assert output_file.stat().st_mtime == original_mtime


class TestMergeBatchWorker:
    """Test merge_batch_worker function"""

    def test_batch_worker_multiple_pairs(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test batch worker processes multiple file pairs"""
        batch = []
        for i in range(3):
            nuc_file = mock_nucleotide_features[i]
            pro_file = mock_protein_features[i]
            output_file = output_dir / f"file_{i}_merged.pt"
            batch.append((nuc_file, pro_file, output_file))

        results = merge_batch_worker(batch)

        # Should return 3 results
        assert len(results) == 3

        # All should be valid paths
        for i, result in enumerate(results):
            assert result is not None
            assert result.exists()

            # Verify merged shape
            merged_data = torch.load(result)
            assert merged_data['data'].shape[1] == 3328

    def test_batch_worker_partial_failure(self, mock_nucleotide_features, mock_protein_features, output_dir, tmp_path):
        """Test batch worker handles partial failures"""
        batch = [
            (mock_nucleotide_features[0], mock_protein_features[0], output_dir / "file_0_merged.pt"),
            (tmp_path / "missing.pt", mock_protein_features[1], output_dir / "file_1_merged.pt"),  # Will fail
            (mock_nucleotide_features[2], mock_protein_features[2], output_dir / "file_2_merged.pt"),
        ]

        results = merge_batch_worker(batch)

        # Should return 3 results
        assert len(results) == 3

        # First and third should succeed, second should be None
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None


class TestGetOptimalSettings:
    """Test get_optimal_settings function"""

    def test_optimal_settings_defaults(self):
        """Test optimal settings with defaults"""
        workers, batch_size, chunksize = get_optimal_settings()

        assert workers == multiprocessing.cpu_count()
        assert batch_size == 1  # Each file pair is substantial work
        assert chunksize == 1  # Conservative default

    def test_optimal_settings_with_file_count(self):
        """Test optimal settings with known file count"""
        workers, batch_size, chunksize = get_optimal_settings(
            num_workers=8,
            num_file_pairs=100
        )

        assert workers == 8
        assert batch_size == 1
        # chunksize = max(1, 100 // (8 * 4)) = max(1, 3) = 3
        assert chunksize == 3

    def test_optimal_settings_small_workload(self):
        """Test optimal settings with small workload"""
        workers, batch_size, chunksize = get_optimal_settings(
            num_workers=8,
            num_file_pairs=10
        )

        assert workers == 8
        assert batch_size == 1
        # chunksize = max(1, 10 // (8 * 4)) = max(1, 0) = 1
        assert chunksize == 1


class TestParallelMergeFeatures:
    """Test parallel_merge_features function"""

    def test_parallel_merge_basic(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test basic parallel merge of multiple file pairs"""
        # Use first 3 files
        nuc_files = mock_nucleotide_features[:3]
        pro_files = mock_protein_features[:3]

        merged_files = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )

        # Should return 3 merged files
        assert len(merged_files) == 3

        # All should exist
        for merged_file in merged_files:
            assert merged_file.exists()

            # Verify merged shape
            merged_data = torch.load(merged_file)
            assert merged_data['data'].shape[1] == 3328

    def test_parallel_merge_output_naming(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test output file naming convention"""
        nuc_files = [mock_nucleotide_features[0]]
        pro_files = [mock_protein_features[0]]

        merged_files = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=1
        )

        # Should strip _DNABERT_S and add _merged
        assert merged_files[0].name == "file_0_merged.pt"

    @patch('virnucpro.pipeline.parallel_merge.multiprocessing.get_context')
    def test_parallel_merge_uses_spawn_context(self, mock_get_context, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test that parallel merge uses spawn context"""
        # Mock the context and pool
        mock_ctx = MagicMock()
        mock_pool = MagicMock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter([output_dir / "file_0_merged.pt"]))
        mock_ctx.Pool = Mock(return_value=mock_pool)
        mock_get_context.return_value = mock_ctx

        # Pre-create output file so worker doesn't fail
        output_file = output_dir / "file_0_merged.pt"
        torch.save({'ids': ['seq_0_0'], 'data': torch.randn(1, 3328)}, output_file)

        nuc_files = [mock_nucleotide_features[0]]
        pro_files = [mock_protein_features[0]]

        parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )

        # Verify spawn context requested
        mock_get_context.assert_called_once_with('spawn')


class TestParallelMergeWithProgress:
    """Test parallel_merge_with_progress function"""

    @patch('virnucpro.pipeline.parallel_merge.ProgressReporter')
    def test_merge_with_progress_reporting(self, mock_progress_class, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test merge with progress bar updates"""
        # Mock progress reporter
        mock_progress = MagicMock()
        mock_pbar = MagicMock()
        mock_pbar.__enter__ = Mock(return_value=mock_pbar)
        mock_pbar.__exit__ = Mock(return_value=False)
        mock_progress.create_file_bar = Mock(return_value=mock_pbar)
        mock_progress_class.return_value = mock_progress

        nuc_files = mock_nucleotide_features[:3]
        pro_files = mock_protein_features[:3]

        merged_files = parallel_merge_with_progress(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2,
            show_progress=True
        )

        # Should create progress bar with correct count
        mock_progress.create_file_bar.assert_called_once_with(3, desc="Merging features")

        # Should update progress bar 3 times (once per file)
        assert mock_pbar.update.call_count == 3

        # Should return 3 merged files
        assert len(merged_files) == 3

    @patch('virnucpro.pipeline.parallel_merge.ProgressReporter')
    def test_merge_progress_disabled(self, mock_progress_class, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test merge with progress reporting disabled"""
        mock_progress = MagicMock()
        mock_progress_class.return_value = mock_progress

        nuc_files = mock_nucleotide_features[:2]
        pro_files = mock_protein_features[:2]

        merged_files = parallel_merge_with_progress(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=1,
            show_progress=False
        )

        # ProgressReporter should be created with disable=True
        mock_progress_class.assert_called_once()
        call_kwargs = mock_progress_class.call_args[1]
        assert call_kwargs['disable'] is True

        # Should still return merged files
        assert len(merged_files) == 2

    def test_merge_progress_import_fallback(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test merge handles missing progress module gracefully"""
        # Patch to simulate import error
        with patch('virnucpro.pipeline.parallel_merge.ProgressReporter', side_effect=ImportError("Module not found")):
            nuc_files = mock_nucleotide_features[:2]
            pro_files = mock_protein_features[:2]

            # Should not raise error, just proceed without progress
            merged_files = parallel_merge_with_progress(
                nuc_files,
                pro_files,
                output_dir,
                num_workers=1,
                show_progress=True
            )

            # Should still return merged files
            assert len(merged_files) == 2


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""

    def test_sequential_vs_parallel_equivalence(self, mock_nucleotide_features, mock_protein_features, tmp_path):
        """Test parallel merge produces same results as sequential"""
        # Import sequential merge
        from virnucpro.pipeline.features import merge_features

        nuc_files = mock_nucleotide_features[:5]
        pro_files = mock_protein_features[:5]

        # Sequential merge
        sequential_dir = tmp_path / "sequential"
        sequential_dir.mkdir()
        sequential_results = []
        for nuc, pro in zip(nuc_files, pro_files):
            base_name = nuc.stem.replace('_DNABERT_S', '')
            output = sequential_dir / f"{base_name}_merged.pt"
            merge_features(nuc, pro, output)
            sequential_results.append(torch.load(output))

        # Parallel merge
        parallel_dir = tmp_path / "parallel"
        parallel_dir.mkdir()
        parallel_files = parallel_merge_features(
            nuc_files,
            pro_files,
            parallel_dir,
            num_workers=2
        )
        parallel_results = [torch.load(f) for f in parallel_files]

        # Compare results
        assert len(sequential_results) == len(parallel_results)
        for seq_res, par_res in zip(sequential_results, parallel_results):
            assert seq_res['ids'] == par_res['ids']
            assert torch.allclose(seq_res['data'], par_res['data'])

    def test_resume_capability(self, mock_nucleotide_features, mock_protein_features, output_dir):
        """Test checkpoint resume skips existing files"""
        nuc_files = mock_nucleotide_features[:5]
        pro_files = mock_protein_features[:5]

        # First run: merge first 3
        parallel_merge_features(
            nuc_files[:3],
            pro_files[:3],
            output_dir,
            num_workers=2
        )

        # Verify 3 files exist
        existing_files = list(output_dir.glob("*.pt"))
        assert len(existing_files) == 3

        # Second run: merge all 5 (should skip first 3)
        merged_files = parallel_merge_features(
            nuc_files,
            pro_files,
            output_dir,
            num_workers=2
        )

        # Should return all 5
        assert len(merged_files) == 5

        # All 5 should exist
        all_files = list(output_dir.glob("*.pt"))
        assert len(all_files) == 5
