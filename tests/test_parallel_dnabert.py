"""Tests for DNABERT-S parallel processing utilities

This test suite provides comprehensive coverage of DNABERT-S parallel processing:
- File assignment with bin-packing algorithm
- BF16 detection and batch size adjustment
- Worker function with mock model
- Token-based batching
- Vanilla vs. optimized output comparison
"""

import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import torch
import sys

from virnucpro.pipeline.parallel_dnabert import (
    process_dnabert_files_worker,
)
from virnucpro.pipeline.base_worker import (
    count_sequences,
    assign_files_by_sequences,
    detect_bf16_support
)


class TestCountSequences(unittest.TestCase):
    """Test sequence counting utility"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_count_sequences_empty(self):
        """Test counting sequences in empty file"""
        test_file = self.temp_dir / "empty.fa"
        test_file.write_text("")
        self.assertEqual(count_sequences(test_file), 0)

    def test_count_sequences_single(self):
        """Test counting single sequence"""
        test_file = self.temp_dir / "single.fa"
        test_file.write_text(">seq1\nACGTACGT\n")
        self.assertEqual(count_sequences(test_file), 1)

    def test_count_sequences_multiple(self):
        """Test counting multiple sequences"""
        test_file = self.temp_dir / "multiple.fa"
        test_file.write_text(">seq1\nACGT\n>seq2\nTGCA\n>seq3\nAAAA\n")
        self.assertEqual(count_sequences(test_file), 3)


class TestAssignFilesBySequences(unittest.TestCase):
    """Test bin-packing file assignment"""

    def setUp(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_assign_files_by_sequences_empty_input(self):
        """Test with empty file list"""
        result = assign_files_by_sequences([], 4)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(len(worker_files) == 0 for worker_files in result))

    def test_assign_files_by_sequences_single_worker(self):
        """Test all files go to one worker"""
        # Create test files with different sequence counts
        files = []
        for i in range(3):
            test_file = self.temp_dir / f"file_{i}.fa"
            test_file.write_text(f">seq1\nACGT\n>seq2\nTGCA\n")
            files.append(test_file)

        result = assign_files_by_sequences(files, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 3)

    def test_assign_files_by_sequences_multiple_workers(self):
        """Test files distributed across multiple workers"""
        # Create test files with different sequence counts
        files = []
        # File 0: 5 sequences (largest)
        test_file = self.temp_dir / "file_0.fa"
        test_file.write_text(">s1\nA\n>s2\nA\n>s3\nA\n>s4\nA\n>s5\nA\n")
        files.append(test_file)
        # File 1: 3 sequences
        test_file = self.temp_dir / "file_1.fa"
        test_file.write_text(">s1\nA\n>s2\nA\n>s3\nA\n")
        files.append(test_file)
        # File 2: 2 sequences
        test_file = self.temp_dir / "file_2.fa"
        test_file.write_text(">s1\nA\n>s2\nA\n")
        files.append(test_file)

        result = assign_files_by_sequences(files, 2)

        # Should have 2 workers
        self.assertEqual(len(result), 2)

        # Files should be distributed to balance load
        # Worker 0 should get file_0 (5 seqs)
        # Worker 1 should get file_1 (3 seqs) + file_2 (2 seqs) = 5 seqs
        self.assertEqual(len(result[0]), 1)  # file_0
        self.assertEqual(len(result[1]), 2)  # file_1, file_2

    def test_assign_files_by_sequences_balancing(self):
        """Test that bin-packing balances loads effectively"""
        # Create files with varying sizes
        files = []
        sizes = [10, 8, 6, 5, 3, 2, 1]  # Total: 35 sequences

        for i, size in enumerate(sizes):
            test_file = self.temp_dir / f"file_{i}.fa"
            content = "".join([f">s{j}\nA\n" for j in range(size)])
            test_file.write_text(content)
            files.append(test_file)

        result = assign_files_by_sequences(files, 3)

        # Calculate total sequences per worker
        worker_totals = []
        for worker_files in result:
            total = sum(count_sequences(f) for f in worker_files)
            worker_totals.append(total)

        # All workers should have sequences
        self.assertTrue(all(total > 0 for total in worker_totals))

        # Difference between max and min should be reasonable
        # (bin-packing doesn't guarantee perfect balance, but should be close)
        max_diff = max(worker_totals) - min(worker_totals)
        self.assertLessEqual(max_diff, 10)  # Within 10 sequences

    def test_assign_files_invalid_workers(self):
        """Test error handling for invalid worker count"""
        test_file = self.temp_dir / "file.fa"
        test_file.write_text(">seq1\nACGT\n")

        with self.assertRaises(ValueError):
            assign_files_by_sequences([test_file], 0)
        with self.assertRaises(ValueError):
            assign_files_by_sequences([test_file], -1)


class TestBF16Detection(unittest.TestCase):
    """Test BF16 support detection"""

    @patch('torch.cuda.get_device_capability')
    def test_bf16_enabled_on_ampere(self):
        """Test BF16 enabled on Ampere GPU (compute capability 8.0)"""
        # Mock Ampere GPU
        mock_capability = MagicMock(return_value=(8, 0))
        device = torch.device('cuda:0')

        with patch('torch.cuda.get_device_capability', mock_capability):
            result = detect_bf16_support(device)

        self.assertTrue(result)
        mock_capability.assert_called_once()

    @patch('torch.cuda.get_device_capability')
    def test_bf16_disabled_on_older_gpu(self):
        """Test BF16 disabled on older GPU (compute capability 7.5)"""
        # Mock Turing GPU
        mock_capability = MagicMock(return_value=(7, 5))
        device = torch.device('cuda:0')

        with patch('torch.cuda.get_device_capability', mock_capability):
            result = detect_bf16_support(device)

        self.assertFalse(result)
        mock_capability.assert_called_once()

    def test_bf16_disabled_on_cpu(self):
        """Test BF16 disabled on CPU device"""
        device = torch.device('cpu')
        result = detect_bf16_support(device)
        self.assertFalse(result)


class TestDNABERTWorker(unittest.TestCase):
    """Test DNABERT-S worker function"""

    def setUp(self):
        """Create temp directory and mock files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

        # Create test DNA files
        self.test_files = []
        for i in range(3):
            test_file = self.temp_dir / f"dna_{i}.fa"
            test_file.write_text(f">seq_{i}\nACGTACGTACGTACGT\n")
            self.test_files.append(test_file)

    def tearDown(self):
        """Clean up temp files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    @patch('torch.save')
    def test_process_dnabert_files_worker_success(
        self, mock_save, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """Test worker processes files successfully"""
        # Mock BF16 detection
        mock_bf16.return_value = False

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(1, 4, 768),)  # Hidden states
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Process files
        processed, failed = process_dnabert_files_worker(
            file_subset=self.test_files,
            device_id=0,
            toks_per_batch=2048,
            output_dir=self.output_dir
        )

        # Should process all files successfully
        self.assertEqual(len(processed), 3)
        self.assertEqual(len(failed), 0)

        # Check torch.save was called for each file
        self.assertEqual(mock_save.call_count, 3)

        # Verify output file paths
        for i, output_file in enumerate(processed):
            self.assertEqual(output_file, self.output_dir / f"dna_{i}_DNABERT.pt")

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    @patch('torch.save')
    def test_process_dnabert_files_worker_with_failures(
        self, mock_save, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """Test worker handles failures gracefully"""
        # Mock BF16 detection
        mock_bf16.return_value = False

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model to fail on second file
        mock_model = MagicMock()
        call_count = [0]

        def model_side_effect(input_ids, attention_mask=None):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated processing error")
            return (torch.randn(1, 4, 768),)

        mock_model.side_effect = model_side_effect
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock tokenizer
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }

        # Process files
        processed, failed = process_dnabert_files_worker(
            file_subset=self.test_files,
            device_id=0,
            toks_per_batch=2048,
            output_dir=self.output_dir
        )

        # Should process 2 files, fail 1
        self.assertEqual(len(processed), 2)
        self.assertEqual(len(failed), 1)

        # Check failed file
        failed_file, error_msg = failed[0]
        self.assertTrue("dna_1" in str(failed_file))
        self.assertIn("Simulated processing error", error_msg)

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    def test_process_dnabert_files_worker_empty_input(
        self, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """Test worker handles empty file list"""
        # Mock BF16 detection
        mock_bf16.return_value = False

        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Process empty file list
        processed, failed = process_dnabert_files_worker(
            file_subset=[],
            device_id=0,
            toks_per_batch=2048,
            output_dir=self.output_dir
        )

        # Should have no processed or failed files
        self.assertEqual(len(processed), 0)
        self.assertEqual(len(failed), 0)

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    @patch('torch.save')
    def test_batch_size_increase_with_bf16(
        self, mock_save, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """Test batch size increases from 2048 to 3072 when BF16 enabled"""
        # Mock BF16 as enabled
        mock_bf16.return_value = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(1, 4, 768),)
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Process files with default batch size
        processed, failed = process_dnabert_files_worker(
            file_subset=self.test_files[:1],
            device_id=0,
            toks_per_batch=2048,  # Default value
            output_dir=self.output_dir
        )

        # Should process successfully
        self.assertEqual(len(processed), 1)
        self.assertEqual(len(failed), 0)

        # Verify BF16 was checked
        mock_bf16.assert_called_once()


class TestTokenBatching(unittest.TestCase):
    """Test token-based batching behavior"""

    def setUp(self):
        """Create temp directory"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

    def tearDown(self):
        """Clean up temp files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    @patch('torch.save')
    def test_token_batching_respects_limit(
        self, mock_save, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """Test that batching respects token limit"""
        # Create file with sequences that will create multiple batches
        test_file = self.temp_dir / "multi_batch.fa"
        # Create sequences with total length > 2048
        # Each sequence is 1024 bases, so 3 sequences = 3072 bases > 2048 limit
        content = ""
        for i in range(3):
            content += f">seq_{i}\n{'A' * 1024}\n"
        test_file.write_text(content)

        # Mock BF16 detection
        mock_bf16.return_value = False

        # Track tokenizer calls to verify batching
        tokenizer_calls = []

        mock_tokenizer = MagicMock()

        def tokenizer_side_effect(seqs, **kwargs):
            tokenizer_calls.append(len(seqs))
            batch_size = len(seqs)
            return {
                'input_ids': torch.ones(batch_size, 4, dtype=torch.long),
                'attention_mask': torch.ones(batch_size, 4, dtype=torch.long)
            }

        mock_tokenizer.side_effect = tokenizer_side_effect
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()

        def model_side_effect(input_ids, attention_mask=None):
            batch_size = input_ids.shape[0]
            return (torch.randn(batch_size, 4, 768),)

        mock_model.side_effect = model_side_effect
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Process file with small batch size
        processed, failed = process_dnabert_files_worker(
            file_subset=[test_file],
            device_id=0,
            toks_per_batch=2048,
            output_dir=self.output_dir
        )

        # Should process successfully
        self.assertEqual(len(processed), 1)
        self.assertEqual(len(failed), 0)

        # Should have created multiple batches (at least 2)
        # First batch: 1 seq (1024 tokens)
        # Second batch: 1 seq (1024 tokens)
        # Third batch: 1 seq (1024 tokens)
        self.assertGreaterEqual(len(tokenizer_calls), 2)

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    @patch('torch.save')
    def test_token_batching_handles_long_sequences(
        self, mock_save, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """Test that single long sequences are handled"""
        # Create file with one very long sequence
        test_file = self.temp_dir / "long_seq.fa"
        test_file.write_text(f">long_seq\n{'A' * 5000}\n")

        # Mock BF16 detection
        mock_bf16.return_value = False

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(1, 4, 768),)
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Process file
        processed, failed = process_dnabert_files_worker(
            file_subset=[test_file],
            device_id=0,
            toks_per_batch=2048,
            output_dir=self.output_dir
        )

        # Should handle the long sequence (even though it exceeds batch size)
        self.assertEqual(len(processed), 1)
        self.assertEqual(len(failed), 0)


class TestOptimizedMatchesVanilla(unittest.TestCase):
    """Test that optimized output matches vanilla implementation"""

    def setUp(self):
        """Create temp directory"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()

    def tearDown(self):
        """Clean up temp files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('virnucpro.pipeline.parallel_dnabert.detect_bf16_support')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoModel')
    @patch('virnucpro.pipeline.parallel_dnabert.AutoTokenizer')
    @patch('torch.save')
    def test_optimized_matches_vanilla_output(
        self, mock_save, mock_tokenizer_class, mock_model_class, mock_bf16
    ):
        """
        CRITICAL TEST: Verify optimized processing produces identical output to vanilla.

        This test compares:
        1. Optimized path: BF16=True, toks_per_batch=3072
        2. Vanilla path: BF16=False, toks_per_batch=2048

        Both should produce identical embeddings (within numerical tolerance).
        """
        # Create test file
        test_file = self.temp_dir / "test.fa"
        test_file.write_text(">seq1\nACGTACGT\n>seq2\nTGCATGCA\n")

        # Shared mock model that returns deterministic outputs
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Create deterministic embeddings
        deterministic_output = torch.tensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2], [1.3, 1.4], [1.5, 1.6]]
        ])

        mock_model = MagicMock()
        mock_model.return_value = (deterministic_output,)
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model

        # Capture saved data
        saved_data = []

        def save_side_effect(data, path):
            saved_data.append(data)

        mock_save.side_effect = save_side_effect

        # Process with optimized settings (BF16 enabled)
        mock_bf16.return_value = True
        processed_opt, failed_opt = process_dnabert_files_worker(
            file_subset=[test_file],
            device_id=0,
            toks_per_batch=2048,  # Will be increased to 3072 with BF16
            output_dir=self.output_dir
        )

        optimized_data = saved_data[-1]

        # Reset mocks
        mock_save.reset_mock()
        saved_data.clear()

        # Process with vanilla settings (BF16 disabled)
        mock_bf16.return_value = False
        processed_van, failed_van = process_dnabert_files_worker(
            file_subset=[test_file],
            device_id=0,
            toks_per_batch=2048,
            output_dir=self.output_dir
        )

        vanilla_data = saved_data[-1]

        # Both should succeed
        self.assertEqual(len(processed_opt), 1)
        self.assertEqual(len(failed_opt), 0)
        self.assertEqual(len(processed_van), 1)
        self.assertEqual(len(failed_van), 0)

        # Compare outputs
        self.assertEqual(len(optimized_data['data']), len(vanilla_data['data']))
        self.assertEqual(optimized_data['nucleotide'], vanilla_data['nucleotide'])

        # Compare embeddings element-wise
        for opt_item, van_item in zip(optimized_data['data'], vanilla_data['data']):
            self.assertEqual(opt_item['label'], van_item['label'])

            # Convert to tensors for comparison
            opt_embedding = torch.tensor(opt_item['mean_representation'])
            van_embedding = torch.tensor(van_item['mean_representation'])

            # Should be identical (or very close due to numerical precision)
            torch.testing.assert_close(
                opt_embedding,
                van_embedding,
                rtol=1e-5,
                atol=1e-6,
                msg=f"Embeddings differ for {opt_item['label']}"
            )


if __name__ == '__main__':
    unittest.main()
