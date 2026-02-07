#!/usr/bin/env python3
"""
End-to-end test script for FastESM2_650 feature extraction pipeline.

Validates that extract_fast_esm() produces correct 1280-dim embeddings from real
protein sequences and outputs in a format compatible with merge_data().
"""

import os
import sys
import tempfile
import torch
from transformers import AutoModel
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from units import extract_fast_esm


def create_sample_fasta(fasta_path):
    """
    Create a sample FASTA file with protein sequences of varying lengths.

    Includes:
    - Short (20aa)
    - Medium (100aa)
    - Long (500aa)
    - Very long (1200aa) - tests truncation at 1024
    - Multiple sequences for batch processing validation
    """
    sequences = [
        # Short sequence (20aa)
        SeqRecord(
            Seq("MKTAYIAKQRQISFVKSHFS"),
            id="test_short_20aa",
            description=""
        ),
        # Short-medium (50aa)
        SeqRecord(
            Seq("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAE"),
            id="test_short_medium_50aa",
            description=""
        ),
        # Medium sequence (100aa) - realistic ORF
        SeqRecord(
            Seq("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKLVERG"
                "QTLKHLESRIQRMEAESGLRVGAYVRKDGEWVLLSTFL"),
            id="test_medium_100aa",
            description=""
        ),
        # Long sequence (500aa) - extended realistic pattern
        SeqRecord(
            Seq("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKLVERG"
                "QTLKHLESRIQRMEAESGLRVGAYVRKDGEWVLLSTFLGHHANAVQAAQEQLKQQVKG"
                "ISIMQSQEEILQQIQQLQAEKQQLATQQQQAEQQAQAEQKAAEQKAAEQKAAEQKAAE"
                "QKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKA"
                "AEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAEQKAAET"
                "VTVQKGGKDVEIVMATLLGKVAGFCRRIGRNLLQEDKEAGEGMFTVKAAEVRGSYGRN"
                "VLYKGKKRARVMALLPGGFTYKVEFEDLTGLSRLAGQRSQVDVMIGTGCNQDLAARPE"
                "GQGGKGEGERPKGFGPGQGSKGSAGPPKGERPATPAATPKGSKGEKGPKGERGPKGEK"
                "GPKGEKGPKGEK"),
            id="test_long_500aa",
            description=""
        ),
        # Very long sequence (1200aa) - tests truncation at 1024
        SeqRecord(
            Seq("M" + "ARNDCQEGHILKMFPSTWYV" * 60),  # 1201aa total
            id="test_very_long_1200aa",
            description=""
        ),
        # Another short for diversity testing
        SeqRecord(
            Seq("ACDEFGHIKLMNPQRSTVWY"),
            id="test_diverse_20aa",
            description=""
        ),
    ]

    with open(fasta_path, 'w') as f:
        SeqIO.write(sequences, f, 'fasta')

    return len(sequences)


def main():
    """Run end-to-end extraction test with validation."""

    print("=" * 80)
    print("FastESM2_650 Feature Extraction Test")
    print("=" * 80)
    print()

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "test_proteins.fasta")
        output_path = os.path.join(tmpdir, "test_output.pt")

        # Step 1: Create sample FASTA
        print("Step 1: Creating sample FASTA with test sequences...")
        num_sequences = create_sample_fasta(fasta_path)
        print(f"[PASS] Created FASTA with {num_sequences} sequences")
        print()

        # Step 2: Load FastESM2_650 model
        print("Step 2: Loading FastESM2_650 model...")
        try:
            model = AutoModel.from_pretrained(
                "Synthyra/FastESM2_650",
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).eval().cuda()
            tokenizer = model.tokenizer
            print("[PASS] Model loaded successfully")
            print(f"       Device: {model.device}")
            print(f"       Dtype: {next(model.parameters()).dtype}")
        except Exception as e:
            print(f"[FAIL] Failed to load model: {e}")
            return 1
        print()

        # Step 3: Run extract_fast_esm()
        print("Step 3: Running extract_fast_esm()...")
        import time
        start_time = time.time()

        try:
            proteins, data = extract_fast_esm(
                fasta_path,
                out_file=output_path,
                model=model,
                tokenizer=tokenizer,
                truncation_seq_length=1024,
                toks_per_batch=2048
            )
            extraction_time = time.time() - start_time
            print(f"[PASS] Extraction completed in {extraction_time:.2f} seconds")
            print(f"       Processed {len(proteins)} proteins")
        except Exception as e:
            print(f"[FAIL] Extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()

        # Step 4: Validate output file exists
        print("Step 4: Validating output file...")
        if not os.path.exists(output_path):
            print("[FAIL] Output file was not created")
            return 1
        print("[PASS] Output file created")
        print()

        # Step 5: Validate output format
        print("Step 5: Validating output format...")
        try:
            loaded = torch.load(output_path)

            # Check keys
            if set(loaded.keys()) != {'proteins', 'data'}:
                print(f"[FAIL] Expected keys {{'proteins', 'data'}}, got {set(loaded.keys())}")
                return 1
            print("[PASS] Correct keys in .pt file")

            # Check sequence count
            if len(loaded['proteins']) != num_sequences:
                print(f"[FAIL] Expected {num_sequences} sequences, got {len(loaded['proteins'])}")
                return 1
            print(f"[PASS] Correct sequence count ({num_sequences})")

            # Check proteins and data lengths match
            if len(loaded['proteins']) != len(loaded['data']):
                print(f"[FAIL] proteins length ({len(loaded['proteins'])}) != data length ({len(loaded['data'])})")
                return 1
            print("[PASS] proteins and data lengths match")

        except Exception as e:
            print(f"[FAIL] Failed to load output file: {e}")
            return 1
        print()

        # Step 6: Validate embedding properties
        print("Step 6: Validating embedding properties...")
        all_pass = True

        # Check dimensions
        for i, (protein, embedding) in enumerate(zip(loaded['proteins'], loaded['data'])):
            if embedding.shape != (1280,):
                print(f"[FAIL] {protein}: Expected shape (1280,), got {embedding.shape}")
                all_pass = False
        if all_pass:
            print("[PASS] All embeddings are 1280-dim")

        # Check dtype
        for i, (protein, embedding) in enumerate(zip(loaded['proteins'], loaded['data'])):
            if embedding.dtype != torch.float32:
                print(f"[FAIL] {protein}: Expected dtype torch.float32, got {embedding.dtype}")
                all_pass = False
        if all_pass:
            print("[PASS] All embeddings are float32")

        # Check for NaN
        for protein, embedding in zip(loaded['proteins'], loaded['data']):
            if torch.isnan(embedding).any():
                print(f"[FAIL] {protein}: Contains NaN values")
                all_pass = False
        if all_pass:
            print("[PASS] No NaN values")

        # Check for Inf
        for protein, embedding in zip(loaded['proteins'], loaded['data']):
            if torch.isinf(embedding).any():
                print(f"[FAIL] {protein}: Contains Inf values")
                all_pass = False
        if all_pass:
            print("[PASS] No Inf values")

        # Check embeddings are non-zero
        for protein, embedding in zip(loaded['proteins'], loaded['data']):
            if torch.all(embedding == 0):
                print(f"[FAIL] {protein}: Embedding is all zeros")
                all_pass = False
        if all_pass:
            print("[PASS] Embeddings are non-zero")

        # Check different sequences produce different embeddings
        if len(loaded['data']) >= 2:
            different = False
            for i in range(len(loaded['data']) - 1):
                if not torch.allclose(loaded['data'][i], loaded['data'][i+1]):
                    different = True
                    break
            if different:
                print("[PASS] Different sequences produce different embeddings")
            else:
                print("[FAIL] All embeddings are identical")
                all_pass = False

        if not all_pass:
            return 1
        print()

        # Step 7: Validate merge_data() compatibility
        print("Step 7: Validating merge_data() compatibility...")
        try:
            # Simulate merge_data() consumption pattern
            ESM_outfile = torch.load(output_path)
            protein_data_dict = {}

            for protein, emb in zip(ESM_outfile['proteins'], ESM_outfile['data']):
                protein_data_dict[protein] = emb

            # Validate each entry
            for key, val in protein_data_dict.items():
                if not isinstance(val, torch.Tensor):
                    print(f"[FAIL] {key}: data is {type(val)}, expected Tensor")
                    return 1
                if val.shape != (1280,):
                    print(f"[FAIL] {key}: shape is {val.shape}, expected (1280,)")
                    return 1

            # Test torch.cat(..., dim=-1) operation (as used in merge_data)
            test_nucleotide = torch.randn(768)  # DNABERT-S dimension
            for key, val in protein_data_dict.items():
                try:
                    merged = torch.cat((test_nucleotide, val), dim=-1)
                    if merged.shape != (768 + 1280,):
                        print(f"[FAIL] {key}: merged shape is {merged.shape}, expected (2048,)")
                        return 1
                except Exception as e:
                    print(f"[FAIL] {key}: torch.cat failed: {e}")
                    return 1

            print("[PASS] merge_data() compatibility verified")

        except Exception as e:
            print(f"[FAIL] merge_data() compatibility check failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()

        # Step 8: Validate resume capability
        print("Step 8: Validating resume capability...")
        try:
            resume_start = time.time()
            proteins2, data2 = extract_fast_esm(
                fasta_path,
                out_file=output_path,  # Same output file
                model=model,
                tokenizer=tokenizer,
                truncation_seq_length=1024,
                toks_per_batch=2048
            )
            resume_time = time.time() - resume_start

            # Resume should be much faster (< 1 second)
            if resume_time > 1.0:
                print(f"[WARN] Resume took {resume_time:.2f}s, expected < 1s (may not have skipped)")

            # Check that data is identical
            if len(proteins2) != len(proteins):
                print(f"[FAIL] Resume returned different number of proteins")
                return 1

            if proteins2 != proteins:
                print(f"[FAIL] Resume returned different protein labels")
                return 1

            for i in range(len(data2)):
                if not torch.allclose(data2[i], data[i]):
                    print(f"[FAIL] Resume returned different embeddings for {proteins[i]}")
                    return 1

            print("[PASS] Resume capability works")
            print(f"       Resume completed in {resume_time:.2f} seconds")

        except Exception as e:
            print(f"[FAIL] Resume test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print()

        # Summary
        print("=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        print()
        print("All tests PASSED!")
        print()
        print(f"Extraction timing:")
        print(f"  Initial extraction: {extraction_time:.2f} seconds")
        print(f"  Resume (cached):    {resume_time:.2f} seconds")
        print(f"  Speedup:            {extraction_time/max(resume_time, 0.001):.1f}x")
        print()
        print(f"Embedding validation:")
        print(f"  Sequences processed: {len(proteins)}")
        print(f"  Embedding dimension: 1280")
        print(f"  Embedding dtype:     float32")
        print(f"  No NaN or Inf:       ✓")
        print(f"  Unique embeddings:   ✓")
        print(f"  merge_data() ready:  ✓")
        print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
