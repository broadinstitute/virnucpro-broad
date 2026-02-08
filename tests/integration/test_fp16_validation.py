"""Integration tests for FP16 vs FP32 embedding equivalence.

These tests require GPU and ESM-2 model. They validate that FP16 precision
produces embeddings matching FP32 baseline (cosine similarity >0.99).

This is the one-time thorough validation proving FP16 works for VirNucPro.
"""

import gc
import pytest
import torch
import torch.nn.functional as F


# Skip entire module if CUDA unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FP16 validation tests"
)


@pytest.fixture(scope="function")
def esm_model_fp32():
    """Load ESM-2 in FP32 (baseline). Function-scoped for memory cleanup."""
    from virnucpro.models.esm2_flash import load_esm2_model

    model, batch_converter = load_esm2_model(
        model_name="esm2_t36_3B_UR50D",
        device="cuda:0",
        enable_fp16=False
    )

    yield model, batch_converter

    # Explicit cleanup to free ~22GB before next model loads
    del model
    del batch_converter
    torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="function")
def esm_model_fp16():
    """Load ESM-2 in FP16. Function-scoped for memory cleanup."""
    from virnucpro.models.esm2_flash import load_esm2_model

    # Ensure previous model fully cleaned up
    torch.cuda.empty_cache()

    model, batch_converter = load_esm2_model(
        model_name="esm2_t36_3B_UR50D",
        device="cuda:0",
        enable_fp16=True
    )

    yield model, batch_converter

    # Cleanup
    del model
    del batch_converter
    torch.cuda.empty_cache()
    gc.collect()


def compare_embeddings(model_fp16, model_fp32, batch_converter, sequences, device, threshold=0.99):
    """Compare FP16 vs FP32 embeddings for a list of sequences."""
    labels, strs, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    with torch.no_grad():
        # FP32 baseline
        out_fp32 = model_fp32(tokens, repr_layers=[36])
        emb_fp32 = out_fp32["representations"][36]

        # FP16
        out_fp16 = model_fp16(tokens, repr_layers=[36])
        emb_fp16 = out_fp16["representations"][36]

    similarities = {}
    for i, (seq_id, seq_str) in enumerate(sequences):
        seq_len = min(len(seq_str), 1024)
        # Mean pool (skip BOS token at position 0)
        e32 = emb_fp32[i, 1:seq_len+1].mean(dim=0).float()
        e16 = emb_fp16[i, 1:seq_len+1].mean(dim=0).float()
        sim = torch.nn.functional.cosine_similarity(e32, e16, dim=0).item()
        similarities[seq_id] = sim

    min_sim = min(similarities.values())
    mean_sim = sum(similarities.values()) / len(similarities)
    failed = [sid for sid, s in similarities.items() if s < threshold]

    return {
        "passed": len(failed) == 0,
        "min_similarity": min_sim,
        "mean_similarity": mean_sim,
        "per_sequence": similarities,
        "failed_sequences": failed,
    }


class TestFP16Equivalence:
    """Test FP16 vs FP32 embedding equivalence across various sequence lengths."""

    def test_short_sequences_fp16_match(self, esm_model_fp16, esm_model_fp32):
        """Short sequences (<50 aa) should match with >0.99 cosine similarity."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        sequences = [
            ("short1", "MKTAYIAK"),
            ("short2", "VLSPADKTNV"),
            ("short3", "MVHLTPEEK"),
            ("short4", "AILVFMWP"),
            ("short5", "DERKH"),
        ]

        result = compare_embeddings(
            model_fp16, model_fp32, batch_converter, sequences,
            torch.device("cuda:0"), threshold=0.99
        )

        print(f"\nShort sequences FP16 validation:")
        print(f"  Min similarity: {result['min_similarity']:.6f}")
        print(f"  Mean similarity: {result['mean_similarity']:.6f}")
        for seq_id, sim in result['per_sequence'].items():
            print(f"    {seq_id}: {sim:.6f}")

        assert result['passed'], f"Failed sequences: {result['failed_sequences']}"
        assert result['min_similarity'] > 0.99, \
            f"Min similarity {result['min_similarity']:.6f} <= 0.99"

    def test_medium_sequences_fp16_match(self, esm_model_fp16, esm_model_fp32):
        """Medium sequences (50-200 aa) should match with >0.99 cosine similarity."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        sequences = [
            ("medium1", "MKTAYIAK" * 10),  # 80 aa
            ("medium2", "VLSPADKTNV" * 15),  # 150 aa
            ("medium3", "MVHLTPEEKSAVTAL" * 10),  # 150 aa
        ]

        result = compare_embeddings(
            model_fp16, model_fp32, batch_converter, sequences,
            torch.device("cuda:0"), threshold=0.99
        )

        print(f"\nMedium sequences FP16 validation:")
        print(f"  Min similarity: {result['min_similarity']:.6f}")
        print(f"  Mean similarity: {result['mean_similarity']:.6f}")
        for seq_id, sim in result['per_sequence'].items():
            print(f"    {seq_id}: {sim:.6f}")

        assert result['passed'], f"Failed sequences: {result['failed_sequences']}"
        assert result['min_similarity'] > 0.99, \
            f"Min similarity {result['min_similarity']:.6f} <= 0.99"

    def test_long_sequences_fp16_match(self, esm_model_fp16, esm_model_fp32):
        """Long sequences (200-500 aa) should match with >0.99 cosine similarity."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        sequences = [
            ("long1", "VLSPADKTNVKAAWGKVG" * 20),  # 360 aa
            ("long2", "MKTAYIAKVLSPADKTNVKAAW" * 20),  # 440 aa
        ]

        result = compare_embeddings(
            model_fp16, model_fp32, batch_converter, sequences,
            torch.device("cuda:0"), threshold=0.99
        )

        print(f"\nLong sequences FP16 validation:")
        print(f"  Min similarity: {result['min_similarity']:.6f}")
        print(f"  Mean similarity: {result['mean_similarity']:.6f}")
        for seq_id, sim in result['per_sequence'].items():
            print(f"    {seq_id}: {sim:.6f}")

        assert result['passed'], f"Failed sequences: {result['failed_sequences']}"
        assert result['min_similarity'] > 0.99, \
            f"Min similarity {result['min_similarity']:.6f} <= 0.99"

    def test_mixed_lengths_fp16_match(self, esm_model_fp16, esm_model_fp32):
        """Mixed length sequences should match with >0.99 cosine similarity."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        sequences = [
            ("tiny", "MKT"),
            ("small", "MKTAYIAK"),
            ("medium1", "VLSPADKTNV" * 10),  # 100 aa
            ("medium2", "MVHLTPEEK" * 20),  # 180 aa
            ("large1", "AILVFMWP" * 30),  # 240 aa
            ("large2", "DERKH" * 50),  # 250 aa
            ("large3", "MKTAYIAKVLSPADKTNV" * 25),  # 450 aa
        ]

        result = compare_embeddings(
            model_fp16, model_fp32, batch_converter, sequences,
            torch.device("cuda:0"), threshold=0.99
        )

        print(f"\nMixed length sequences FP16 validation:")
        print(f"  Min similarity: {result['min_similarity']:.6f}")
        print(f"  Mean similarity: {result['mean_similarity']:.6f}")
        for seq_id, sim in result['per_sequence'].items():
            seq = next(s[1] for s in sequences if s[0] == seq_id)
            print(f"    {seq_id} ({len(seq)}aa): {sim:.6f}")

        assert result['passed'], f"Failed sequences: {result['failed_sequences']}"
        assert result['min_similarity'] > 0.99, \
            f"Min similarity {result['min_similarity']:.6f} <= 0.99"

    def test_packed_fp16_vs_fp32_unpacked(self, esm_model_fp16, esm_model_fp32):
        """Validate FP16 packed maintains FP32-level quality (cosine similarity >0.99).

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32
        device = model_fp16.device

        sequences = [
            ("seq1", "MKTAYIAKQRQISFV"),
            ("seq2", "VLSPADKTNVKAAWGKVG"),
            ("seq3", "MVHLTPEEKSAVTALWG"),
            ("seq4", "MKTAYIAK" * 10),
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp32 = model_fp32(tokens, repr_layers=[36])
        emb_fp32 = out_fp32["representations"][36]

        input_ids_list = []
        for i in range(len(sequences)):
            seq_tokens = tokens[i]
            eos_idx = (seq_tokens == 2).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                end_idx = eos_idx[0].item() + 1
            else:
                end_idx = len(seq_tokens)
            input_ids_list.append(seq_tokens[:end_idx])

        input_ids = torch.cat(input_ids_list)
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_packed = model_fp16.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
        emb_packed = out_packed['representations'][36]

        packed_offset = 0
        similarities = []
        for i, seq_len in enumerate([len(ids) for ids in input_ids_list]):
            emb_f = emb_fp32[i, 1:seq_len].float()
            emb_p = emb_packed[packed_offset+1:packed_offset+seq_len].float()
            sim = F.cosine_similarity(emb_f, emb_p, dim=-1).mean().item()
            similarities.append(sim)
            packed_offset += seq_len

        mean_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)

        print(f"\nFP16 packed vs FP32 unpacked:")
        print(f"  Mean sequence similarity: {mean_sim:.6f}")
        print(f"  Min sequence similarity: {min_sim:.6f}")
        print(f"  Per-sequence: {[f'{s:.6f}' for s in similarities]}")

        assert mean_sim > 0.99, f"FP16 packed vs FP32 unpacked mean similarity {mean_sim:.4f} < 0.99"
        assert min_sim > 0.98, f"FP16 packed vs FP32 unpacked min similarity {min_sim:.4f} < 0.98"


class TestFP16PackedEquivalence:
    """Test FP16 forward_packed() implementation correctness.

    These tests validate that forward_packed() produces the same results as
    standard forward() when both use FP16. This verifies the packing implementation
    itself (RoPE timing, FlashAttention varlen, boundary handling), not FP16 precision.

    Note: FP32 + forward_packed() is illegal (FlashAttention requires FP16/BF16),
    so we compare FP16 packed vs FP16 unpacked. FP16 vs FP32 equivalence is already
    validated in TestFP16Equivalence.
    """

    def test_packed_inference_fp16_correctness(self, esm_model_fp16):
        """Verify FP16 forward_packed matches FP16 standard forward (validates packing implementation).

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        # Create sequences
        sequences = [
            ("seq1", "MKTAYIAKQRQISFV"),
            ("seq2", "VLSPADKTNVKAAWGKVG"),
            ("seq3", "MVHLTPEEKSAVTALWG")
        ]

        # Tokenize
        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        # === Standard forward (unpacked) ===
        with torch.no_grad():
            out_unpacked = model_fp16(tokens, repr_layers=[36])
        emb_unpacked = out_unpacked['representations'][36]  # [batch, max_len, hidden]

        # === Packed forward ===
        # Extract actual tokens (excluding padding)
        input_ids_list = []
        for i in range(len(sequences)):
            seq_tokens = tokens[i]
            # Find EOS token (value 2)
            eos_idx = (seq_tokens == 2).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                end_idx = eos_idx[0].item() + 1  # Include EOS
            else:
                end_idx = len(seq_tokens)
            input_ids_list.append(seq_tokens[:end_idx])

        # Concatenate into 1D packed format
        input_ids = torch.cat(input_ids_list)

        # Create cu_seqlens
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_packed = model_fp16.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
        emb_packed = out_packed['representations'][36]  # [total_tokens, hidden]

        # Compare embeddings per sequence
        packed_offset = 0
        similarities = []
        for i, seq_len in enumerate([len(ids) for ids in input_ids_list]):
            # Unpacked: [batch_idx, 1:seq_len] (skip BOS at position 0)
            emb_u = emb_unpacked[i, 1:seq_len].float()
            # Packed: [packed_offset+1:packed_offset+seq_len] (skip BOS)
            emb_p = emb_packed[packed_offset+1:packed_offset+seq_len].float()

            # Compute per-token similarity
            sim = F.cosine_similarity(emb_u, emb_p, dim=-1).mean().item()
            similarities.append(sim)
            packed_offset += seq_len

        mean_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)

        print(f"\nPacked vs Unpacked (FP16):")
        print(f"  Mean sequence similarity: {mean_sim:.6f}")
        print(f"  Min sequence similarity: {min_sim:.6f}")
        print(f"  Per-sequence: {[f'{s:.6f}' for s in similarities]}")

        assert mean_sim > 0.99, f"Packed vs unpacked mean similarity {mean_sim:.4f} < 0.99"
        assert min_sim > 0.95, f"Packed vs unpacked min similarity {min_sim:.4f} < 0.95"

    def test_packed_long_sequences_fp16(self, esm_model_fp16):
        """Test forward_packed with sequences >400aa (stresses RoPE and FlashAttention with FP16).

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        # Long sequences
        sequences = [
            ("long1", "VLSPADKTNVKAAWGKVGAHAG" * 20),  # 440 aa
            ("long2", "MKTAYIAKVLSPADKTNVKAAW" * 22),  # 484 aa
        ]

        # Tokenize and run standard forward
        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_unpacked = model_fp16(tokens, repr_layers=[36])
        emb_unpacked = out_unpacked['representations'][36]

        # Create packed batch
        input_ids_list = []
        for i in range(len(sequences)):
            seq_tokens = tokens[i]
            eos_idx = (seq_tokens == 2).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                end_idx = eos_idx[0].item() + 1
            else:
                end_idx = len(seq_tokens)
            input_ids_list.append(seq_tokens[:end_idx])

        input_ids = torch.cat(input_ids_list)
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_packed = model_fp16.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
        emb_packed = out_packed['representations'][36]

        # Compare per sequence
        packed_offset = 0
        similarities = []
        for i, seq_len in enumerate([len(ids) for ids in input_ids_list]):
            emb_u = emb_unpacked[i, 1:seq_len].float()
            emb_p = emb_packed[packed_offset+1:packed_offset+seq_len].float()
            sim = F.cosine_similarity(emb_u, emb_p, dim=-1).mean().item()
            similarities.append(sim)
            packed_offset += seq_len

        mean_sim = sum(similarities) / len(similarities)

        print(f"\nPacked long sequences validation:")
        print(f"  Sequence lengths: {[len(s[1]) for s in sequences]}")
        print(f"  Total tokens: {len(input_ids)}")
        print(f"  Mean similarity: {mean_sim:.6f}")

        assert mean_sim > 0.99, f"Long packed similarity {mean_sim:.4f} < 0.99"

    def test_packed_boundary_effects_fp16(self, esm_model_fp16):
        """Test cu_seqlens boundaries don't cause precision issues at sequence transitions.

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        # Multiple short sequences to test many boundaries
        sequences = [
            ("boundary1", "MKTAYIAK"),
            ("boundary2", "VLSPAD"),
            ("boundary3", "MVHLTPEEK"),
            ("boundary4", "AILVF"),
            ("boundary5", "DERKH"),
        ]

        # Tokenize and run standard forward
        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_unpacked = model_fp16(tokens, repr_layers=[36])
        emb_unpacked = out_unpacked['representations'][36]

        # Create packed batch
        input_ids_list = []
        for i in range(len(sequences)):
            seq_tokens = tokens[i]
            eos_idx = (seq_tokens == 2).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                end_idx = eos_idx[0].item() + 1
            else:
                end_idx = len(seq_tokens)
            input_ids_list.append(seq_tokens[:end_idx])

        input_ids = torch.cat(input_ids_list)
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_packed = model_fp16.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
        emb_packed = out_packed['representations'][36]

        # Check similarity near boundaries (tokens adjacent to cu_seqlens boundaries)
        boundary_sims = []
        packed_offset = 0
        for i, seq_len in enumerate([len(ids) for ids in input_ids_list]):
            # Compare boundary tokens (first and last of each sequence)
            emb_u_first = emb_unpacked[i, 1].float()  # First token (skip BOS)
            emb_p_first = emb_packed[packed_offset+1].float()
            sim_first = F.cosine_similarity(emb_u_first.unsqueeze(0), emb_p_first.unsqueeze(0), dim=-1).item()

            if seq_len > 2:  # Has tokens beyond BOS/EOS
                emb_u_last = emb_unpacked[i, seq_len-1].float()  # Last token before EOS
                emb_p_last = emb_packed[packed_offset+seq_len-1].float()
                sim_last = F.cosine_similarity(emb_u_last.unsqueeze(0), emb_p_last.unsqueeze(0), dim=-1).item()
                boundary_sims.extend([sim_first, sim_last])
            else:
                boundary_sims.append(sim_first)

            packed_offset += seq_len

        mean_boundary_sim = sum(boundary_sims) / len(boundary_sims)
        min_boundary_sim = min(boundary_sims)

        print(f"\nBoundary effects test:")
        print(f"  Sequences: {len(sequences)}")
        print(f"  Boundary tokens checked: {len(boundary_sims)}")
        print(f"  Mean boundary similarity: {mean_boundary_sim:.6f}")
        print(f"  Min boundary similarity: {min_boundary_sim:.6f}")

        assert mean_boundary_sim > 0.99, f"Boundary mean similarity {mean_boundary_sim:.4f} < 0.99"
        assert min_boundary_sim > 0.95, f"Boundary min similarity {min_boundary_sim:.4f} < 0.95"

    def test_packed_fp16_precision_alignment(self, esm_model_fp16):
        """Validate packed forward path precision matches standard ESM-2 forward for FP16.

        This test verifies the precision alignment fix that:
        1. Uses esm.modules.gelu instead of torch.nn.functional.gelu for consistent FP16 behavior
        2. Computes RoPE using position_ids.to(inv_freq.dtype) matching ESM-2's RotaryEmbedding

        The test compares packed forward output to standard forward output with tight
        tolerances (cosine similarity >0.999) to catch precision regressions.

        Note: This test requires script mode execution for proper tensor handling.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        sequences = [
            ("align1", "MKTAYIAKQRQISFVKS"),
            ("align2", "VLSPADKTNVKAAWGKVGAH"),
            ("align3", "MVHLTPEEKSAVTALWGKV"),
            ("align4", "DERKH" * 5),
            ("align5", "AILVFMWP" * 4),
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_unpacked = model_fp16(tokens, repr_layers=[36])
        emb_unpacked = out_unpacked['representations'][36]

        input_ids_list = []
        for i in range(len(sequences)):
            seq_tokens = tokens[i]
            eos_idx = (seq_tokens == 2).nonzero(as_tuple=True)[0]
            if len(eos_idx) > 0:
                end_idx = eos_idx[0].item() + 1
            else:
                end_idx = len(seq_tokens)
            input_ids_list.append(seq_tokens[:end_idx])

        input_ids = torch.cat(input_ids_list)
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_packed = model_fp16.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
        emb_packed = out_packed['representations'][36]

        packed_offset = 0
        similarities = []
        max_abs_diffs = []
        for i, seq_len in enumerate([len(ids) for ids in input_ids_list]):
            emb_u = emb_unpacked[i, 1:seq_len]
            emb_p = emb_packed[packed_offset+1:packed_offset+seq_len]

            sim = F.cosine_similarity(emb_u.float(), emb_p.float(), dim=-1).mean().item()
            similarities.append(sim)

            abs_diff = (emb_u.float() - emb_p.float()).abs()
            max_abs_diff = abs_diff.max().item()
            max_abs_diffs.append(max_abs_diff)

            packed_offset += seq_len

        mean_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_diff = max(max_abs_diffs)

        print(f"\nFP16 Precision Alignment Validation:")
        print(f"  Mean cosine similarity: {mean_sim:.6f}")
        print(f"  Min cosine similarity: {min_sim:.6f}")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Per-sequence similarities: {[f'{s:.6f}' for s in similarities]}")

        assert mean_sim > 0.999, f"Mean similarity {mean_sim:.6f} < 0.999 (precision alignment issue)"
        assert min_sim > 0.995, f"Min similarity {min_sim:.6f} < 0.995 (precision alignment issue)"
        assert max_diff < 1e-3, f"Max absolute difference {max_diff:.6e} >= 1e-3 (precision regression)"



class TestFP16NumericalStability:
    """Test FP16 numerical stability (NaN/Inf detection)."""

    def test_no_nan_in_fp16_embeddings(self, esm_model_fp16):
        """Run 10 sequences through FP16 model, verify no NaN in output.

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        sequences = [
            (f"seq{i}", "MKTAYIAKVLSPADKTNV" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp16 = model_fp16(tokens, repr_layers=[36])
            emb_fp16 = out_fp16["representations"][36]

        has_nan = torch.isnan(emb_fp16).any().item()

        print(f"\nFP16 NaN detection:")
        print(f"  Sequences tested: {len(sequences)}")
        print(f"  Embedding shape: {emb_fp16.shape}")
        print(f"  Has NaN: {has_nan}")

        assert not has_nan, "FP16 embeddings contain NaN values"

    def test_no_inf_in_fp16_embeddings(self, esm_model_fp16):
        """Run 10 sequences through FP16 model, verify no Inf in output.

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        sequences = [
            (f"seq{i}", "VLSPADKTNVKAAWGKVG" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp16 = model_fp16(tokens, repr_layers=[36])
            emb_fp16 = out_fp16["representations"][36]

        has_inf = torch.isinf(emb_fp16).any().item()

        print(f"\nFP16 Inf detection:")
        print(f"  Sequences tested: {len(sequences)}")
        print(f"  Embedding shape: {emb_fp16.shape}")
        print(f"  Has Inf: {has_inf}")

        assert not has_inf, "FP16 embeddings contain Inf values"

    def test_fp16_embedding_magnitude_reasonable(self, esm_model_fp16):
        """Verify embedding L2 norms are finite and within reasonable range (e.g., 1-1000).

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        device = model_fp16.device

        sequences = [
            ("mag1", "MKTAYIAK"),
            ("mag2", "VLSPADKTNV" * 10),
            ("mag3", "MVHLTPEEK" * 20),
            ("mag4", "AILVFMWP" * 30),
            ("mag5", "DERKH" * 40),
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp16 = model_fp16(tokens, repr_layers=[36])
            emb_fp16 = out_fp16["representations"][36]

        # Compute L2 norms for each sequence (mean-pooled embeddings)
        norms = []
        for i in range(len(sequences)):
            seq_len = min(len(sequences[i][1]), 1024)
            embedding = emb_fp16[i, 1:seq_len+1].mean(dim=0).float()
            norm = torch.norm(embedding, p=2).item()
            norms.append(norm)

        min_norm = min(norms)
        max_norm = max(norms)
        mean_norm = sum(norms) / len(norms)

        print(f"\nFP16 embedding magnitude validation:")
        print(f"  Min L2 norm: {min_norm:.4f}")
        print(f"  Max L2 norm: {max_norm:.4f}")
        print(f"  Mean L2 norm: {mean_norm:.4f}")

        # ESM-2 embeddings typically have L2 norms in range ~10-100
        assert min_norm > 1, f"Min norm {min_norm:.4f} suspiciously small (<1)"
        assert max_norm < 1000, f"Max norm {max_norm:.4f} suspiciously large (>1000)"
        assert all(n > 0 and n < float('inf') for n in norms), "Some norms are not finite"


class TestFP16StatisticalValidation:
    """Validate embedding distributions match statistically (beyond cosine similarity)."""

    def test_mean_std_similar(self, esm_model_fp16, esm_model_fp32):
        """Compare mean and std of FP16 vs FP32 embeddings with realistic thresholds for ESM-2.

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32
        device = model_fp16.device

        # Run inference on 10 sequences (mix of short/medium/long)
        sequences = [
            ("stat1", "MKTAYIAK"),
            ("stat2", "VLSPADKTNV"),
            ("stat3", "MVHLTPEEK" * 5),
            ("stat4", "AILVFMWP" * 10),
            ("stat5", "DERKH" * 15),
            ("stat6", "MKTAYIAKVLS" * 20),
            ("stat7", "VLSPADKTNVKAAW" * 25),
            ("stat8", "MVHLT" * 40),
            ("stat9", "AILVFMWPGK" * 30),
            ("stat10", "DERKHAAEFG" * 35),
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp32 = model_fp32(tokens, repr_layers=[36])
            emb_fp32 = out_fp32["representations"][36]

            out_fp16 = model_fp16(tokens, repr_layers=[36])
            emb_fp16 = out_fp16["representations"][36]

        # ESM-2 embeddings: mean ≈ 0.0, std ≈ 0.5-1.5
        fp32_mean = emb_fp32.mean().item()
        fp16_mean = emb_fp16.float().mean().item()
        mean_abs_diff = abs(fp32_mean - fp16_mean)

        # Realistic threshold: absolute difference <0.01 (not 0.1)
        # FP16 mantissa is 10 bits, expect ~1e-3 precision
        assert mean_abs_diff < 0.01, f"Mean diff {mean_abs_diff:.4f} exceeds 0.01"

        fp32_std = emb_fp32.std().item()
        fp16_std = emb_fp16.float().std().item()
        std_rel_diff = abs(fp32_std - fp16_std) / fp32_std

        # Relative threshold: <5% difference (not absolute 0.1)
        assert std_rel_diff < 0.05, f"Std relative diff {std_rel_diff:.4f} exceeds 5%"

        print(f"\nFP16 vs FP32 statistical comparison:")
        print(f"  Mean: FP32={fp32_mean:.6f}, FP16={fp16_mean:.6f}, diff={mean_abs_diff:.6f}")
        print(f"  Std: FP32={fp32_std:.6f}, FP16={fp16_std:.6f}, rel_diff={std_rel_diff:.4f}")

    def test_l2_norm_distribution_similar(self, esm_model_fp16, esm_model_fp32):
        """Compare L2 norm distributions. Mean norm relative difference should be <5%.

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32
        device = model_fp16.device

        sequences = [
            (f"norm{i}", "MKTAYIAKVLSPADKTNV" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp32 = model_fp32(tokens, repr_layers=[36])
            emb_fp32 = out_fp32["representations"][36]

            out_fp16 = model_fp16(tokens, repr_layers=[36])
            emb_fp16 = out_fp16["representations"][36]

        # Compute L2 norms for each sequence
        norms_fp32 = []
        norms_fp16 = []
        for i in range(len(sequences)):
            seq_len = min(len(sequences[i][1]), 1024)

            e32 = emb_fp32[i, 1:seq_len+1].mean(dim=0).float()
            norm32 = torch.norm(e32, p=2).item()
            norms_fp32.append(norm32)

            e16 = emb_fp16[i, 1:seq_len+1].mean(dim=0).float()
            norm16 = torch.norm(e16, p=2).item()
            norms_fp16.append(norm16)

        mean_norm32 = sum(norms_fp32) / len(norms_fp32)
        mean_norm16 = sum(norms_fp16) / len(norms_fp16)
        rel_diff = abs(mean_norm32 - mean_norm16) / mean_norm32

        print(f"\nL2 norm distribution comparison:")
        print(f"  Mean FP32 norm: {mean_norm32:.4f}")
        print(f"  Mean FP16 norm: {mean_norm16:.4f}")
        print(f"  Relative difference: {rel_diff:.4f}")

        assert rel_diff < 0.05, f"L2 norm mean relative diff {rel_diff:.4f} exceeds 5%"

    def test_outlier_count_similar(self, esm_model_fp16, esm_model_fp32):
        """Count Z-score >3 outliers in both. Outlier count difference should be <10% of total elements.

        Note: This test requires script mode execution (not notebook environments)
        for proper tensor handling and CUDA stream synchronization.
        """
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32
        device = model_fp16.device

        sequences = [
            (f"outlier{i}", "VLSPADKTNVKAAWGKVG" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            out_fp32 = model_fp32(tokens, repr_layers=[36])
            emb_fp32 = out_fp32["representations"][36]

            out_fp16 = model_fp16(tokens, repr_layers=[36])
            emb_fp16 = out_fp16["representations"][36]

        # Compute Z-scores
        mean32 = emb_fp32.mean()
        std32 = emb_fp32.std()
        z_scores32 = (emb_fp32 - mean32) / std32
        outliers32 = (z_scores32.abs() > 3).sum().item()

        mean16 = emb_fp16.float().mean()
        std16 = emb_fp16.float().std()
        z_scores16 = (emb_fp16.float() - mean16) / std16
        outliers16 = (z_scores16.abs() > 3).sum().item()

        total_elements = emb_fp32.numel()
        outlier_diff = abs(outliers32 - outliers16)
        outlier_diff_pct = (outlier_diff / total_elements) * 100

        print(f"\nOutlier count comparison:")
        print(f"  Total elements: {total_elements}")
        print(f"  FP32 outliers (Z>3): {outliers32} ({(outliers32/total_elements)*100:.4f}%)")
        print(f"  FP16 outliers (Z>3): {outliers16} ({(outliers16/total_elements)*100:.4f}%)")
        print(f"  Difference: {outlier_diff} ({outlier_diff_pct:.4f}%)")

        # Outlier count difference should be <10% of total elements
        assert outlier_diff_pct < 10, \
            f"Outlier count difference {outlier_diff_pct:.4f}% exceeds 10% of total elements"
