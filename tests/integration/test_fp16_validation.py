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


class TestFP16PackedEquivalence:
    """Test FP16 vs FP32 in production forward_packed() code path."""

    def test_packed_inference_fp16_vs_fp32(self, esm_model_fp16, esm_model_fp32):
        """Verify FP16 forward_packed matches FP32 forward_packed."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        # Create sequences
        sequences = [
            ("seq1", "MKTAYIAKQRQISFV"),
            ("seq2", "VLSPADKTNVKAAWGKVG"),
            ("seq3", "MVHLTPEEKSAVTALWG")
        ]

        # Tokenize
        labels, strs, tokens = batch_converter(sequences)

        # Create packed batch manually
        # ESM uses 0 for padding, token IDs start at 4 (A), BOS=0, EOS=2
        # Extract actual tokens (excluding padding)
        input_ids_list = []
        for i in range(len(sequences)):
            # Find sequence length (tokens before padding/EOS)
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
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)

        max_seqlen = max(len(ids) for ids in input_ids_list)

        # Compare forward_packed() outputs
        with torch.no_grad():
            out_fp32 = model_fp32.forward_packed(
                input_ids=input_ids.to("cuda:0"),
                cu_seqlens=cu_seqlens.to("cuda:0"),
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
            out_fp16 = model_fp16.forward_packed(
                input_ids=input_ids.to("cuda:0"),
                cu_seqlens=cu_seqlens.to("cuda:0"),
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )

        # Compare embeddings (both should be [total_tokens, hidden_dim])
        emb_fp32 = out_fp32['representations'][36]
        emb_fp16 = out_fp16['representations'][36]

        # Compute per-token cosine similarity
        similarity = F.cosine_similarity(
            emb_fp32.float(),
            emb_fp16.float(),
            dim=-1
        )

        min_sim = similarity.min().item()
        mean_sim = similarity.mean().item()

        print(f"\nPacked inference FP16 vs FP32:")
        print(f"  Min token similarity: {min_sim:.6f}")
        print(f"  Mean token similarity: {mean_sim:.6f}")
        print(f"  Shape: {emb_fp16.shape}")

        assert mean_sim > 0.99, f"Packed FP16 mean similarity {mean_sim:.4f} < 0.99"
        assert min_sim > 0.95, f"Packed FP16 min similarity {min_sim:.4f} < 0.95 (some tokens very different)"

    def test_packed_long_sequences_fp16(self, esm_model_fp16, esm_model_fp32):
        """Test forward_packed with sequences >400aa (stresses RoPE and FlashAttention with FP16)."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        # Long sequences
        sequences = [
            ("long1", "VLSPADKTNVKAAWGKVGAHAG" * 20),  # 440 aa
            ("long2", "MKTAYIAKVLSPADKTNVKAAW" * 22),  # 484 aa
        ]

        # Tokenize
        labels, strs, tokens = batch_converter(sequences)

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
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_fp32 = model_fp32.forward_packed(
                input_ids=input_ids.to("cuda:0"),
                cu_seqlens=cu_seqlens.to("cuda:0"),
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
            out_fp16 = model_fp16.forward_packed(
                input_ids=input_ids.to("cuda:0"),
                cu_seqlens=cu_seqlens.to("cuda:0"),
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )

        emb_fp32 = out_fp32['representations'][36]
        emb_fp16 = out_fp16['representations'][36]

        similarity = F.cosine_similarity(emb_fp32.float(), emb_fp16.float(), dim=-1)
        mean_sim = similarity.mean().item()

        print(f"\nPacked long sequences FP16 validation:")
        print(f"  Sequence lengths: {[len(s[1]) for s in sequences]}")
        print(f"  Total tokens: {len(input_ids)}")
        print(f"  Mean similarity: {mean_sim:.6f}")

        assert mean_sim > 0.99, f"Long packed FP16 similarity {mean_sim:.4f} < 0.99"

    def test_packed_boundary_effects_fp16(self, esm_model_fp16, esm_model_fp32):
        """Test cu_seqlens boundaries don't cause FP16 precision issues at sequence transitions."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        # Multiple short sequences to test many boundaries
        sequences = [
            ("boundary1", "MKTAYIAK"),
            ("boundary2", "VLSPAD"),
            ("boundary3", "MVHLTPEEK"),
            ("boundary4", "AILVF"),
            ("boundary5", "DERKH"),
        ]

        # Tokenize
        labels, strs, tokens = batch_converter(sequences)

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
        cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32)
        cu_seqlens[0] = 0
        for i, ids in enumerate(input_ids_list):
            cu_seqlens[i + 1] = cu_seqlens[i] + len(ids)
        max_seqlen = max(len(ids) for ids in input_ids_list)

        with torch.no_grad():
            out_fp32 = model_fp32.forward_packed(
                input_ids=input_ids.to("cuda:0"),
                cu_seqlens=cu_seqlens.to("cuda:0"),
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )
            out_fp16 = model_fp16.forward_packed(
                input_ids=input_ids.to("cuda:0"),
                cu_seqlens=cu_seqlens.to("cuda:0"),
                max_seqlen=max_seqlen,
                repr_layers=[36]
            )

        emb_fp32 = out_fp32['representations'][36]
        emb_fp16 = out_fp16['representations'][36]

        # Check similarity near boundaries (tokens adjacent to cu_seqlens boundaries)
        boundary_sims = []
        for i in range(len(sequences)):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            # Check first and last token of each sequence
            if start < end:
                first_token_sim = F.cosine_similarity(
                    emb_fp32[start].float().unsqueeze(0),
                    emb_fp16[start].float().unsqueeze(0),
                    dim=-1
                ).item()
                boundary_sims.append(first_token_sim)

            if end - 1 >= start:
                last_token_sim = F.cosine_similarity(
                    emb_fp32[end - 1].float().unsqueeze(0),
                    emb_fp16[end - 1].float().unsqueeze(0),
                    dim=-1
                ).item()
                boundary_sims.append(last_token_sim)

        min_boundary_sim = min(boundary_sims)
        mean_boundary_sim = sum(boundary_sims) / len(boundary_sims)

        print(f"\nPacked boundary effects FP16 validation:")
        print(f"  Num boundaries: {len(sequences) - 1}")
        print(f"  Min boundary token similarity: {min_boundary_sim:.6f}")
        print(f"  Mean boundary token similarity: {mean_boundary_sim:.6f}")

        assert min_boundary_sim > 0.95, \
            f"Boundary token similarity {min_boundary_sim:.4f} < 0.95 (precision issue at boundaries)"
        assert mean_boundary_sim > 0.99, \
            f"Mean boundary similarity {mean_boundary_sim:.4f} < 0.99"


class TestFP16NumericalStability:
    """Test FP16 numerical stability (NaN/Inf detection)."""

    def test_no_nan_in_fp16_embeddings(self, esm_model_fp16):
        """Run 10 sequences through FP16 model, verify no NaN in output."""
        model_fp16, batch_converter = esm_model_fp16

        sequences = [
            (f"seq{i}", "MKTAYIAKVLSPADKTNV" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to("cuda:0")

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
        """Run 10 sequences through FP16 model, verify no Inf in output."""
        model_fp16, batch_converter = esm_model_fp16

        sequences = [
            (f"seq{i}", "VLSPADKTNVKAAWGKVG" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to("cuda:0")

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
        """Verify embedding L2 norms are finite and within reasonable range (e.g., 1-1000)."""
        model_fp16, batch_converter = esm_model_fp16

        sequences = [
            ("mag1", "MKTAYIAK"),
            ("mag2", "VLSPADKTNV" * 10),
            ("mag3", "MVHLTPEEK" * 20),
            ("mag4", "AILVFMWP" * 30),
            ("mag5", "DERKH" * 40),
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to("cuda:0")

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
        """Compare mean and std of FP16 vs FP32 embeddings with realistic thresholds for ESM-2."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

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
        tokens = tokens.to("cuda:0")

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
        """Compare L2 norm distributions. Mean norm relative difference should be <5%."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        sequences = [
            (f"norm{i}", "MKTAYIAKVLSPADKTNV" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to("cuda:0")

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
        """Count Z-score >3 outliers in both. Outlier count difference should be <10% of total elements."""
        model_fp16, batch_converter = esm_model_fp16
        model_fp32, _ = esm_model_fp32

        sequences = [
            (f"outlier{i}", "VLSPADKTNVKAAWGKVG" * (i + 1))
            for i in range(10)
        ]

        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to("cuda:0")

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
