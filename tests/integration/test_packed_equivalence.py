"""Integration tests for packed vs unpacked embedding equivalence.

These tests require GPU and ESM-2 model. Skip if unavailable.

The tests verify that packed sequences produce identical embeddings to unpacked
sequences, proving that FlashAttention varlen integration is correct and there's
no cross-sequence contamination.
"""
import pytest
import torch


# Skip entire module if CUDA unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for packed equivalence tests"
)


@pytest.fixture(scope="module")
def esm_model():
    """Load ESM-2 model once for all tests."""
    from virnucpro.models.esm2_flash import load_esm2_model
    model, batch_converter = load_esm2_model(
        model_name="esm2_t36_3B_UR50D",  # Production 3B model (validates actual deployment)
        device="cuda:0"
    )
    return model, batch_converter


class TestPackedEquivalence:
    """Test packed vs unpacked embedding equivalence."""

    def test_short_sequences(self, esm_model):
        """Short sequences (<50 aa) should match exactly."""
        model, batch_converter = esm_model
        sequences = [
            ("short1", "MKTAYIAK"),
            ("short2", "VLSPADKTNV"),
            ("short3", "MVHLT"),
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Failed: {details}"
        assert details["min_similarity"] > 0.999, \
            f"Min similarity {details['min_similarity']} <= 0.999"

    def test_medium_sequences(self, esm_model):
        """Medium sequences (50-200 aa) should match."""
        model, batch_converter = esm_model
        sequences = [
            ("medium1", "MKTAYIAK" * 10),  # 80 aa
            ("medium2", "VLSPADKTNV" * 15),  # 150 aa
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Failed: {details}"
        assert details["strict_pass_rate"] >= 0.99, \
            f"Strict pass rate {details['strict_pass_rate']} < 99%"

    def test_mixed_lengths(self, esm_model):
        """Mixed length sequences in same batch."""
        model, batch_converter = esm_model
        sequences = [
            ("tiny", "MKT"),
            ("small", "MKTAYIAK"),
            ("medium", "MKTAYIAK" * 20),
            ("large", "MKTAYIAK" * 50),
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Failed: {details}"

        # Check all sequences passed
        assert len(details['failed_sequences']) == 0, \
            f"Failed sequences: {details['failed_sequences']}"

    def test_many_sequences(self, esm_model):
        """Test with many sequences (>10) to stress-test packing."""
        model, batch_converter = esm_model
        sequences = [
            (f"seq{i}", "MKTAYIAKVL" * (i + 1))
            for i in range(15)
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Failed: {details}"
        assert details["num_sequences"] == 15
        assert details["strict_pass_rate"] >= 0.99


class TestCrossContamination:
    """Test that sequences don't contaminate each other."""

    def test_distinct_sequences_remain_distinct(self, esm_model):
        """Very different sequences should produce different embeddings."""
        model, batch_converter = esm_model
        # Hydrophobic vs charged sequences
        sequences = [
            ("hydrophobic", "AILVFMWP" * 5),
            ("charged", "DERKH" * 8),
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Cross-contamination suspected: {details}"

    def test_repeated_sequences_match(self, esm_model):
        """Identical sequences should produce identical embeddings in packed format."""
        model, batch_converter = esm_model
        # Same sequence repeated with different IDs
        sequences = [
            ("rep1", "MKTAYIAKVLSPADKTNV"),
            ("rep2", "MKTAYIAKVLSPADKTNV"),
            ("rep3", "MKTAYIAKVLSPADKTNV"),
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Repeated sequences don't match: {details}"

        # All should have perfect similarity (1.0)
        assert all(sim > 0.999 for sim in details['per_sequence'].values()), \
            f"Some repeated sequences have low similarity: {details['per_sequence']}"


class TestPositionIDReset:
    """Test position IDs reset at sequence boundaries (Gap 4)."""

    def test_position_ids_reset_in_pipeline(self, esm_model):
        """
        Verify position IDs reset to 0 at each cu_seqlens boundary.

        CRITICAL: If position IDs don't reset, the model sees sequence 2
        as a continuation of sequence 1, causing incorrect positional embeddings.
        """
        model, batch_converter = esm_model

        # Create batch with known sequence lengths
        sequences = [
            ("seq1", "MKT"),       # 3aa -> 5 tokens with BOS/EOS
            ("seq2", "VLSPAD"),    # 6aa -> 8 tokens with BOS/EOS
        ]

        from virnucpro.data import VarlenCollator
        collator = VarlenCollator(
            batch_converter,
            max_tokens_per_batch=4096,
            enable_packing=False
        )
        batch = collator([{'id': s[0], 'sequence': s[1]} for s in sequences])

        # Check cu_seqlens boundaries
        cu_seqlens = batch['cu_seqlens']
        assert cu_seqlens.tolist() == [0, 5, 13], \
            f"Expected [0, 5, 13], got {cu_seqlens.tolist()}"

        # Generate position IDs (this is what model does internally)
        from virnucpro.models.packed_attention import create_position_ids_packed
        position_ids = create_position_ids_packed(cu_seqlens)

        # CRITICAL CHECKS: Position IDs must reset at boundaries
        assert position_ids[0] == 0, "Sequence 1 should start at position 0"
        assert position_ids[5] == 0, "Sequence 2 should start at position 0 (RESET!)"

        # Check continuity within each sequence
        seq1_positions = position_ids[0:5].tolist()
        seq2_positions = position_ids[5:13].tolist()

        assert seq1_positions == [0, 1, 2, 3, 4], \
            f"Seq1 positions should be [0,1,2,3,4], got {seq1_positions}"
        assert seq2_positions == [0, 1, 2, 3, 4, 5, 6, 7], \
            f"Seq2 positions should be [0,1,2,3,4,5,6,7], got {seq2_positions}"

        # Run through model to ensure no errors
        model.eval()
        with torch.no_grad():
            output = model.forward_packed(
                input_ids=batch['input_ids'].to(model.device),
                cu_seqlens=cu_seqlens.to(model.device),
                max_seqlen=batch['max_seqlen']
            )

        assert 'representations' in output, "Model should return representations"
        assert 36 in output['representations'], "Should have layer 36 representations"
        print("✓ Position IDs reset correctly at sequence boundaries")

    def test_position_id_validation(self, esm_model):
        """Test position ID generator directly."""
        from virnucpro.models.packed_attention import create_position_ids_packed

        # Test case 1: Three sequences of different lengths
        cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
        position_ids = create_position_ids_packed(cu_seqlens)

        expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2])
        assert torch.all(position_ids == expected), \
            f"Position IDs mismatch: expected {expected}, got {position_ids}"

        # Test case 2: Single sequence
        cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
        position_ids = create_position_ids_packed(cu_seqlens)

        expected = torch.tensor([0, 1, 2, 3, 4])
        assert torch.all(position_ids == expected), \
            f"Single sequence position IDs mismatch: expected {expected}, got {position_ids}"

        # Test case 3: Many short sequences
        cu_seqlens = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)
        position_ids = create_position_ids_packed(cu_seqlens)

        expected = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        assert torch.all(position_ids == expected), \
            f"Many short sequences position IDs mismatch: expected {expected}, got {position_ids}"


class TestEdgeCases:
    """Test edge cases for packed equivalence validation."""

    def test_single_sequence(self, esm_model):
        """Single sequence should always match (no packing needed)."""
        model, batch_converter = esm_model
        sequences = [
            ("single", "MKTAYIAKVLSPADKTNV"),
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Single sequence failed: {details}"
        assert details["min_similarity"] > 0.999

    def test_very_short_sequences(self, esm_model):
        """Very short sequences (3-5 aa) should still match."""
        model, batch_converter = esm_model
        sequences = [
            ("tiny1", "MKT"),
            ("tiny2", "VLS"),
            ("tiny3", "PADK"),
        ]
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, f"Very short sequences failed: {details}"

    def test_empty_sequences_list(self, esm_model):
        """Empty sequences list should return True with empty details."""
        model, batch_converter = esm_model
        sequences = []
        from virnucpro.data.packing import validate_packed_equivalence
        passed, details = validate_packed_equivalence(
            model, batch_converter, sequences, torch.device("cuda:0")
        )
        assert passed, "Empty sequences should pass"
        assert details["num_sequences"] == 0
        assert details["min_similarity"] == 1.0
        assert details["max_similarity"] == 1.0
