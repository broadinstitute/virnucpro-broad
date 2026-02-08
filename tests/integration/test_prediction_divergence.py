"""Integration test for prediction-level divergence between v1.0 and v2.0 attention.

This test measures the downstream impact of FlashAttention vs standard attention
divergence by comparing classification labels and confidence scores.

Purpose:
    The embedding-level divergence (70.3% match at 1e-3, 96% at 1e-2) is alarming
    but may not affect final predictions. Small embedding differences can cancel out
    in the downstream MLP classifier. Before implementing any fix, we need to measure
    what actually matters: do v1.0 and v2.0 produce the same viral/non-viral
    classification labels?

Architecture:
    - v1.0 path: model.forward() uses standard ESM-2 attention (torch.bmm with cuBLAS,
      FP16 accumulation on Ampere+)
    - v2.0 path: model.forward_packed() uses FlashAttention varlen (FP32 accumulation)

Decision Tree (from 10.2-RESEARCH.md):
    - >99% label agreement: ACCEPT v2.0 - divergence is cosmetic
    - 95-99%: VALIDATE on production dataset before accepting
    - <95%: IMPLEMENT v1.0-compatible attention fallback
"""

import pytest
import torch
import torch.nn.functional as F
import logging
import os
from typing import List, Tuple, Dict, Any

logger = logging.getLogger('virnucpro.tests.integration.prediction_divergence')

# Skip entire module if CUDA unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for prediction divergence tests"
)


@pytest.fixture(scope="module")
def esm_model():
    """Load ESM-2 3B model in FP16 (production configuration)."""
    from virnucpro.models.esm2_flash import load_esm2_model

    logger.info("Loading ESM-2 3B model in FP16...")
    model, batch_converter = load_esm2_model(
        model_name="esm2_t36_3B_UR50D",
        device="cuda:0",
        enable_fp16=True  # Production configuration
    )
    logger.info(f"Model loaded: {model}")
    return model, batch_converter


@pytest.fixture(scope="module")
def test_sequences():
    """Create deterministic test sequences with varying lengths.

    Uses repeating amino acid patterns (not random) for reproducibility.
    Lengths range from 10 to 400 aa to cover typical viral protein range.
    """
    sequences = [
        # Short sequences (10-50 aa)
        ("short_01", "MKTAYIAKVL"),  # 10 aa
        ("short_02", "VLSPADKTNVKAAW" * 2),  # 28 aa
        ("short_03", "MKTAYIAKVLSPAD" * 3),  # 42 aa

        # Medium sequences (50-150 aa)
        ("medium_01", "MKTAYIAKVL" * 8),  # 80 aa
        ("medium_02", "VLSPADKTNV" * 10),  # 100 aa
        ("medium_03", "MKTAYIAKVLSPADKTNV" * 7),  # 126 aa

        # Medium-long sequences (150-250 aa)
        ("medlong_01", "MKTAYIAKVL" * 18),  # 180 aa
        ("medlong_02", "VLSPADKTNVKAAWGKV" * 12),  # 204 aa
        ("medlong_03", "MKTAYIAKVLSPADKTNV" * 13),  # 234 aa

        # Long sequences (250-400 aa)
        ("long_01", "MKTAYIAKVL" * 28),  # 280 aa
        ("long_02", "VLSPADKTNV" * 32),  # 320 aa
        ("long_03", "MKTAYIAKVLSPADKTNV" * 20),  # 360 aa

        # Very diverse sequences (different amino acid compositions)
        ("hydrophobic", "AILVFMWP" * 12),  # 96 aa, hydrophobic
        ("charged", "DERKH" * 24),  # 120 aa, charged
        ("polar", "STNQY" * 30),  # 150 aa, polar

        # Edge cases
        ("tiny", "MKT"),  # 3 aa
        ("small", "MKTAYIAK"),  # 8 aa
        ("repeating_single", "M" * 50),  # 50 aa, single amino acid
    ]

    logger.info(f"Created {len(sequences)} test sequences")
    return sequences


def extract_v1_embeddings(
    model,
    batch_converter,
    sequences: List[Tuple[str, str]],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Extract embeddings via v1.0 path (standard attention).

    Uses model.forward() which goes through standard ESM-2 attention
    (torch.bmm with cuBLAS, FP16 accumulation on Ampere+).

    Args:
        model: ESM2WithFlashAttention model
        batch_converter: ESM alphabet batch converter
        sequences: List of (id, sequence) tuples
        device: CUDA device

    Returns:
        Dict mapping sequence IDs to mean-pooled embeddings (FP32, CPU)
    """
    embeddings = {}
    model.eval()

    num_layers = len(model.model.layers)

    with torch.no_grad():
        for seq_id, seq_str in sequences:
            # Tokenize single sequence
            labels, strs, tokens = batch_converter([(seq_id, seq_str)])
            tokens = tokens.to(device)

            # v1.0 path: standard forward with bmm attention
            output = model.forward(tokens, repr_layers=[num_layers])

            # Extract mean-pooled embedding (skip BOS token at position 0)
            # Mean over positions 1:len+1 (skip BOS, exclude EOS and padding)
            seq_len = min(len(seq_str), 1022)  # ESM-2 max
            embedding = output['representations'][num_layers][0, 1:seq_len + 1].mean(dim=0)

            # Convert to FP32 for comparison
            embeddings[seq_id] = embedding.float().cpu()

    logger.info(f"Extracted v1.0 embeddings for {len(embeddings)} sequences")
    return embeddings


def extract_v2_embeddings(
    model,
    batch_converter,
    sequences: List[Tuple[str, str]],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Extract embeddings via v2.0 path (FlashAttention varlen).

    Uses model.forward_packed() which goes through FlashAttention varlen
    (FP32 accumulation, tiled computation).

    Args:
        model: ESM2WithFlashAttention model
        batch_converter: ESM alphabet batch converter
        sequences: List of (id, sequence) tuples
        device: CUDA device

    Returns:
        Dict mapping sequence IDs to mean-pooled embeddings (FP32, CPU)
    """
    embeddings = {}
    model.eval()

    num_layers = len(model.model.layers)

    # For each sequence, construct packed inputs manually
    with torch.no_grad():
        for seq_id, seq_str in sequences:
            # Tokenize and get tokens
            labels, strs, tokens = batch_converter([(seq_id, seq_str)])

            # Remove padding (padding_idx=1 for ESM-2)
            # tokens shape: [1, seq_len_with_padding]
            # Strip padding from right side
            seq_tokens = tokens[0]  # Shape: [seq_len_with_padding]
            # Find first padding token
            padding_mask = seq_tokens == 1  # padding_idx=1
            if padding_mask.any():
                first_pad_idx = padding_mask.nonzero(as_tuple=True)[0][0].item()
                seq_tokens = seq_tokens[:first_pad_idx]

            # Now seq_tokens contains [BOS, ...seq..., EOS] without padding
            # Shape: [actual_seq_len] where actual_seq_len = len(seq_str) + 2

            # Create packed inputs (single sequence, but in packed format)
            input_ids = seq_tokens.to(device)  # 1D tensor
            cu_seqlens = torch.tensor([0, len(seq_tokens)], dtype=torch.int32, device=device)
            max_seqlen = len(seq_tokens)

            # v2.0 path: FlashAttention varlen
            output = model.forward_packed(
                input_ids=input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                repr_layers=[num_layers]
            )

            # Extract mean-pooled embedding
            # Packed output shape: [total_tokens, hidden_dim]
            # For single sequence: total_tokens = len(seq_tokens)
            packed_repr = output['representations'][num_layers]

            # Mean-pool positions (skip BOS at position 0, skip EOS at position len-1)
            # Mean over positions 1:len-1
            embedding = packed_repr[1:-1].mean(dim=0)

            # Convert to FP32 for comparison
            embeddings[seq_id] = embedding.float().cpu()

    logger.info(f"Extracted v2.0 embeddings for {len(embeddings)} sequences")
    return embeddings


def compute_embedding_metrics(
    v1_embeddings: Dict[str, torch.Tensor],
    v2_embeddings: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """Compute embedding-level similarity metrics.

    Args:
        v1_embeddings: Embeddings from v1.0 path (standard attention)
        v2_embeddings: Embeddings from v2.0 path (FlashAttention)

    Returns:
        Dict with per-sequence and aggregate metrics
    """
    per_sequence = {}
    cosine_sims = []
    mean_abs_diffs = []

    for seq_id in v1_embeddings.keys():
        v1_emb = v1_embeddings[seq_id]
        v2_emb = v2_embeddings[seq_id]

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            v1_emb.unsqueeze(0),
            v2_emb.unsqueeze(0),
            dim=1
        ).item()

        # Mean absolute difference
        mean_abs_diff = (v1_emb - v2_emb).abs().mean().item()

        per_sequence[seq_id] = {
            'cosine_similarity': cos_sim,
            'mean_abs_diff': mean_abs_diff
        }

        cosine_sims.append(cos_sim)
        mean_abs_diffs.append(mean_abs_diff)

    return {
        'per_sequence': per_sequence,
        'min_cosine': min(cosine_sims),
        'mean_cosine': sum(cosine_sims) / len(cosine_sims),
        'max_cosine': max(cosine_sims),
        'min_abs_diff': min(mean_abs_diffs),
        'mean_abs_diff': sum(mean_abs_diffs) / len(mean_abs_diffs),
        'max_abs_diff': max(mean_abs_diffs),
    }


def simulate_classification(
    v1_embeddings: Dict[str, torch.Tensor],
    v2_embeddings: Dict[str, torch.Tensor],
    seed: int = 42
) -> Dict[str, Any]:
    """Simulate MLP classification to measure prediction-level divergence.

    Since the real MLP expects concatenated DNABERT-S + ESM-2 features (3328 dim),
    and we only have ESM-2 (2560 dim), we create a deterministic random classifier
    that maps 2560 -> 2. This is a PROXY for the real classifier, not a replacement.

    The point is to measure whether embedding differences survive a linear projection
    + softmax, which is the same operation structure as the real MLP.

    Args:
        v1_embeddings: Embeddings from v1.0 path
        v2_embeddings: Embeddings from v2.0 path
        seed: Random seed for deterministic classifier

    Returns:
        Dict with classification metrics (label agreement, confidence correlation)
    """
    torch.manual_seed(seed)

    # Create deterministic random classifier: 2560 -> 2
    # Use simple linear layer (no hidden layer) for this diagnostic
    hidden_dim = 2560  # ESM-2 3B embedding size
    num_classes = 2  # Binary classification

    classifier = torch.nn.Linear(hidden_dim, num_classes)
    classifier.eval()

    # Extract predictions for both embedding sets
    v1_predictions = {}
    v2_predictions = {}

    with torch.no_grad():
        for seq_id in v1_embeddings.keys():
            # v1.0 path predictions
            v1_logits = classifier(v1_embeddings[seq_id].unsqueeze(0))
            v1_probs = F.softmax(v1_logits, dim=1).squeeze(0)
            v1_label = v1_logits.argmax(dim=1).item()
            v1_confidence = v1_probs.max().item()

            v1_predictions[seq_id] = {
                'label': v1_label,
                'confidence': v1_confidence,
                'probabilities': v1_probs.tolist()
            }

            # v2.0 path predictions
            v2_logits = classifier(v2_embeddings[seq_id].unsqueeze(0))
            v2_probs = F.softmax(v2_logits, dim=1).squeeze(0)
            v2_label = v2_logits.argmax(dim=1).item()
            v2_confidence = v2_probs.max().item()

            v2_predictions[seq_id] = {
                'label': v2_label,
                'confidence': v2_confidence,
                'probabilities': v2_probs.tolist()
            }

    # Compute classification metrics
    label_matches = []
    confidence_v1 = []
    confidence_v2 = []
    max_confidence_diffs = []

    for seq_id in v1_predictions.keys():
        v1_pred = v1_predictions[seq_id]
        v2_pred = v2_predictions[seq_id]

        # Label agreement
        label_matches.append(v1_pred['label'] == v2_pred['label'])

        # Confidence scores
        confidence_v1.append(v1_pred['confidence'])
        confidence_v2.append(v2_pred['confidence'])

        # Max confidence difference
        max_confidence_diffs.append(abs(v1_pred['confidence'] - v2_pred['confidence']))

    label_agreement = sum(label_matches) / len(label_matches) if label_matches else 0.0

    # Pearson correlation for confidence scores
    if len(confidence_v1) > 1:
        confidence_v1_tensor = torch.tensor(confidence_v1)
        confidence_v2_tensor = torch.tensor(confidence_v2)

        # Pearson correlation
        v1_mean = confidence_v1_tensor.mean()
        v2_mean = confidence_v2_tensor.mean()
        v1_centered = confidence_v1_tensor - v1_mean
        v2_centered = confidence_v2_tensor - v2_mean

        correlation = (v1_centered * v2_centered).sum() / (
            torch.sqrt((v1_centered ** 2).sum()) * torch.sqrt((v2_centered ** 2).sum())
        )
        confidence_correlation = correlation.item()
    else:
        confidence_correlation = 1.0

    return {
        'label_agreement': label_agreement,
        'confidence_correlation': confidence_correlation,
        'mean_confidence_diff': sum(max_confidence_diffs) / len(max_confidence_diffs),
        'max_confidence_diff': max(max_confidence_diffs),
        'per_sequence': {
            seq_id: {
                'v1_label': v1_predictions[seq_id]['label'],
                'v2_label': v2_predictions[seq_id]['label'],
                'label_match': v1_predictions[seq_id]['label'] == v2_predictions[seq_id]['label'],
                'v1_confidence': v1_predictions[seq_id]['confidence'],
                'v2_confidence': v2_predictions[seq_id]['confidence'],
                'confidence_diff': abs(
                    v1_predictions[seq_id]['confidence'] - v2_predictions[seq_id]['confidence']
                ),
            }
            for seq_id in v1_predictions.keys()
        }
    }


@pytest.mark.gpu
@pytest.mark.slow
class TestPredictionLevelDivergence:
    """Test prediction-level impact of FlashAttention vs standard attention."""

    def test_prediction_level_divergence(self, esm_model, test_sequences):
        """
        Quantify downstream prediction impact of FlashAttention divergence.

        This test runs identical sequences through both v1.0 (standard attention)
        and v2.0 (FlashAttention) paths, extracts mean-pooled embeddings, and
        compares:
        1. Embedding-level similarity (cosine, mean abs diff)
        2. Prediction-level agreement (label agreement, confidence correlation)

        The test produces actionable recommendations based on the research
        decision tree from 10.2-RESEARCH.md.
        """
        model, batch_converter = esm_model
        device = torch.device("cuda:0")

        logger.info("=" * 80)
        logger.info("PREDICTION-LEVEL DIVERGENCE TEST")
        logger.info("=" * 80)
        logger.info(f"Testing {len(test_sequences)} sequences")
        logger.info(f"Model: {model}")

        # Step 1: Extract v1.0 embeddings (standard attention)
        logger.info("\n[1/4] Extracting v1.0 embeddings (standard attention)...")
        v1_embeddings = extract_v1_embeddings(model, batch_converter, test_sequences, device)

        # Step 2: Extract v2.0 embeddings (FlashAttention)
        logger.info("[2/4] Extracting v2.0 embeddings (FlashAttention varlen)...")
        v2_embeddings = extract_v2_embeddings(model, batch_converter, test_sequences, device)

        # Step 3: Compute embedding-level metrics
        logger.info("[3/4] Computing embedding-level metrics...")
        embedding_metrics = compute_embedding_metrics(v1_embeddings, v2_embeddings)

        # Step 4: Simulate classification
        logger.info("[4/4] Simulating classification...")
        classification_metrics = simulate_classification(v1_embeddings, v2_embeddings, seed=42)

        # Generate report
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)

        logger.info("\n--- EMBEDDING-LEVEL METRICS ---")
        logger.info(f"Cosine Similarity:")
        logger.info(f"  Min:  {embedding_metrics['min_cosine']:.6f}")
        logger.info(f"  Mean: {embedding_metrics['mean_cosine']:.6f}")
        logger.info(f"  Max:  {embedding_metrics['max_cosine']:.6f}")
        logger.info(f"Mean Absolute Difference:")
        logger.info(f"  Min:  {embedding_metrics['min_abs_diff']:.6f}")
        logger.info(f"  Mean: {embedding_metrics['mean_abs_diff']:.6f}")
        logger.info(f"  Max:  {embedding_metrics['max_abs_diff']:.6f}")

        logger.info("\n--- PREDICTION-LEVEL METRICS ---")
        logger.info(f"Label Agreement:         {classification_metrics['label_agreement']:.1%}")
        logger.info(f"Confidence Correlation:  {classification_metrics['confidence_correlation']:.6f}")
        logger.info(f"Mean Confidence Diff:    {classification_metrics['mean_confidence_diff']:.6f}")
        logger.info(f"Max Confidence Diff:     {classification_metrics['max_confidence_diff']:.6f}")

        # Decision tree recommendation
        label_agreement = classification_metrics['label_agreement']
        logger.info("\n--- RECOMMENDATION (from research decision tree) ---")
        if label_agreement > 0.99:
            recommendation = "ACCEPT v2.0 - divergence is cosmetic"
            logger.info(f"✓ {recommendation}")
            logger.info("  Rationale: >99% label agreement means embedding differences")
            logger.info("  do not affect downstream predictions.")
        elif label_agreement >= 0.95:
            recommendation = "VALIDATE on production dataset before accepting"
            logger.info(f"⚠ {recommendation}")
            logger.info("  Rationale: 95-99% label agreement is borderline.")
            logger.info("  Test on full production dataset before final decision.")
        else:
            recommendation = "IMPLEMENT v1.0-compatible attention fallback"
            logger.info(f"✗ {recommendation}")
            logger.info("  Rationale: <95% label agreement is unacceptable.")
            logger.info("  Must fix attention implementation to match v1.0 behavior.")

        logger.info("=" * 80)

        # Assertions (based on Phase 8 validation thresholds)
        # Cosine similarity should be >0.99 for all sequences
        assert embedding_metrics['min_cosine'] > 0.99, (
            f"Min cosine similarity {embedding_metrics['min_cosine']:.6f} <= 0.99. "
            f"This indicates significant embedding divergence."
        )

        # Label agreement should be >= 95% (minimum threshold from decision tree)
        assert classification_metrics['label_agreement'] >= 0.95, (
            f"Label agreement {classification_metrics['label_agreement']:.1%} < 95%. "
            f"Recommendation: {recommendation}"
        )

        # Log per-sequence details for debugging
        logger.info("\n--- PER-SEQUENCE DETAILS (first 5) ---")
        for i, seq_id in enumerate(list(test_sequences)[:5]):
            seq_id_str = seq_id[0]
            emb_metrics = embedding_metrics['per_sequence'][seq_id_str]
            class_metrics = classification_metrics['per_sequence'][seq_id_str]

            logger.info(f"\n{seq_id_str}:")
            logger.info(f"  Cosine:  {emb_metrics['cosine_similarity']:.6f}")
            logger.info(f"  Abs diff: {emb_metrics['mean_abs_diff']:.6f}")
            logger.info(f"  v1 label: {class_metrics['v1_label']} (conf: {class_metrics['v1_confidence']:.4f})")
            logger.info(f"  v2 label: {class_metrics['v2_label']} (conf: {class_metrics['v2_confidence']:.4f})")
            logger.info(f"  Match: {'✓' if class_metrics['label_match'] else '✗'}")

        logger.info("\nTest complete.")


@pytest.mark.gpu
@pytest.mark.slow
class TestV1CompatiblePath:
    """Validate that v1_compatible=True produces identical output to forward()."""

    def test_v1_compatible_matches_standard(self, esm_model, test_sequences):
        """
        Verify v1_compatible path produces embeddings identical to standard forward().

        This test validates that when v1_compatible=True is set, the packed forward
        path delegates to standard attention (torch.bmm with FP16 accumulation) and
        produces embeddings nearly identical to the standard forward() path.

        Purpose: Ensure the v1.0 compatibility mode works as intended, providing a
        safety valve for production workloads that require exact v1.0 reproduction.
        """
        model, batch_converter = esm_model
        device = torch.device("cuda:0")

        logger.info("=" * 80)
        logger.info("V1-COMPATIBLE PATH VALIDATION TEST")
        logger.info("=" * 80)
        logger.info(f"Testing {len(test_sequences)} sequences")
        logger.info(f"Model: {model}")

        num_layers = len(model.model.layers)
        cosine_sims = []
        max_abs_diffs = []

        # Test each sequence individually
        with torch.no_grad():
            for seq_id, seq_str in test_sequences:
                # 1. Standard forward path (v1.0)
                labels, strs, tokens = batch_converter([(seq_id, seq_str)])
                tokens = tokens.to(device)
                output_standard = model.forward(tokens, repr_layers=[num_layers])

                # Extract mean-pooled embedding (skip BOS at position 0, exclude EOS)
                seq_len = min(len(seq_str), 1022)
                embedding_standard = output_standard['representations'][num_layers][0, 1:seq_len].mean(dim=0)
                embedding_standard = embedding_standard.float().cpu()

                # 2. Packed forward path with v1_compatible=True
                # Remove padding from tokens
                seq_tokens = tokens[0]  # Shape: [seq_len_with_padding]
                padding_mask = seq_tokens == 1  # padding_idx=1
                if padding_mask.any():
                    first_pad_idx = padding_mask.nonzero(as_tuple=True)[0][0].item()
                    seq_tokens = seq_tokens[:first_pad_idx]

                # Create packed inputs
                input_ids = seq_tokens.to(device)
                cu_seqlens = torch.tensor([0, len(seq_tokens)], dtype=torch.int32, device=device)
                max_seqlen = len(seq_tokens)

                # Call forward_packed with v1_compatible=True
                output_v1_compat = model.forward_packed(
                    input_ids=input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    repr_layers=[num_layers],
                    v1_compatible=True  # Use standard attention
                )

                # Extract mean-pooled embedding (skip BOS at position 0, skip EOS at position len-1)
                packed_repr = output_v1_compat['representations'][num_layers]
                embedding_v1_compat = packed_repr[1:-1].mean(dim=0)
                embedding_v1_compat = embedding_v1_compat.float().cpu()

                # 3. Compare embeddings
                cos_sim = F.cosine_similarity(
                    embedding_standard.unsqueeze(0),
                    embedding_v1_compat.unsqueeze(0),
                    dim=1
                ).item()

                max_abs_diff = (embedding_standard - embedding_v1_compat).abs().max().item()

                cosine_sims.append(cos_sim)
                max_abs_diffs.append(max_abs_diff)

                logger.info(f"{seq_id}: cosine={cos_sim:.6f}, max_abs_diff={max_abs_diff:.6f}")

        # Aggregate results
        min_cosine = min(cosine_sims)
        mean_cosine = sum(cosine_sims) / len(cosine_sims)
        mean_abs_diff = sum(max_abs_diffs) / len(max_abs_diffs)
        max_abs_diff_overall = max(max_abs_diffs)

        logger.info("\n" + "=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Cosine Similarity:")
        logger.info(f"  Min:  {min_cosine:.6f}")
        logger.info(f"  Mean: {mean_cosine:.6f}")
        logger.info(f"Max Absolute Difference:")
        logger.info(f"  Mean: {mean_abs_diff:.6f}")
        logger.info(f"  Max:  {max_abs_diff_overall:.6f}")

        # Assertions: v1_compatible path should be MUCH closer to standard forward
        # than FlashAttention path (which was 0.999999-1.0 in Plan 01)
        # Here we expect near-perfect match since both use standard attention
        assert min_cosine > 0.999, (
            f"Min cosine similarity {min_cosine:.6f} <= 0.999. "
            f"v1_compatible path should match standard forward() very closely."
        )

        assert max_abs_diff_overall < 0.001, (
            f"Max absolute difference {max_abs_diff_overall:.6f} >= 0.001. "
            f"v1_compatible path should produce nearly identical embeddings to standard forward()."
        )

        logger.info("\n✓ v1_compatible path matches standard forward() (cosine > 0.999, max_abs_diff < 0.001)")
        logger.info("  This confirms the v1.0 compatibility mode works as intended.")
        logger.info("=" * 80)

    def teardown_method(self):
        os.environ.pop('VIRNUCPRO_V1_ATTENTION', None)

