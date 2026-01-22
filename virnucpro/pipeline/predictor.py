"""Model prediction functionality"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
import logging

from virnucpro.pipeline.models import MLPClassifier, PredictDataBatchDataset

logger = logging.getLogger('virnucpro.predictor')


def predict_sequences(
    merged_feature_files: List[Path],
    model_path: Path,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4
) -> List[Tuple[str, str, float, float]]:
    """
    Perform batch prediction on merged features.

    Based on prediction.py:74-95

    Args:
        merged_feature_files: List of .pt files with merged features
        model_path: Path to trained MLP model
        device: PyTorch device
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers

    Returns:
        List of tuples: (sequence_id, prediction, score_class0, score_class1)
        where prediction is 'others' or 'virus'
    """
    logger.info(f"Loading model from {model_path}")

    # Handle model loading with class location changes
    # The model was saved with classes in __main__, but they're now in virnucpro.pipeline.models
    import sys

    # Save original __main__ attributes
    original_main = sys.modules['__main__']
    original_attrs = {}

    # Temporarily inject our classes into __main__ for unpickling
    try:
        # Save any existing attributes with the same names
        for attr in ['MLPClassifier', 'PredictDataBatchDataset']:
            if hasattr(original_main, attr):
                original_attrs[attr] = getattr(original_main, attr)

        # Inject our classes
        setattr(original_main, 'MLPClassifier', MLPClassifier)
        setattr(original_main, 'PredictDataBatchDataset', PredictDataBatchDataset)

        # Now load the model normally
        model = torch.load(model_path, map_location=device, weights_only=False)

    finally:
        # Restore original attributes
        for attr in ['MLPClassifier', 'PredictDataBatchDataset']:
            if attr in original_attrs:
                setattr(original_main, attr, original_attrs[attr])
            elif hasattr(original_main, attr):
                delattr(original_main, attr)

    model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = PredictDataBatchDataset(merged_feature_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    logger.info(f"Running prediction on {len(dataset)} sequences")

    # Perform prediction
    results = []

    with torch.no_grad():
        for batch_data, batch_ids in dataloader:
            batch_data = batch_data.to(device)

            # Forward pass
            logits = model(batch_data)

            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Get predictions and scores
            pred_classes = torch.argmax(probabilities, dim=1)

            # Process batch results
            for i in range(len(batch_ids)):
                seq_id = batch_ids[i]
                pred_class = pred_classes[i].item()
                score_0 = probabilities[i, 0].item()
                score_1 = probabilities[i, 1].item()

                # Map class to label
                prediction = 'virus' if pred_class == 1 else 'others'

                results.append((seq_id, prediction, score_0, score_1))

    logger.info(f"Prediction complete: {len(results)} sequences processed")
    return results


def compute_consensus(
    predictions: List[Tuple[str, str, float, float]]
) -> Dict[str, Tuple[str, float, float]]:
    """
    Compute consensus predictions by grouping reading frames.

    Based on prediction.py:183-191

    Groups predictions by original sequence ID (removes F1-F3, R1-R3 suffixes)
    and determines final classification based on highest scores.

    Args:
        predictions: List of (seq_id, prediction, score_0, score_1) tuples

    Returns:
        Dictionary mapping original_id -> (prediction, max_score_0, max_score_1)
    """
    logger.info("Computing consensus predictions")

    # Group by original sequence ID
    grouped = {}

    for seq_id, prediction, score_0, score_1 in predictions:
        # Remove frame indicator (F1-F3, R1-R3)
        # Assumes format: original_id_chunk_N{F|R}{1-3}
        if seq_id.endswith(('F1', 'F2', 'F3', 'R1', 'R2', 'R3')):
            original_id = seq_id[:-2]  # Remove last 2 characters
        else:
            original_id = seq_id

        if original_id not in grouped:
            grouped[original_id] = []

        grouped[original_id].append((score_0, score_1))

    # Compute consensus for each group
    consensus = {}

    for original_id, scores in grouped.items():
        # Find maximum scores across all frames
        max_score_0 = max(s[0] for s in scores)
        max_score_1 = max(s[1] for s in scores)

        # Determine prediction based on max scores
        is_virus = max_score_1 >= max_score_0
        prediction = 'virus' if is_virus else 'others'

        consensus[original_id] = (prediction, max_score_0, max_score_1)

    logger.info(f"Consensus computed for {len(consensus)} original sequences")
    return consensus
