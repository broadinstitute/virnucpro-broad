#!/usr/bin/env python3
"""
Evaluate FastESM2-650M model on 150bp short reads.

Runs predictions on 150bp embeddings and calculates metrics to compare
with full-length sequence performance.

Usage:
    python scripts/evaluate_150bp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import numpy as np

from units import MERGED_DIM

# MLPClassifier matching trained model
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_class=2):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, num_class)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        if x.shape[1] != MERGED_DIM:
            raise ValueError(f"Expected {MERGED_DIM}-dim input, got {x.shape[1]}")
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def load_model(checkpoint_path, device):
    """Load trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Validate checkpoint
    if 'metadata' in checkpoint:
        meta = checkpoint['metadata']
        if meta.get('merged_dim') != MERGED_DIM:
            raise ValueError(f"Checkpoint trained on {meta.get('merged_dim')}-dim, but expected {MERGED_DIM}-dim")

    model = MLPClassifier(input_dim=MERGED_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='300_model_fastesm650.pth',
                        help='Model checkpoint to evaluate')
    parser.add_argument('--viral-data', default='data/test_150bp_50k/viral_merged.pt',
                        help='Viral test data')
    parser.add_argument('--nonviral-data', default='data/test_150bp_50k/nonviral_merged.pt',
                        help='Non-viral test data')
    parser.add_argument('--output', default='reports/150bp_evaluation.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print("=" * 70)
    print("150bp Short Read Evaluation - FastESM2-650M")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Paths
    model_path = args.model
    viral_pt = args.viral_data
    nonviral_pt = args.nonviral_data

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path, device)

    # Load 150bp embeddings
    print(f"\nLoading 150bp embeddings...")
    print(f"  Viral: {viral_pt}")
    print(f"  Non-viral: {nonviral_pt}")

    viral_data = torch.load(viral_pt, weights_only=False)
    nonviral_data = torch.load(nonviral_pt, weights_only=False)

    viral_features = viral_data['data']
    nonviral_features = nonviral_data['data']

    print(f"\nViral windows: {len(viral_features)}")
    print(f"Non-viral windows: {len(nonviral_features)}")
    print(f"Total: {len(viral_features) + len(nonviral_features)}")

    # Combine data
    features = torch.cat([nonviral_features, viral_features], dim=0)
    labels = torch.tensor([0] * len(nonviral_features) + [1] * len(viral_features))

    print(f"\nCombined dataset shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Run predictions with speed measurement
    print("\nRunning predictions...")
    features = features.to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(features)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    inference_time = time.time() - start_time

    # Move to CPU for metrics
    predictions = predictions.cpu().numpy()
    probs_cpu = probs.cpu().numpy()
    labels_np = labels.numpy()

    # Calculate metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    accuracy = accuracy_score(labels_np, predictions)
    precision = precision_score(labels_np, predictions, pos_label=1)
    recall = recall_score(labels_np, predictions, pos_label=1)
    f1 = f1_score(labels_np, predictions, pos_label=1)
    roc_auc = roc_auc_score(labels_np, probs_cpu[:, 1])

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels_np, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-Viral  Viral")
    print(f"Actual Non-Viral     {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"       Viral         {cm[1,0]:5d}    {cm[1,1]:5d}")

    # Speed metrics
    print("\n" + "=" * 70)
    print("SPEED METRICS")
    print("=" * 70)
    samples_per_sec = len(features) / inference_time
    print(f"\nInference time: {inference_time:.2f}s")
    print(f"Samples: {len(features)}")
    print(f"Speed: {samples_per_sec:.0f} sequences/sec")

    # Save results
    results = {
        'dataset': '150bp_short_reads',
        'model': '300_model_fastesm650.pth',
        'sample_size': len(features),
        'viral_samples': len(viral_features),
        'nonviral_samples': len(nonviral_features),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'confusion_matrix': {
            'true_negative': int(cm[0,0]),
            'false_positive': int(cm[0,1]),
            'false_negative': int(cm[1,0]),
            'true_positive': int(cm[1,1])
        },
        'speed': {
            'inference_time_sec': float(inference_time),
            'samples_per_sec': float(samples_per_sec)
        }
    }

    output_file = args.output
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Load full-length results for comparison
    try:
        with open('reports/validation_report.md') as f:
            content = f.read()
            # Extract full-length metrics for comparison
            print("\n" + "=" * 70)
            print("COMPARISON: 150bp vs Full-Length Sequences")
            print("=" * 70)
            print(f"\n150bp reads:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"\nFull-length (from validation report):")
            print(f"  Accuracy:  0.9020")
            print(f"  Recall:    0.9206")
            print(f"  F1 Score:  0.9078")

            acc_drop = 0.9020 - accuracy
            recall_drop = 0.9206 - recall
            f1_drop = 0.9078 - f1

            print(f"\nPerformance degradation on 150bp reads:")
            print(f"  Accuracy drop:  {acc_drop:.4f} ({acc_drop/0.9020*100:.1f}%)")
            print(f"  Recall drop:    {recall_drop:.4f} ({recall_drop/0.9206*100:.1f}%)")
            print(f"  F1 drop:        {f1_drop:.4f} ({f1_drop/0.9078*100:.1f}%)")
    except:
        pass

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
