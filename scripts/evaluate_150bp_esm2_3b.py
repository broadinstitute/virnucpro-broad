#!/usr/bin/env python3
"""
Evaluate ESM2-3B model on 150bp short reads and compare with FastESM2-650M.

Usage:
    python scripts/evaluate_150bp_esm2_3b.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import numpy as np

# MLPClassifier matching the original architecture in prediction.py
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def load_model(checkpoint_path, device):
    """Load trained ESM2-3B model."""
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The old 300_model.pth is saved as the complete model object
    # Just use it directly
    model.to(device)
    model.eval()

    print(f"Loaded model: {type(model).__name__}")
    print(f"Model architecture: {model}")

    return model


def main():
    print("=" * 70)
    print("150bp Short Read Evaluation - ESM2-3B")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Paths
    model_path = '300_model.pth'
    viral_pt = 'data/test_150bp_50k_esm2_3b/viral_merged.pt'
    nonviral_pt = 'data/test_150bp_50k_esm2_3b/nonviral_merged.pt'

    # Load model
    print(f"\nLoading ESM2-3B model from {model_path}...")
    model = load_model(model_path, device)

    # Load 150bp embeddings
    print(f"\nLoading 150bp ESM2-3B embeddings...")
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
    print("PERFORMANCE METRICS - ESM2-3B")
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
        'model': 'ESM2-3B (300_model.pth)',
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

    output_file = 'reports/150bp_evaluation_esm2_3b.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Load FastESM2-650M and full-length results for comparison
    try:
        with open('reports/150bp_evaluation.json') as f:
            fastesm_results = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON: ESM2-3B vs FastESM2-650M on 150bp reads")
        print("=" * 70)
        print(f"\nESM2-3B (this run):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        fast_acc = fastesm_results['metrics']['accuracy']
        fast_prec = fastesm_results['metrics']['precision']
        fast_rec = fastesm_results['metrics']['recall']
        fast_f1 = fastesm_results['metrics']['f1_score']

        print(f"\nFastESM2-650M:")
        print(f"  Accuracy:  {fast_acc:.4f}")
        print(f"  Precision: {fast_prec:.4f}")
        print(f"  Recall:    {fast_rec:.4f}")
        print(f"  F1 Score:  {fast_f1:.4f}")

        print(f"\nDifference (ESM2-3B - FastESM2-650M):")
        print(f"  Accuracy:  {accuracy - fast_acc:+.4f}")
        print(f"  Precision: {precision - fast_prec:+.4f}")
        print(f"  Recall:    {recall - fast_rec:+.4f}")
        print(f"  F1 Score:  {f1 - fast_f1:+.4f}")

        print(f"\n" + "=" * 70)
        print("COMPARISON: 150bp vs Full-Length (ESM2-3B)")
        print("=" * 70)
        print(f"\n150bp reads (ESM2-3B):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\nFull-length (ESM2-3B from validation):")
        print(f"  Accuracy:  0.9048")
        print(f"  Recall:    0.9541")
        print(f"  F1 Score:  0.9191")

        acc_drop = 0.9048 - accuracy
        recall_drop = 0.9541 - recall
        f1_drop = 0.9191 - f1

        print(f"\nPerformance degradation on 150bp (ESM2-3B):")
        print(f"  Accuracy drop:  {acc_drop:.4f} ({acc_drop/0.9048*100:.1f}%)")
        print(f"  Recall drop:    {recall_drop:.4f} ({recall_drop/0.9541*100:.1f}%)")
        print(f"  F1 drop:        {f1_drop:.4f} ({f1_drop/0.9191*100:.1f}%)")
    except Exception as e:
        print(f"\nCould not load comparison data: {e}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
