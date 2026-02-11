#!/usr/bin/env python3
"""
Evaluate DNA-only, protein-only, and combined models on 150bp test set.

This tells us which modality contributes more to 150bp classification.

Usage:
    python scripts/evaluate_modality_150bp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Simple MLP matching training script
class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_class=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def extract_modality_features(features, modality):
    """Extract specific modality from combined features."""
    if modality == 'dna':
        return features[:, :768]
    elif modality == 'protein':
        return features[:, 768:]
    else:  # combined
        return features


def evaluate_model(model, features, labels, device):
    """Run evaluation and return metrics."""
    model.eval()
    features = features.to(device)

    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    preds = predictions.cpu().numpy()
    probs_np = probs[:, 1].cpu().numpy()
    labels_np = labels.numpy()

    return {
        'accuracy': float(accuracy_score(labels_np, preds)),
        'precision': float(precision_score(labels_np, preds)),
        'recall': float(recall_score(labels_np, preds)),
        'f1_score': float(f1_score(labels_np, preds)),
        'roc_auc': float(roc_auc_score(labels_np, probs_np))
    }


def main():
    print("=" * 70)
    print("Modality Comparison on 150bp Short Reads")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load 150bp test data
    viral_data = torch.load('data/test_150bp_50k/viral_merged.pt', weights_only=False)
    nonviral_data = torch.load('data/test_150bp_50k/nonviral_merged.pt', weights_only=False)

    # Combine
    features = torch.cat([nonviral_data['data'], viral_data['data']], dim=0)
    labels = torch.tensor([0] * len(nonviral_data['data']) + [1] * len(viral_data['data']))

    print(f"\n150bp test set: {len(features)} samples")
    print(f"  Non-viral: {len(nonviral_data['data'])}")
    print(f"  Viral: {len(viral_data['data'])}")

    # Test each modality
    modalities = {
        'dna': (768, '300_model_fastesm650_dna.pth'),
        'protein': (1280, '300_model_fastesm650_protein.pth'),
        'combined': (2048, '300_model_fastesm650_combined.pth')
    }

    results = {}

    for modality, (input_dim, model_path) in modalities.items():
        print(f"\n{'=' * 70}")
        print(f"Evaluating {modality.upper()} model ({input_dim}-dim)")
        print('=' * 70)

        # Load model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = SimpleMLPClassifier(input_dim=input_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract modality-specific features
        modality_features = extract_modality_features(features, modality)

        # Evaluate
        metrics = evaluate_model(model, modality_features, labels, device)

        print(f"\n{modality.upper()} Results on 150bp:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        results[modality] = {
            'input_dim': input_dim,
            'metrics': metrics,
            'model_path': model_path
        }

    # Save results
    output_file = 'reports/modality_comparison_150bp.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODALITY COMPARISON: 150bp vs Full-Length")
    print("=" * 70)

    # Load full-length results
    with open('reports/modality_comparison.json') as f:
        full_length_results = json.load(f)

    print(f"\n{'Modality':<12} {'Full-Length Acc':<18} {'150bp Acc':<12} {'Degradation':<12}")
    print("-" * 70)

    for modality in ['dna', 'protein', 'combined']:
        full_acc = full_length_results[modality]['test_metrics']['accuracy']
        short_acc = results[modality]['metrics']['accuracy']
        degradation = full_acc - short_acc

        print(f"{modality:<12} {full_acc:<18.4f} {short_acc:<12.4f} {degradation:<12.4f} ({degradation/full_acc*100:.1f}%)")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Calculate relative performance
    dna_acc = results['dna']['metrics']['accuracy']
    protein_acc = results['protein']['metrics']['accuracy']
    combined_acc = results['combined']['metrics']['accuracy']

    print(f"\n150bp Performance:")
    print(f"  DNA-only:     {dna_acc:.4f}")
    print(f"  Protein-only: {protein_acc:.4f}")
    print(f"  Combined:     {combined_acc:.4f}")

    if dna_acc > protein_acc:
        diff = dna_acc - protein_acc
        print(f"\n✓ DNA features are MORE informative for 150bp (+{diff:.4f} accuracy)")
        print(f"  → 50 amino acids provide limited discriminative power")
        print(f"  → DNA sequence patterns more reliable for short reads")
    else:
        diff = protein_acc - dna_acc
        print(f"\n✓ Protein features are MORE informative for 150bp (+{diff:.4f} accuracy)")
        print(f"  → Even 50 amino acids provide valuable viral signatures")

    synergy = combined_acc - max(dna_acc, protein_acc)
    if synergy > 0.01:
        print(f"\n✓ Combining modalities provides synergy (+{synergy:.4f} over best single)")
    elif synergy < -0.01:
        print(f"\n✗ Combining modalities HURTS performance ({synergy:.4f} vs best single)")
        print(f"  → Consider using DNA-only for 150bp reads")
    else:
        print(f"\n= Combining modalities provides no additional benefit")

    print(f"\n\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
