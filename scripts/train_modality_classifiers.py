#!/usr/bin/env python3
"""
Train separate classifiers for DNA-only, protein-only, and combined features.

This helps us understand the individual contributions of:
- DNABERT-S (768-dim DNA features)
- FastESM2-650M (1280-dim protein features)
- Combined (2048-dim)

Usage:
    python scripts/train_modality_classifiers.py --data-dir data/data_merge
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import argparse
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Simple MLP for different input dimensions
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


class ModalityDataset(Dataset):
    """Dataset that extracts specific modality features."""
    def __init__(self, pt_files, modality='combined'):
        """
        Args:
            pt_files: List of .pt files
            modality: 'dna', 'protein', or 'combined'
        """
        self.modality = modality
        self.features = []
        self.labels = []

        for pt_file in pt_files:
            data = torch.load(pt_file, weights_only=False)
            features_full = data['data']  # Shape: [N, 2048]
            label = data['labels'][0]  # 0 or 1

            # Extract modality-specific features
            if modality == 'dna':
                # First 768 dimensions
                features = features_full[:, :768]
            elif modality == 'protein':
                # Last 1280 dimensions
                features = features_full[:, 768:]
            else:  # combined
                features = features_full

            self.features.append(features)
            self.labels.extend([label] * len(features))

        self.features = torch.cat(self.features, dim=0)
        self.labels = torch.tensor(self.labels)

        print(f"Loaded {modality} dataset: {len(self.features)} samples")
        print(f"  Feature dim: {self.features.shape[1]}")
        print(f"  Labels: {(self.labels == 0).sum()} non-viral, {(self.labels == 1).sum()} viral")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """Train a modality classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None
    history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []
        val_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)

                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(predictions.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels_list, val_preds)
        val_auc = roc_auc_score(val_labels_list, val_probs)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc
        })

    # Restore best model
    model.load_state_dict(best_model_state)

    return model, history, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-metadata', default='data/test_set/test_metadata.json')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("Train Modality-Specific Classifiers")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load test metadata to get train/test split
    with open(args.test_metadata) as f:
        metadata = json.load(f)

    train_files = metadata['train_files']
    test_files = metadata['test_files']

    print(f"\nTrain files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    # Train models for each modality
    modalities = {
        'dna': 768,
        'protein': 1280,
        'combined': 2048
    }

    results = {}

    for modality, input_dim in modalities.items():
        print("\n" + "=" * 70)
        print(f"Training {modality.upper()} classifier ({input_dim}-dim)")
        print("=" * 70)

        # Create datasets
        print("\nLoading training data...")
        train_dataset = ModalityDataset(train_files, modality=modality)
        print("\nLoading test data...")
        test_dataset = ModalityDataset(test_files, modality=modality)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        model = SimpleMLPClassifier(input_dim=input_dim).to(device)

        # Train
        print("\nTraining...")
        model, history, best_val_acc = train_model(
            model, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr
        )

        # Final evaluation
        print("\nFinal test set evaluation...")
        model.eval()
        test_preds = []
        test_labels = []
        test_probs = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                test_preds.extend(predictions.cpu().numpy())
                test_labels.extend(labels.numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        auc = roc_auc_score(test_labels, test_probs)

        print(f"\n{modality.upper()} Test Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")

        # Save model
        output_path = f"300_model_fastesm650_{modality}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': {
                'modality': modality,
                'input_dim': input_dim,
                'merged_dim': input_dim,
                'checkpoint_version': '2.0.0',
                'timestamp': datetime.now().isoformat()
            }
        }, output_path)

        print(f"\nModel saved: {output_path}")

        # Store results
        results[modality] = {
            'input_dim': input_dim,
            'test_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(auc)
            },
            'training_history': history,
            'model_path': output_path
        }

    # Save comparison report
    with open('reports/modality_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print comparison
    print("\n" + "=" * 70)
    print("MODALITY COMPARISON (Full-Length Sequences)")
    print("=" * 70)
    print(f"\n{'Modality':<12} {'Dim':<6} {'Accuracy':<10} {'F1 Score':<10} {'ROC-AUC':<10}")
    print("-" * 70)
    for modality in ['dna', 'protein', 'combined']:
        metrics = results[modality]['test_metrics']
        dim = results[modality]['input_dim']
        print(f"{modality:<12} {dim:<6} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['roc_auc']:<10.4f}")

    print(f"\nResults saved to: reports/modality_comparison.json")
    print(f"\nNext: Test on 150bp reads with evaluate_modality_150bp.py")


if __name__ == '__main__':
    main()
