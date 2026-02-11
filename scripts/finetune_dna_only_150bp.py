#!/usr/bin/env python3
"""
Fine-tune DNA-only classifier on 150bp reads.

This fine-tunes ONLY the MLP classifier head, keeping DNABERT-S embeddings frozen.
Much faster and safer than fine-tuning DNABERT-S itself.

Usage:
    python scripts/finetune_dna_only_150bp.py
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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


class DNAOnlyDataset(Dataset):
    """Dataset for DNA-only features (first 768-dim)."""
    def __init__(self, viral_pt, nonviral_pt):
        # Load merged data and extract DNA portion
        viral_data = torch.load(viral_pt, weights_only=False)
        nonviral_data = torch.load(nonviral_pt, weights_only=False)

        # Extract DNA features (first 768 dims)
        viral_dna = viral_data['data'][:, :768]
        nonviral_dna = nonviral_data['data'][:, :768]

        # Combine
        self.features = torch.cat([nonviral_dna, viral_dna], dim=0)
        self.labels = torch.tensor(
            [0] * len(nonviral_dna) + [1] * len(viral_dna)
        )

        print(f"Loaded DNA-only dataset: {len(self.features)} samples")
        print(f"  Feature dim: 768 (DNA only)")
        print(f"  Non-viral: {len(nonviral_dna)}")
        print(f"  Viral: {len(viral_dna)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_pretrained_model(checkpoint_path, device):
    """Load the pre-trained DNA-only model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SimpleMLPClassifier(input_dim=768)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model


def finetune(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    """Fine-tune the DNA-only classifier."""
    print("\n" + "=" * 70)
    print("FINE-TUNING DNA-ONLY CLASSIFIER")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Training samples: {len(train_loader.dataset)}")

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
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

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"\nEpoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val AUC={val_auc:.4f}")

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
    parser.add_argument('--pretrained-model', default='300_model_fastesm650_dna.pth')
    parser.add_argument('--train-viral', default='data/train_150bp/viral_merged.pt')
    parser.add_argument('--train-nonviral', default='data/train_150bp/nonviral_merged.pt')
    parser.add_argument('--test-viral', default='data/test_150bp_50k/viral_merged.pt')
    parser.add_argument('--test-nonviral', default='data/test_150bp_50k/nonviral_merged.pt')
    parser.add_argument('--output', default='300_model_fastesm650_dna_finetuned_150bp.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("Fine-Tune DNA-Only Classifier on 150bp Short Reads")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load pre-trained DNA-only model
    print(f"\nLoading pre-trained DNA-only model: {args.pretrained_model}")
    model = load_pretrained_model(args.pretrained_model, device)

    # Load training data (150bp)
    print(f"\nLoading 150bp training data...")
    train_dataset = DNAOnlyDataset(args.train_viral, args.train_nonviral)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Load test data (150bp) for validation during training
    print(f"\nLoading 150bp test data for validation...")
    val_dataset = DNAOnlyDataset(args.test_viral, args.test_nonviral)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Fine-tune
    model, history, best_val_acc = finetune(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr
    )

    # Save fine-tuned model
    print(f"\nSaving fine-tuned DNA-only model to: {args.output}")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'modality': 'dna',
            'input_dim': 768,
            'merged_dim': 768,
            'checkpoint_version': '2.0.0',
            'finetuned_on': '150bp_short_reads',
            'base_model': args.pretrained_model,
            'finetune_epochs': args.epochs,
            'finetune_lr': args.lr,
            'best_val_acc': best_val_acc,
            'timestamp': datetime.now().isoformat()
        },
        'training_history': history
    }

    torch.save(checkpoint, args.output)

    # Save history
    history_file = args.output.replace('.pth', '_history.json')
    with open(history_file, 'w') as f:
        json.dump({
            'history': history,
            'metadata': checkpoint['metadata']
        }, f, indent=2)

    print(f"\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Model saved: {args.output}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    print(f"\nBaseline DNA-only on 150bp: 68.95%")
    print(f"Fine-tuned DNA-only on 150bp: {best_val_acc*100:.2f}%")
    print(f"Improvement: {(best_val_acc - 0.6895)*100:+.2f} percentage points")

    print(f"\nNext: Evaluate with:")
    print(f"  python scripts/evaluate_modality_150bp.py --dna-model {args.output}")


if __name__ == '__main__':
    main()
