#!/usr/bin/env python3
"""
Fine-tune FastESM2-650M model on 150bp short reads.

This script takes the existing trained model and fine-tunes it on 150bp data
to test if exposure to short reads improves performance.

Usage:
    python scripts/finetune_150bp.py --data-dir data/train_150bp --epochs 10
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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


class ShortReadDataset(Dataset):
    """Dataset for 150bp training data."""
    def __init__(self, viral_pt, nonviral_pt):
        # Load data
        viral_data = torch.load(viral_pt, weights_only=False)
        nonviral_data = torch.load(nonviral_pt, weights_only=False)

        # Combine
        self.features = torch.cat([nonviral_data['data'], viral_data['data']], dim=0)
        self.labels = torch.tensor(
            [0] * len(nonviral_data['data']) + [1] * len(viral_data['data'])
        )

        print(f"Loaded dataset: {len(self.features)} samples")
        print(f"  Non-viral: {len(nonviral_data['data'])}")
        print(f"  Viral: {len(viral_data['data'])}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_pretrained_model(checkpoint_path, device):
    """Load the pre-trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Validate checkpoint
    if 'metadata' in checkpoint:
        meta = checkpoint['metadata']
        if meta.get('merged_dim') != MERGED_DIM:
            raise ValueError(f"Checkpoint trained on {meta.get('merged_dim')}-dim, but expected {MERGED_DIM}-dim")

    model = MLPClassifier(input_dim=MERGED_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model


def finetune(model, train_loader, device, epochs=10, lr=1e-4, freeze_layers=False):
    """
    Fine-tune the model on 150bp data.

    Args:
        model: Pre-trained model
        train_loader: DataLoader for training data
        device: CUDA device
        epochs: Number of fine-tuning epochs
        lr: Learning rate (much lower than initial training)
        freeze_layers: If True, freeze hidden layer and only train output layer
    """
    print("\n" + "=" * 70)
    print("FINE-TUNING CONFIGURATION")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Freeze layers: {freeze_layers}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    # Optionally freeze layers
    if freeze_layers:
        print("\nFreezing hidden layer, only training output layer...")
        model.hidden_layer.requires_grad_(False)
        model.bn1.requires_grad_(False)
        params_to_train = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in params_to_train)}")
    else:
        params_to_train = model.parameters()

    # Optimizer and loss
    optimizer = torch.optim.Adam(params_to_train, lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        })

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-model', default='300_model_fastesm650.pth',
                        help='Pre-trained model checkpoint')
    parser.add_argument('--viral-data', default='data/train_150bp/viral_merged.pt',
                        help='Viral 150bp training data')
    parser.add_argument('--nonviral-data', default='data/train_150bp/nonviral_merged.pt',
                        help='Non-viral 150bp training data')
    parser.add_argument('--output', default='300_model_fastesm650_finetuned_150bp.pth',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower than initial training)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--freeze-layers', action='store_true',
                        help='Freeze hidden layers, only train output layer')
    args = parser.parse_args()

    print("=" * 70)
    print("Fine-Tune FastESM2-650M on 150bp Short Reads")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load pre-trained model
    print(f"\nLoading pre-trained model: {args.pretrained_model}")
    model = load_pretrained_model(args.pretrained_model, device)

    # Load training data
    print(f"\nLoading 150bp training data...")
    dataset = ShortReadDataset(args.viral_data, args.nonviral_data)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Fine-tune
    print("\nStarting fine-tuning...")
    model, history = finetune(
        model, train_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        freeze_layers=args.freeze_layers
    )

    # Save fine-tuned model
    print(f"\nSaving fine-tuned model to: {args.output}")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': {
            'merged_dim': MERGED_DIM,
            'checkpoint_version': '2.0.0',
            'finetuned_on': '150bp_short_reads',
            'base_model': args.pretrained_model,
            'finetune_epochs': args.epochs,
            'finetune_lr': args.lr,
            'freeze_layers': args.freeze_layers,
            'training_samples': len(dataset),
            'timestamp': datetime.now().isoformat()
        },
        'training_history': history
    }

    torch.save(checkpoint, args.output)

    # Save training history
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
    print(f"History saved: {history_file}")
    print(f"\nFinal training accuracy: {history[-1]['accuracy']:.4f}")
    print(f"\nNext step: Evaluate on 150bp test set with evaluate_150bp.py")
    print(f"  python scripts/evaluate_150bp.py --model {args.output}")


if __name__ == '__main__':
    main()
