#!/usr/bin/env python3
"""
Enhanced training script for FastESM2 MLP classifier.

This script trains the MLP classifier using only the training split (excluding test files)
with early stopping, detailed per-epoch logging, and checkpoint saving with metadata.
"""

import argparse
import random
import time
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import json

from units import DimensionError, DNA_DIM, PROTEIN_DIM, MERGED_DIM, CHECKPOINT_VERSION


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Worker initialization function for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FileBatchDataset(Dataset):
    """
    Dataset that loads data from multiple .pt files.
    Duplicated from train.py to avoid module-level execution.
    """
    def __init__(self, file_list):
        self.file_list = file_list
        self.data = []
        self.labels = []
        self._load_all_data()

    def _load_all_data(self):
        for file_path in self.file_list:
            data_dict = torch.load(file_path)
            data = data_dict['data'][:, :]
            label = data_dict['labels'][0]
            self.data.append(data)
            self.labels.append(label)

    def __len__(self):
        return sum(d.size(0) for d in self.data)

    def __getitem__(self, idx):
        cumulative_size = 0
        for data, label in zip(self.data, self.labels):
            if cumulative_size + data.size(0) > idx:
                index_in_file = idx - cumulative_size
                sample = data[index_in_file]
                return sample, torch.tensor(label)
            cumulative_size += data.size(0)
        raise IndexError("Index out of range")


class MLPClassifier(nn.Module):
    """
    MLP classifier for viral/non-viral classification.
    Duplicated from train.py to avoid module-level execution.
    """
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLPClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Critical path validation - always runs
        if x.shape[-1] != self.input_dim:
            raise DimensionError(
                expected_dim=self.input_dim,
                actual_dim=x.shape[-1],
                tensor_name="model_input",
                location="MLPClassifier.forward()"
            )
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    Duplicated from train.py.
    """
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop_counter = 0
        self.best_model_wts = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = epoch
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
            self.best_epoch = epoch
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {self.best_score:.4f} at epoch {self.best_epoch}")
                model.load_state_dict(self.best_model_wts)
                return True
        return False


def save_checkpoint_with_metadata(model, optimizer, epoch, best_loss, filepath='model_fastesm650.pth'):
    """
    Save checkpoint with metadata including version, model type, and dimensions.
    Duplicated from train.py.
    """
    metadata = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'model_type': 'fastesm650',
        'huggingface_model_id': 'Synthyra/FastESM2_650',
        'dna_dim': DNA_DIM,
        'protein_dim': PROTEIN_DIM,
        'merged_dim': MERGED_DIM,
        'input_dim': MERGED_DIM,
        'hidden_dim': 512,
        'num_class': 2,
        'training_date': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'metadata': metadata
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} with metadata (version {CHECKPOINT_VERSION})")


def write_log(filename, message):
    """Write a message to the log file."""
    with open(filename, 'a') as f:
        f.write(message + '\n')


def load_metadata(metadata_path):
    """Load test set metadata from JSON file."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def validate_input_dimensions(dataloader, device, expected_dim):
    """
    Validate input dimensions before training.
    Load one batch and check shape matches expected dimension.
    """
    print("\n=== Validating input dimensions ===")
    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(device)
        actual_dim = batch_data.shape[-1]

        if actual_dim != expected_dim:
            raise DimensionError(
                expected_dim=expected_dim,
                actual_dim=actual_dim,
                tensor_name="training_data",
                location="validate_input_dimensions()"
            )

        print(f"Validation passed: input dimension = {actual_dim}")
        break  # Only check first batch


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.

    Returns:
        tuple of (avg_loss, accuracy, precision, recall, f1, auc)
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return avg_loss, accuracy, precision, recall, f1, auc


def train_model(model, optimizer, scheduler, criterion, train_loader, val_loader,
                num_epochs, log_file, patience, device, save_best=False):
    """
    Train the model with early stopping and detailed logging.

    Args:
        model: MLP model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        log_file: Path to log file
        patience: Early stopping patience
        device: Device to train on
        save_best: If True, return best model weights instead of final

    Returns:
        tuple of (best_epoch, best_val_loss)
    """
    early_stopping = EarlyStopping(patience=patience)
    start_time = time.time()

    # Write header to log file
    header = f"\n{'='*80}\n"
    header += f"Training started at {datetime.now().isoformat()}\n"
    header += f"Max epochs: {num_epochs}, Patience: {patience}\n"
    header += f"{'='*80}\n"
    write_log(log_file, header)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        # Training loop
        for batch_data, batch_labels in tqdm(train_loader, total=len(train_loader),
                                             desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate_model(
            model, val_loader, criterion, device
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time

        # Log epoch results
        log_message = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s (Total: {total_time:.2f}s)"
        )
        print(log_message)
        write_log(log_file, log_message)

        # Detailed metrics to log file only
        detail_message = (
            f"  Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"AUC: {val_auc:.4f}"
        )
        write_log(log_file, detail_message)

        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            break

        # Step scheduler
        if scheduler:
            scheduler.step()

    # Training complete
    total_time = time.time() - start_time
    summary = f"\n{'='*80}\n"
    summary += f"Training completed at {datetime.now().isoformat()}\n"
    summary += f"Total epochs: {epoch + 1}\n"
    summary += f"Best epoch: {early_stopping.best_epoch + 1}\n"
    summary += f"Best val loss: {early_stopping.best_score:.4f}\n"
    summary += f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)\n"
    summary += f"{'='*80}\n"
    print(summary)
    write_log(log_file, summary)

    return early_stopping.best_epoch, early_stopping.best_score


def main():
    parser = argparse.ArgumentParser(
        description='Train FastESM2 MLP classifier with early stopping and detailed logging'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='./data/test_set/test_metadata.json',
        help='Path to test set metadata JSON file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Maximum number of training epochs (default: 200)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=7,
        help='Early stopping patience (default: 7)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0002,
        help='Learning rate (default: 0.0002)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_fastesm650.pth',
        help='Output checkpoint path (default: model_fastesm650.pth)'
    )
    parser.add_argument(
        '--save-best',
        action='store_true',
        help='Save best validation model instead of final model'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='training_log_fastesm650.txt',
        help='Training log file path (default: training_log_fastesm650.txt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=12,
        help='Number of DataLoader workers (default: 12)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split from training data (default: 0.1)'
    )

    args = parser.parse_args()

    # Set all seeds for reproducibility
    print(f"\n=== Setting random seed: {args.seed} ===")
    set_all_seeds(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata
    print(f"\n=== Loading metadata from {args.metadata} ===")
    metadata = load_metadata(args.metadata)

    train_files = metadata['train_files']
    test_files = metadata['test_files']

    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")

    # Create dataset from training files only
    print("\n=== Creating datasets ===")
    full_train_dataset = FileBatchDataset(train_files)
    test_dataset = FileBatchDataset(test_files)

    print(f"Full training dataset size: {len(full_train_dataset)} sequences")
    print(f"Test dataset size: {len(test_dataset)} sequences")

    # Create validation split from training data
    total_train_size = len(full_train_dataset)
    indices = list(range(total_train_size))
    random.shuffle(indices)

    val_size = int(total_train_size * args.val_split)
    train_size = total_train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    print(f"Training split: {len(train_dataset)} sequences ({(1-args.val_split)*100:.0f}%)")
    print(f"Validation split: {len(val_dataset)} sequences ({args.val_split*100:.0f}%)")

    # Create DataLoaders with reproducibility
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Validate input dimensions
    validate_input_dimensions(train_loader, device, MERGED_DIM)

    # Create model
    print(f"\n=== Creating MLPClassifier ===")
    print(f"Input dim: {MERGED_DIM} (DNA: {DNA_DIM} + Protein: {PROTEIN_DIM})")
    print(f"Hidden dim: 512")
    print(f"Num classes: 2")

    model = MLPClassifier(input_dim=MERGED_DIM, hidden_dim=512, num_class=2)
    model = model.to(device)

    # Create optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.85)
    criterion = nn.CrossEntropyLoss()

    print(f"Optimizer: SGD(lr={args.lr}, momentum=0.9)")
    print(f"Scheduler: StepLR(step_size=10, gamma=0.85)")

    # Train model
    print(f"\n=== Starting training ===")
    best_epoch, best_val_loss = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        log_file=args.log_file,
        patience=args.patience,
        device=device,
        save_best=args.save_best
    )

    # Final evaluation on test set
    print(f"\n=== Final evaluation on test set ===")
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate_model(
        model, test_loader, criterion, device
    )

    test_results = (
        f"Test Loss: {test_loss:.4f}\n"
        f"Test Accuracy: {test_accuracy:.4f}\n"
        f"Test Precision: {test_precision:.4f}\n"
        f"Test Recall: {test_recall:.4f}\n"
        f"Test F1 Score: {test_f1:.4f}\n"
        f"Test AUROC: {test_auc:.4f}\n"
    )
    print(test_results)
    write_log(args.log_file, f"\n{'='*80}\n")
    write_log(args.log_file, "FINAL TEST SET EVALUATION\n")
    write_log(args.log_file, test_results)
    write_log(args.log_file, f"{'='*80}\n")

    # Save checkpoint
    print(f"\n=== Saving checkpoint to {args.output} ===")
    save_checkpoint_with_metadata(
        model=model,
        optimizer=optimizer,
        epoch=best_epoch,
        best_loss=best_val_loss,
        filepath=args.output
    )

    print("\n=== Training complete ===")


if __name__ == '__main__':
    main()
