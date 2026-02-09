#!/usr/bin/env python3
"""
Evaluation and comparison script for FastESM2 vs ESM2 3B baseline.

Evaluates the trained FastESM2 model on the test set, calculates comprehensive metrics
(F1, Accuracy, Precision, Recall, ROC-AUC, confusion matrix), and optionally compares
against ESM2 3B baseline with strict threshold validation (<5% accuracy drop).

Generates both terminal summary and markdown validation report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Import dimension constants and error handling
from units import DimensionError, DNA_DIM, PROTEIN_DIM, MERGED_DIM, CHECKPOINT_VERSION


class FileBatchDataset(Dataset):
    """Dataset that loads .pt files containing merged features."""

    def __init__(self, file_list):
        self.file_list = file_list
        self.data = []
        self.labels = []
        self._load_all_data()

    def _load_all_data(self):
        for file_path in self.file_list:
            data_dict = torch.load(file_path, weights_only=False)
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
    """MLP classifier with dimension validation."""

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


def load_checkpoint_with_validation(checkpoint_path):
    """
    Load checkpoint with validation to reject old ESM2 3B checkpoints.

    Args:
        checkpoint_path: path to checkpoint file

    Returns:
        checkpoint dict with validated metadata

    Raises:
        ValueError: if checkpoint is from ESM2 3B pipeline (no metadata or version 1.x)
        DimensionError: if checkpoint dimensions don't match expected values
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Check for metadata key
    if 'metadata' not in checkpoint:
        raise ValueError(
            f"This checkpoint is from ESM2 3B pipeline (no metadata found). "
            f"Re-extract features with FastESM2_650 and retrain. "
            f"Old checkpoint: {checkpoint_path}"
        )

    metadata = checkpoint['metadata']

    # Check checkpoint version
    version = metadata.get('checkpoint_version', '0.0.0')
    try:
        major_version = int(version.split('.')[0])
    except (ValueError, AttributeError):
        major_version = 0

    if major_version < 2:
        raise ValueError(
            f"This checkpoint uses ESM2 3B (2560-dim, version {version}). "
            f"Re-extract features with FastESM2_650 and retrain. "
            f"Incompatible checkpoint: {checkpoint_path}"
        )

    # Validate merged_dim
    merged_dim = metadata.get('merged_dim')
    if merged_dim != MERGED_DIM:
        raise DimensionError(
            expected_dim=MERGED_DIM,
            actual_dim=merged_dim,
            tensor_name="checkpoint_merged_dim",
            location="load_checkpoint_with_validation()"
        )

    # Print checkpoint info
    print(f"Loaded checkpoint:")
    print(f"  Version: {version}")
    print(f"  Model type: {metadata.get('model_type', 'unknown')}")
    print(f"  DNA dim: {metadata.get('dna_dim')}")
    print(f"  Protein dim: {metadata.get('protein_dim')}")
    print(f"  Merged dim: {metadata.get('merged_dim')}")
    print(f"  Training date: {metadata.get('training_date', 'unknown')}")

    return checkpoint


def load_test_metadata(metadata_path):
    """Load test metadata JSON file containing test file list."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata.get('test_files', [])


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set and calculate all metrics.

    Returns:
        dict containing all metrics and predictions
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability for viral class

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'labels': all_labels,
        'predictions': all_predictions
    }


def load_baseline_metrics(baseline_arg):
    """
    Load baseline metrics from JSON file or inline JSON string.

    Args:
        baseline_arg: path to JSON file or inline JSON string

    Returns:
        dict of baseline metrics or None if not provided
    """
    if not baseline_arg:
        return None

    # Try to load as file first
    if os.path.exists(baseline_arg):
        with open(baseline_arg, 'r') as f:
            return json.load(f)

    # Try to parse as inline JSON
    try:
        return json.loads(baseline_arg)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse baseline metrics from: {baseline_arg}")
        print("Provide either a path to JSON file or valid JSON string")
        return None


def validate_threshold(fastesm_metrics, baseline_metrics, threshold):
    """
    Validate that accuracy drop is within acceptable threshold.

    Returns:
        tuple of (passed: bool, difference: float, message: str)
    """
    if baseline_metrics is None:
        return None, None, "Threshold validation skipped (no baseline provided)"

    baseline_acc = baseline_metrics.get('accuracy')
    if baseline_acc is None:
        return None, None, "Threshold validation skipped (baseline missing 'accuracy')"

    fastesm_acc = fastesm_metrics['accuracy']
    difference = fastesm_acc - baseline_acc

    passed = (difference >= -threshold)

    if passed:
        message = f"PASSED - Accuracy drop {abs(difference)*100:.2f}% is within {threshold*100:.0f}% threshold"
    else:
        message = f"FAILED - Accuracy drop {abs(difference)*100:.2f}% exceeds {threshold*100:.0f}% threshold"

    return passed, difference, message


def generate_markdown_report(fastesm_metrics, baseline_metrics, threshold_result,
                              test_file_count, viral_count, host_count, report_path):
    """Generate comprehensive markdown validation report."""

    with open(report_path, 'w') as f:
        # Header
        f.write("# FastESM2_650 Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"**Model:** FastESM2_650 (1280-dim protein embeddings)\n")
        f.write(f"**Feature dimensions:** DNA={DNA_DIM}, Protein={PROTEIN_DIM}, Merged={MERGED_DIM}\n\n")

        # Test set info
        f.write("## Test Set Information\n\n")
        f.write(f"- Total samples: {viral_count + host_count}\n")
        f.write(f"- Viral sequences: {viral_count}\n")
        f.write(f"- Non-viral (host) sequences: {host_count}\n")
        f.write(f"- Test files processed: {test_file_count}\n\n")

        # Metrics table
        f.write("## Performance Metrics\n\n")

        if baseline_metrics:
            # Comparison table
            f.write("### FastESM2 vs ESM2 3B Baseline\n\n")
            df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                'FastESM2_650': [
                    f"{fastesm_metrics['accuracy']:.4f}",
                    f"{fastesm_metrics['precision']:.4f}",
                    f"{fastesm_metrics['recall']:.4f}",
                    f"{fastesm_metrics['f1']:.4f}",
                    f"{fastesm_metrics['roc_auc']:.4f}"
                ],
                'ESM2_3B_Baseline': [
                    f"{baseline_metrics.get('accuracy', 'N/A'):.4f}" if isinstance(baseline_metrics.get('accuracy'), float) else 'N/A',
                    f"{baseline_metrics.get('precision', 'N/A'):.4f}" if isinstance(baseline_metrics.get('precision'), float) else 'N/A',
                    f"{baseline_metrics.get('recall', 'N/A'):.4f}" if isinstance(baseline_metrics.get('recall'), float) else 'N/A',
                    f"{baseline_metrics.get('f1', 'N/A'):.4f}" if isinstance(baseline_metrics.get('f1'), float) else 'N/A',
                    f"{baseline_metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(baseline_metrics.get('roc_auc'), float) else 'N/A'
                ]
            })

            # Calculate differences for metrics that exist in baseline
            differences = []
            for metric_key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                if isinstance(baseline_metrics.get(metric_key), float):
                    diff = fastesm_metrics[metric_key] - baseline_metrics[metric_key]
                    differences.append(f"{diff:+.4f}")
                else:
                    differences.append('N/A')

            df['Difference'] = differences
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
        else:
            # FastESM2-only table
            f.write("### FastESM2_650 Performance\n\n")
            f.write("_Note: ESM2 3B baseline metrics not provided. Old model features were overwritten during Phase 4 re-extraction._\n\n")
            df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                'FastESM2_650': [
                    f"{fastesm_metrics['accuracy']:.4f}",
                    f"{fastesm_metrics['precision']:.4f}",
                    f"{fastesm_metrics['recall']:.4f}",
                    f"{fastesm_metrics['f1']:.4f}",
                    f"{fastesm_metrics['roc_auc']:.4f}"
                ]
            })
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

        # Confusion matrix
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        conf_matrix = fastesm_metrics['confusion_matrix']
        f.write(f"                 Predicted\n")
        f.write(f"                 Non-Viral  Viral\n")
        f.write(f"Actual Non-Viral    {conf_matrix[0][0]:6d}   {conf_matrix[0][1]:6d}\n")
        f.write(f"       Viral        {conf_matrix[1][0]:6d}   {conf_matrix[1][1]:6d}\n")
        f.write("```\n\n")

        # Threshold validation
        f.write("## Threshold Validation\n\n")
        if threshold_result[0] is None:
            f.write(f"**Status:** SKIPPED\n\n")
            f.write(f"**Reason:** {threshold_result[2]}\n\n")
            f.write("To enable threshold validation, provide baseline metrics via:\n")
            f.write("```bash\n")
            f.write("python scripts/evaluate_compare.py --baseline-metrics baseline_metrics.json\n")
            f.write("```\n\n")
        elif threshold_result[0]:
            f.write(f"**Status:** ✅ PASSED\n\n")
            f.write(f"**Result:** {threshold_result[2]}\n\n")
            f.write(f"FastESM2_650 maintains performance within acceptable bounds. Ready for deployment.\n\n")
        else:
            f.write(f"**Status:** ❌ FAILED\n\n")
            f.write(f"**Result:** {threshold_result[2]}\n\n")
            f.write("### DEPLOYMENT HALTED\n\n")
            f.write("FastESM2_650 accuracy drop exceeds the 5% threshold. Suggested next steps:\n\n")
            f.write("1. **Increase training epochs** - Model may need more training time\n")
            f.write("2. **Adjust learning rate** - Try different learning rate schedules\n")
            f.write("3. **Investigate data quality** - Verify FastESM2 embeddings match expected dimensions\n")
            f.write("4. **Consider fine-tuning FastESM2** - Fine-tune the embedding model on viral sequences\n")
            f.write("5. **Hyperparameter tuning** - Experiment with hidden layer size, dropout rate\n\n")

        f.write("---\n")
        f.write("_Report generated by scripts/evaluate_compare.py_\n")

    print(f"\nFull report saved to: {report_path}")


def print_terminal_summary(fastesm_metrics, baseline_metrics, threshold_result):
    """Print comprehensive terminal summary."""

    print("\n" + "="*70)
    print("FastESM2_650 EVALUATION SUMMARY")
    print("="*70)

    print(f"\nFastESM2_650 Metrics:")
    print(f"  Accuracy:  {fastesm_metrics['accuracy']:.4f}")
    print(f"  Precision: {fastesm_metrics['precision']:.4f}")
    print(f"  Recall:    {fastesm_metrics['recall']:.4f}")
    print(f"  F1 Score:  {fastesm_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {fastesm_metrics['roc_auc']:.4f}")

    if baseline_metrics:
        print(f"\nESM2 3B Baseline:")
        print(f"  Accuracy:  {baseline_metrics.get('accuracy', 'N/A'):.4f}" if isinstance(baseline_metrics.get('accuracy'), float) else "  Accuracy:  N/A")
        print(f"  Precision: {baseline_metrics.get('precision', 'N/A'):.4f}" if isinstance(baseline_metrics.get('precision'), float) else "  Precision: N/A")
        print(f"  Recall:    {baseline_metrics.get('recall', 'N/A'):.4f}" if isinstance(baseline_metrics.get('recall'), float) else "  Recall:    N/A")
        print(f"  F1 Score:  {baseline_metrics.get('f1', 'N/A'):.4f}" if isinstance(baseline_metrics.get('f1'), float) else "  F1 Score:  N/A")
        print(f"  ROC-AUC:   {baseline_metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(baseline_metrics.get('roc_auc'), float) else "  ROC-AUC:   N/A")

    print(f"\nThreshold Validation:")
    if threshold_result[0] is None:
        print(f"  Status: SKIPPED - {threshold_result[2]}")
    elif threshold_result[0]:
        print(f"  Status: ✅ PASSED")
        print(f"  {threshold_result[2]}")
    else:
        print(f"  Status: ❌ FAILED")
        print(f"  {threshold_result[2]}")
        print("\n  DEPLOYMENT HALTED - See report for suggested next steps")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FastESM2 model and compare against ESM2 3B baseline"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='model_fastesm650.pth',
        help='Path to FastESM2 checkpoint (default: model_fastesm650.pth)'
    )
    parser.add_argument(
        '--test-metadata',
        type=str,
        default='./data/test_set/test_metadata.json',
        help='Path to test metadata JSON (default: ./data/test_set/test_metadata.json)'
    )
    parser.add_argument(
        '--baseline-metrics',
        type=str,
        default=None,
        help='Path to JSON file or inline JSON with ESM2 3B baseline metrics (optional)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Maximum acceptable accuracy drop (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        default='./reports/',
        help='Output directory for validation report (default: ./reports/)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Evaluation batch size (default: 256)'
    )

    args = parser.parse_args()

    # Create report directory
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, 'validation_report.md')

    # Load baseline metrics if provided
    baseline_metrics = load_baseline_metrics(args.baseline_metrics)
    if args.baseline_metrics and baseline_metrics is None:
        print("ERROR: Failed to load baseline metrics. Exiting.")
        sys.exit(1)

    if baseline_metrics is None:
        print("\n" + "="*70)
        print("NOTE: ESM2 3B baseline metrics not provided.")
        print("="*70)
        print("\nOld model features were overwritten during Phase 4 re-extraction,")
        print("so direct re-evaluation is not possible.")
        print("\nTo compare against baseline, provide historical metrics via:")
        print(f"  --baseline-metrics <json_file_or_string>")
        print("\nExample:")
        print('  --baseline-metrics \'{"accuracy": 0.95, "f1": 0.94}\'')
        print("\nReporting FastESM2 metrics only.")
        print("="*70 + "\n")

    # Load test metadata
    print(f"Loading test metadata from: {args.test_metadata}")
    if not os.path.exists(args.test_metadata):
        print(f"ERROR: Test metadata file not found: {args.test_metadata}")
        print("\nTest metadata should be created during model training (Plan 05-01)")
        print("or manually created with format:")
        print('  {"test_files": ["path/to/file1.pt", "path/to/file2.pt", ...]}')
        sys.exit(1)

    test_files = load_test_metadata(args.test_metadata)
    if not test_files:
        print("ERROR: No test files found in metadata")
        sys.exit(1)

    print(f"Found {len(test_files)} test files")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = FileBatchDataset(test_files)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Calculate test set statistics
    total_samples = len(test_dataset)
    viral_count = sum(1 for _, label in test_dataset if label == 1)
    host_count = total_samples - viral_count

    print(f"Test set loaded: {total_samples} samples ({viral_count} viral, {host_count} host)")

    # Load model
    print(f"\nLoading model from: {args.model}")
    checkpoint = load_checkpoint_with_validation(args.model)

    model = MLPClassifier(
        input_dim=checkpoint['metadata']['input_dim'],
        hidden_dim=checkpoint['metadata']['hidden_dim'],
        num_class=checkpoint['metadata']['num_class']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Evaluate model
    print("\nEvaluating model on test set...")
    fastesm_metrics = evaluate_model(model, test_loader, device)

    # Threshold validation
    threshold_result = validate_threshold(fastesm_metrics, baseline_metrics, args.threshold)

    # Generate markdown report
    generate_markdown_report(
        fastesm_metrics,
        baseline_metrics,
        threshold_result,
        len(test_files),
        viral_count,
        host_count,
        report_path
    )

    # Print terminal summary
    print_terminal_summary(fastesm_metrics, baseline_metrics, threshold_result)
    print(f"\nFull report: {report_path}\n")


if __name__ == '__main__':
    main()
