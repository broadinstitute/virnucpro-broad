#!/usr/bin/env python3
"""
Run predictions on test set and save detailed results for comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from tqdm import tqdm
import argparse

from units import MERGED_DIM, CHECKPOINT_VERSION

# Duplicate MLPClassifier from train.py to match actual saved model architecture
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
    """Load trained model with validation."""
    # weights_only=False needed for PyTorch 2.6+ to load older checkpoints
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='300_model_fastesm650.pth', help='Model checkpoint')
    parser.add_argument('--test-metadata', default='data/test_set/test_metadata.json', help='Test metadata')
    parser.add_argument('--output', default='prediction_results_fastesm650.txt', help='Output file')
    parser.add_argument('--output-csv', default='prediction_results_fastesm650_highestscore.csv', help='Output CSV')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    
    # Load test metadata
    with open(args.test_metadata) as f:
        metadata = json.load(f)
    
    test_files = metadata['test_files']
    print(f"Found {len(test_files)} test files")
    
    # Run predictions
    all_predictions = []
    
    for test_file in tqdm(test_files, desc="Processing test files"):
        data = torch.load(test_file, map_location='cpu', weights_only=False)
        
        # Handle symlinks - data might be in different formats
        if isinstance(data, dict) and 'data' in data:
            # Merged format: {data: tensor, labels: tensor, ids: list}
            features = data['data']
            labels = data['labels']
            seq_ids = data.get('ids', [f"seq_{i}" for i in range(len(features))])
        else:
            print(f"Unexpected format in {test_file}, skipping")
            continue
        
        # Determine if viral or non-viral from path
        is_viral = 'viral' in test_file
        
        # Run predictions
        features = features.to(device)
        with torch.no_grad():
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # Save results
        probs_cpu = probs.cpu().numpy()
        preds_cpu = predictions.cpu().numpy()
        
        for i, seq_id in enumerate(seq_ids):
            pred_label = 'virus' if preds_cpu[i] == 1 else 'others'
            score_others = probs_cpu[i][0]
            score_virus = probs_cpu[i][1]
            
            all_predictions.append({
                'seq_id': seq_id,
                'prediction': pred_label,
                'score_others': score_others,
                'score_virus': score_virus,
                'true_label': 'viral' if is_viral else 'non-viral'
            })
    
    # Write detailed results
    print(f"\nWriting {len(all_predictions)} predictions to {args.output}...")
    with open(args.output, 'w') as f:
        f.write("Sequence_ID\tPrediction\tscore1\tscore2\n")
        for pred in all_predictions:
            f.write(f"{pred['seq_id']}\t{pred['prediction']}\t{pred['score_others']:.8f}\t{pred['score_virus']:.8f}\n")
    
    # Write CSV (highest score per sequence)
    # Group by base sequence ID (remove chunk/ORF info)
    from collections import defaultdict
    seq_votes = defaultdict(lambda: {'virus': 0, 'others': 0})
    
    for pred in all_predictions:
        # Extract base sequence ID
        base_id = pred['seq_id'].split('_chunk_')[0] if '_chunk_' in pred['seq_id'] else pred['seq_id']
        seq_votes[base_id][pred['prediction']] += 1
    
    print(f"Writing aggregated results to {args.output_csv}...")
    with open(args.output_csv, 'w') as f:
        f.write("Modified_ID\tIs_Virus\n")
        for seq_id, votes in seq_votes.items():
            is_virus = votes['virus'] > votes['others']
            f.write(f"{seq_id}\t{is_virus}\n")
    
    print("Done!")

if __name__ == '__main__':
    main()
