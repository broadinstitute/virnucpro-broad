#!/usr/bin/env python3
"""
Compare VirNucPro prediction outputs between implementations.

This script compares prediction results from two VirNucPro runs (e.g., vanilla vs refactored)
to validate that optimizations produce equivalent results.

Supports two output formats:
1. Per-frame predictions: prediction_results.txt (seq_id, prediction, score1, score2)
2. Consensus predictions: prediction_results_highestscore.csv (Modified_ID, Is_Virus, max_score_0, max_score_1)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np


class PredictionComparator:
    """Compare VirNucPro prediction outputs."""

    def __init__(self, score_tolerance: float = 1e-5):
        """
        Initialize comparator.

        Args:
            score_tolerance: Maximum allowed difference for float scores
        """
        self.score_tolerance = score_tolerance
        self.mismatches = []
        self.missing_in_ref = []
        self.missing_in_vanilla = []

    def load_per_frame_predictions(self, file_path: Path) -> pd.DataFrame:
        """
        Load per-frame prediction results.

        Args:
            file_path: Path to prediction_results.txt

        Returns:
            DataFrame with columns: Sequence_ID, Prediction, score1, score2
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {file_path}")

        df = pd.read_csv(file_path, sep='\t')

        # Validate expected columns
        expected_cols = {'Sequence_ID', 'Prediction', 'score1', 'score2'}
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"Expected columns {expected_cols}, got {set(df.columns)}")

        return df

    def load_consensus_predictions(self, file_path: Path) -> pd.DataFrame:
        """
        Load consensus prediction results.

        Args:
            file_path: Path to prediction_results_highestscore.csv

        Returns:
            DataFrame with columns: Modified_ID, Is_Virus, and optionally max_score_0, max_score_1
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Consensus file not found: {file_path}")

        df = pd.read_csv(file_path, sep='\t')

        # Validate required columns
        required_cols = {'Modified_ID', 'Is_Virus'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Expected columns {required_cols}, got {set(df.columns)}")

        return df

    def compare_per_frame(self, vanilla_df: pd.DataFrame, refactored_df: pd.DataFrame) -> Dict:
        """
        Compare per-frame prediction outputs.

        Args:
            vanilla_df: Predictions from vanilla implementation
            refactored_df: Predictions from refactored implementation

        Returns:
            Dictionary with comparison results and statistics
        """
        # Reset mismatch tracking
        self.mismatches = []
        self.missing_in_ref = []
        self.missing_in_vanilla = []

        # Index by Sequence_ID for fast lookup
        vanilla_dict = vanilla_df.set_index('Sequence_ID').to_dict('index')
        refactored_dict = refactored_df.set_index('Sequence_ID').to_dict('index')

        # Find all unique sequence IDs
        all_seq_ids = set(vanilla_dict.keys()) | set(refactored_dict.keys())

        # Track statistics
        total_sequences = len(all_seq_ids)
        matching_predictions = 0
        score_differences = []

        # Compare each sequence
        for seq_id in sorted(all_seq_ids):
            vanilla_pred = vanilla_dict.get(seq_id)
            refactored_pred = refactored_dict.get(seq_id)

            # Check for missing sequences
            if vanilla_pred is None:
                self.missing_in_vanilla.append(seq_id)
                continue
            if refactored_pred is None:
                self.missing_in_ref.append(seq_id)
                continue

            # Compare predictions
            vanilla_class = vanilla_pred['Prediction']
            refactored_class = refactored_pred['Prediction']

            vanilla_score1 = float(vanilla_pred['score1'])
            vanilla_score2 = float(vanilla_pred['score2'])
            refactored_score1 = float(refactored_pred['score1'])
            refactored_score2 = float(refactored_pred['score2'])

            # Check for prediction mismatch
            prediction_match = vanilla_class == refactored_class

            # Check for score differences
            score1_diff = abs(vanilla_score1 - refactored_score1)
            score2_diff = abs(vanilla_score2 - refactored_score2)
            max_score_diff = max(score1_diff, score2_diff)

            score_match = (score1_diff <= self.score_tolerance and
                          score2_diff <= self.score_tolerance)

            if prediction_match and score_match:
                matching_predictions += 1
            else:
                self.mismatches.append({
                    'seq_id': seq_id,
                    'vanilla_pred': vanilla_class,
                    'refactored_pred': refactored_class,
                    'vanilla_score1': vanilla_score1,
                    'vanilla_score2': vanilla_score2,
                    'refactored_score1': refactored_score1,
                    'refactored_score2': refactored_score2,
                    'score1_diff': score1_diff,
                    'score2_diff': score2_diff,
                    'max_diff': max_score_diff,
                    'prediction_mismatch': not prediction_match,
                    'score_mismatch': not score_match
                })

            # Track score differences for statistics
            score_differences.append(max_score_diff)

        # Calculate statistics
        score_diff_array = np.array(score_differences) if score_differences else np.array([0.0])

        return {
            'total_sequences': total_sequences,
            'compared_sequences': len(score_differences),
            'matching': matching_predictions,
            'mismatching': len(self.mismatches),
            'missing_in_refactored': len(self.missing_in_ref),
            'missing_in_vanilla': len(self.missing_in_vanilla),
            'match_percentage': (matching_predictions / len(score_differences) * 100) if score_differences else 0.0,
            'score_stats': {
                'mean_diff': float(np.mean(score_diff_array)),
                'median_diff': float(np.median(score_diff_array)),
                'max_diff': float(np.max(score_diff_array)),
                'min_diff': float(np.min(score_diff_array)),
                'std_diff': float(np.std(score_diff_array))
            }
        }

    def compare_consensus(self, vanilla_df: pd.DataFrame, refactored_df: pd.DataFrame) -> Dict:
        """
        Compare consensus prediction outputs.

        Handles files with or without score columns (max_score_0, max_score_1).
        When scores are missing from either file, only label agreement is compared.

        Args:
            vanilla_df: Consensus from vanilla implementation
            refactored_df: Consensus from refactored implementation

        Returns:
            Dictionary with comparison results and statistics
        """
        # Reset mismatch tracking
        self.mismatches = []
        self.missing_in_ref = []
        self.missing_in_vanilla = []

        # Detect whether score columns are available
        has_vanilla_scores = 'max_score_0' in vanilla_df.columns and 'max_score_1' in vanilla_df.columns
        has_refactored_scores = 'max_score_0' in refactored_df.columns and 'max_score_1' in refactored_df.columns
        has_scores = has_vanilla_scores and has_refactored_scores

        if not has_scores:
            missing = []
            if not has_vanilla_scores:
                missing.append("vanilla")
            if not has_refactored_scores:
                missing.append("refactored")
            print(f"Note: Score columns missing from {', '.join(missing)} file(s). Comparing labels only.")

        # Index by Modified_ID for fast lookup
        vanilla_dict = vanilla_df.set_index('Modified_ID').to_dict('index')
        refactored_dict = refactored_df.set_index('Modified_ID').to_dict('index')

        # Find all unique IDs
        all_ids = set(vanilla_dict.keys()) | set(refactored_dict.keys())

        # Track statistics
        total_sequences = len(all_ids)
        matching_predictions = 0
        score_differences = []

        # Compare each sequence
        for seq_id in sorted(all_ids):
            vanilla_pred = vanilla_dict.get(seq_id)
            refactored_pred = refactored_dict.get(seq_id)

            # Check for missing sequences
            if vanilla_pred is None:
                self.missing_in_vanilla.append(seq_id)
                continue
            if refactored_pred is None:
                self.missing_in_ref.append(seq_id)
                continue

            # Compare predictions (Is_Virus is boolean)
            vanilla_is_virus = bool(vanilla_pred['Is_Virus'])
            refactored_is_virus = bool(refactored_pred['Is_Virus'])

            prediction_match = vanilla_is_virus == refactored_is_virus

            if has_scores:
                vanilla_score0 = float(vanilla_pred['max_score_0'])
                vanilla_score1 = float(vanilla_pred['max_score_1'])
                refactored_score0 = float(refactored_pred['max_score_0'])
                refactored_score1 = float(refactored_pred['max_score_1'])

                score0_diff = abs(vanilla_score0 - refactored_score0)
                score1_diff = abs(vanilla_score1 - refactored_score1)
                max_score_diff = max(score0_diff, score1_diff)

                score_match = (score0_diff <= self.score_tolerance and
                              score1_diff <= self.score_tolerance)

                if prediction_match and score_match:
                    matching_predictions += 1
                else:
                    self.mismatches.append({
                        'seq_id': seq_id,
                        'vanilla_pred': 'virus' if vanilla_is_virus else 'others',
                        'refactored_pred': 'virus' if refactored_is_virus else 'others',
                        'vanilla_score0': vanilla_score0,
                        'vanilla_score1': vanilla_score1,
                        'refactored_score0': refactored_score0,
                        'refactored_score1': refactored_score1,
                        'score0_diff': score0_diff,
                        'score1_diff': score1_diff,
                        'max_diff': max_score_diff,
                        'prediction_mismatch': not prediction_match,
                        'score_mismatch': not score_match
                    })

                score_differences.append(max_score_diff)
            else:
                # Label-only comparison
                if prediction_match:
                    matching_predictions += 1
                else:
                    self.mismatches.append({
                        'seq_id': seq_id,
                        'vanilla_pred': 'virus' if vanilla_is_virus else 'others',
                        'refactored_pred': 'virus' if refactored_is_virus else 'others',
                        'max_diff': 0.0,
                        'prediction_mismatch': True,
                        'score_mismatch': False
                    })

        # Calculate statistics
        compared = matching_predictions + len(self.mismatches)

        result = {
            'total_sequences': total_sequences,
            'compared_sequences': compared,
            'matching': matching_predictions,
            'mismatching': len(self.mismatches),
            'missing_in_refactored': len(self.missing_in_ref),
            'missing_in_vanilla': len(self.missing_in_vanilla),
            'match_percentage': (matching_predictions / compared * 100) if compared else 0.0,
            'has_scores': has_scores,
        }

        if has_scores:
            score_diff_array = np.array(score_differences) if score_differences else np.array([0.0])
            result['score_stats'] = {
                'mean_diff': float(np.mean(score_diff_array)),
                'median_diff': float(np.median(score_diff_array)),
                'max_diff': float(np.max(score_diff_array)),
                'min_diff': float(np.min(score_diff_array)),
                'std_diff': float(np.std(score_diff_array))
            }
        else:
            result['score_stats'] = None

        return result

    def print_summary(self, results: Dict, mode: str):
        """Print comparison summary."""
        print(f"\n{'='*80}")
        print(f"VirNucPro Output Comparison - {mode.upper()} MODE")
        print(f"{'='*80}")
        print(f"\nScore tolerance: {self.score_tolerance}")
        print(f"\nTotal sequences: {results['total_sequences']}")
        print(f"Compared sequences: {results['compared_sequences']}")
        print(f"Missing in refactored: {results['missing_in_refactored']}")
        print(f"Missing in vanilla: {results['missing_in_vanilla']}")
        print(f"\nMatching predictions: {results['matching']} ({results['match_percentage']:.2f}%)")
        print(f"Mismatching predictions: {results['mismatching']}")

        if results.get('score_stats'):
            print(f"\nScore Difference Statistics:")
            print(f"  Mean: {results['score_stats']['mean_diff']:.6e}")
            print(f"  Median: {results['score_stats']['median_diff']:.6e}")
            print(f"  Max: {results['score_stats']['max_diff']:.6e}")
            print(f"  Min: {results['score_stats']['min_diff']:.6e}")
            print(f"  Std Dev: {results['score_stats']['std_diff']:.6e}")
        else:
            print(f"\nScore comparison: N/A (score columns not present in both files)")

        if results['mismatching'] == 0 and results['missing_in_refactored'] == 0 and results['missing_in_vanilla'] == 0:
            print(f"\n{'='*80}")
            print("RESULT: PERFECT MATCH - Implementations produce identical outputs!")
            print(f"{'='*80}\n")
        elif results['mismatching'] > 0:
            print(f"\n{'='*80}")
            print(f"RESULT: MISMATCHES FOUND - {results['mismatching']} sequences differ")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"RESULT: MISSING SEQUENCES - Check file completeness")
            print(f"{'='*80}\n")

    def print_detailed_mismatches(self, limit: int = 20):
        """Print detailed mismatch information."""
        if not self.mismatches:
            return

        print(f"\n{'='*80}")
        print(f"DETAILED MISMATCHES (showing first {min(limit, len(self.mismatches))} of {len(self.mismatches)})")
        print(f"{'='*80}\n")

        for i, mismatch in enumerate(self.mismatches[:limit], 1):
            print(f"{i}. Sequence: {mismatch['seq_id']}")

            if mismatch['prediction_mismatch']:
                print(f"   PREDICTION MISMATCH:")
                print(f"     Vanilla:    {mismatch['vanilla_pred']}")
                print(f"     Refactored: {mismatch['refactored_pred']}")

            if mismatch['score_mismatch']:
                print(f"   SCORE DIFFERENCES:")
                if 'score1' in mismatch:  # Per-frame format
                    print(f"     Score1: {mismatch['vanilla_score1']:.6f} vs {mismatch['refactored_score1']:.6f} (diff: {mismatch['score1_diff']:.6e})")
                    print(f"     Score2: {mismatch['vanilla_score2']:.6f} vs {mismatch['refactored_score2']:.6f} (diff: {mismatch['score2_diff']:.6e})")
                else:  # Consensus format
                    print(f"     Score0: {mismatch['vanilla_score0']:.6f} vs {mismatch['refactored_score0']:.6f} (diff: {mismatch['score0_diff']:.6e})")
                    print(f"     Score1: {mismatch['vanilla_score1']:.6f} vs {mismatch['refactored_score1']:.6f} (diff: {mismatch['score1_diff']:.6e})")

            print(f"   Max difference: {mismatch['max_diff']:.6e}")
            print()

    def print_missing_sequences(self):
        """Print information about missing sequences."""
        if self.missing_in_ref:
            print(f"\n{'='*80}")
            print(f"SEQUENCES MISSING IN REFACTORED ({len(self.missing_in_ref)})")
            print(f"{'='*80}\n")
            for seq_id in self.missing_in_ref[:20]:
                print(f"  - {seq_id}")
            if len(self.missing_in_ref) > 20:
                print(f"  ... and {len(self.missing_in_ref) - 20} more")

        if self.missing_in_vanilla:
            print(f"\n{'='*80}")
            print(f"SEQUENCES MISSING IN VANILLA ({len(self.missing_in_vanilla)})")
            print(f"{'='*80}\n")
            for seq_id in self.missing_in_vanilla[:20]:
                print(f"  - {seq_id}")
            if len(self.missing_in_vanilla) > 20:
                print(f"  ... and {len(self.missing_in_vanilla) - 20} more")

    def export_mismatches_csv(self, output_path: Path, mode: str):
        """Export mismatches to CSV file."""
        if not self.mismatches:
            print(f"No mismatches to export")
            return

        df = pd.DataFrame(self.mismatches)
        df.to_csv(output_path, index=False)
        print(f"\nExported {len(self.mismatches)} mismatches to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare VirNucPro prediction outputs between implementations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare per-frame predictions
  %(prog)s --vanilla vanilla_output/prediction_results.txt \\
           --refactored refactored_output/prediction_results.txt

  # Compare consensus predictions
  %(prog)s --vanilla vanilla_output/prediction_results_highestscore.csv \\
           --refactored refactored_output/prediction_results_highestscore.csv \\
           --mode consensus

  # Show detailed mismatches
  %(prog)s --vanilla vanilla.txt --refactored refactored.txt \\
           --output detailed

  # Export mismatches to CSV
  %(prog)s --vanilla vanilla.txt --refactored refactored.txt \\
           --export-csv mismatches.csv
        """
    )

    parser.add_argument(
        '--vanilla',
        type=Path,
        required=True,
        help='Path to vanilla implementation output file'
    )

    parser.add_argument(
        '--refactored',
        type=Path,
        required=True,
        help='Path to refactored implementation output file'
    )

    parser.add_argument(
        '--mode',
        choices=['per-frame', 'consensus'],
        default='per-frame',
        help='Comparison mode (default: per-frame)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-5,
        help='Score difference tolerance (default: 1e-5)'
    )

    parser.add_argument(
        '--output',
        choices=['summary', 'detailed'],
        default='summary',
        help='Output format (default: summary)'
    )

    parser.add_argument(
        '--export-csv',
        type=Path,
        help='Export mismatches to CSV file'
    )

    parser.add_argument(
        '--mismatch-limit',
        type=int,
        default=20,
        help='Maximum number of mismatches to display in detailed mode (default: 20)'
    )

    args = parser.parse_args()

    try:
        # Initialize comparator
        comparator = PredictionComparator(score_tolerance=args.tolerance)

        # Load data based on mode
        if args.mode == 'per-frame':
            vanilla_df = comparator.load_per_frame_predictions(args.vanilla)
            refactored_df = comparator.load_per_frame_predictions(args.refactored)
            results = comparator.compare_per_frame(vanilla_df, refactored_df)
        else:  # consensus
            vanilla_df = comparator.load_consensus_predictions(args.vanilla)
            refactored_df = comparator.load_consensus_predictions(args.refactored)
            results = comparator.compare_consensus(vanilla_df, refactored_df)

        # Print results
        comparator.print_summary(results, args.mode)

        if args.output == 'detailed':
            comparator.print_detailed_mismatches(limit=args.mismatch_limit)
            comparator.print_missing_sequences()

        # Export to CSV if requested
        if args.export_csv:
            comparator.export_mismatches_csv(args.export_csv, args.mode)

        # Exit with appropriate code
        if results['mismatching'] > 0 or results['missing_in_refactored'] > 0 or results['missing_in_vanilla'] > 0:
            sys.exit(1)  # Differences found
        else:
            sys.exit(0)  # Perfect match

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
