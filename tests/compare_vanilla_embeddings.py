"""
Compare embeddings between vanilla and refactored VirNucPro implementations.

Usage:
    python tests/compare_vanilla_embeddings.py <vanilla_dir> <refactored_dir>

Example:
    python tests/compare_vanilla_embeddings.py \\
        tests/data/reference_vanilla_output \\
        tests/data/test_with_orfs_output/test_with_orfs_nucleotide
"""

import torch
import numpy as np
from pathlib import Path
import sys


def compare_embeddings(vanilla_file, refactored_file, tolerance_rtol=1e-4, tolerance_atol=1e-6):
    """
    Compare embeddings from vanilla vs refactored implementation.

    Args:
        vanilla_file: Path to vanilla .pt file
        refactored_file: Path to refactored .pt file
        tolerance_rtol: Relative tolerance (default 1e-4 = 0.01%)
        tolerance_atol: Absolute tolerance (default 1e-6)

    Returns:
        dict with comparison results
    """
    vanilla = torch.load(vanilla_file)
    refactored = torch.load(refactored_file)

    # Get sequence IDs from whichever key exists
    vanilla_ids = vanilla.get('nucleotide', vanilla.get('proteins', vanilla.get('ids')))
    refactored_ids = refactored.get('nucleotide', refactored.get('proteins', refactored.get('ids')))

    # Check if same set of IDs (order may differ)
    vanilla_id_set = set(vanilla_ids) if vanilla_ids else set()
    refactored_id_set = set(refactored_ids) if refactored_ids else set()

    results = {
        'vanilla_file': str(vanilla_file),
        'refactored_file': str(refactored_file),
        'ids_match': vanilla_id_set == refactored_id_set,
        'num_sequences': len(vanilla_id_set),
        'mismatches': []
    }

    # Build dictionaries for ID-based lookup
    vanilla_dict = {}
    for i, seq_id in enumerate(vanilla_ids):
        vanilla_dict[seq_id] = vanilla['data'][i]

    refactored_dict = {}
    for i, seq_id in enumerate(refactored_ids):
        refactored_dict[seq_id] = refactored['data'][i]

    # Compare embeddings by ID
    for seq_id in vanilla_ids:
        if seq_id not in refactored_dict:
            continue  # Skip if not in refactored

        v_data = vanilla_dict[seq_id]
        r_data = refactored_dict[seq_id]

        # Handle different formats (dict vs tensor)
        if isinstance(v_data, dict):
            v_emb = torch.tensor(v_data['mean_representation'])
            r_emb = torch.tensor(r_data['mean_representation'])
        else:
            v_emb = v_data
            r_emb = r_data

        # Check if embeddings are close
        close = torch.allclose(v_emb, r_emb, rtol=tolerance_rtol, atol=tolerance_atol)

        if not close:
            # Calculate actual difference
            abs_diff = torch.abs(v_emb - r_emb)
            rel_diff = abs_diff / (torch.abs(v_emb) + 1e-10)

            results['mismatches'].append({
                'id': seq_id,
                'max_abs_diff': abs_diff.max().item(),
                'max_rel_diff': rel_diff.max().item(),
                'mean_abs_diff': abs_diff.mean().item(),
                'embedding_dim': len(v_emb)
            })

    results['all_match'] = len(results['mismatches']) == 0
    results['match_rate'] = (results['num_sequences'] - len(results['mismatches'])) / results['num_sequences'] if results['num_sequences'] > 0 else 0

    return results


def compare_all_embeddings(vanilla_dir, refactored_dir):
    """
    Compare all embedding files between vanilla and refactored outputs.
    """
    vanilla_dir = Path(vanilla_dir)
    refactored_dir = Path(refactored_dir)

    print("=" * 80)
    print("Vanilla vs Refactored Embedding Comparison")
    print("=" * 80)
    print(f"Vanilla dir:    {vanilla_dir}")
    print(f"Refactored dir: {refactored_dir}")
    print()

    # Find all .pt files
    vanilla_files = sorted(vanilla_dir.glob("*.pt"))

    if not vanilla_files:
        print(f"❌ No .pt files found in vanilla directory: {vanilla_dir}")
        return {}

    all_results = {}

    for v_file in vanilla_files:
        r_file = refactored_dir / v_file.name

        if not r_file.exists():
            print(f"\n⚠️  {v_file.name}: Missing in refactored output")
            continue

        print(f"\n📊 Comparing: {v_file.name}")
        results = compare_embeddings(v_file, r_file)
        all_results[v_file.name] = results

        if results['all_match']:
            print(f"   ✅ Perfect match! ({results['num_sequences']} sequences)")
        else:
            print(f"   ❌ Mismatches: {len(results['mismatches'])}/{results['num_sequences']} sequences")
            print(f"   Match rate: {results['match_rate']*100:.2f}%")

            # Show first 3 mismatches
            for m in results['mismatches'][:3]:
                print(f"      - {m['id']}: max_abs_diff={m['max_abs_diff']:.2e}, "
                      f"max_rel_diff={m['max_rel_diff']:.2e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    perfect_matches = sum(1 for r in all_results.values() if r['all_match'])
    total_files = len(all_results)

    print(f"Perfect matches: {perfect_matches}/{total_files} files")

    if perfect_matches == total_files:
        print("\n🎉 All embeddings match! Refactored implementation is mathematically equivalent.")
        return all_results
    else:
        print("\n⚠️  Some embeddings differ. Review mismatches above.")
        print("\nTolerance used: rtol=1e-4 (0.01%), atol=1e-6")
        print("Differences within tolerance are acceptable for floating-point operations.")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    vanilla_dir = sys.argv[1]
    refactored_dir = sys.argv[2]

    results = compare_all_embeddings(vanilla_dir, refactored_dir)

    # Exit code: 0 if all match, 1 if any mismatches
    all_match = all(r['all_match'] for r in results.values())
    sys.exit(0 if all_match else 1)
