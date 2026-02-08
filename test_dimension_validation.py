#!/usr/bin/env python
"""
Test script for dimension validation functionality added in Phase 3, Plan 01.
Run this inside the docker container: docker compose exec virnucpro python test_dimension_validation.py
"""

import torch
import tempfile
import os

from units import (
    DimensionError, DNA_DIM, PROTEIN_DIM, MERGED_DIM, VALIDATE_DIMS,
    validate_merge_inputs, validate_merged_output, validate_protein_embeddings,
    merge_data
)

def test_constants_and_imports():
    """Test 1: Verify constants and DimensionError class are importable"""
    print("=" * 80)
    print("TEST 1: Constants and Imports")
    print("=" * 80)

    print(f'DNA={DNA_DIM}, PROTEIN={PROTEIN_DIM}, MERGED={MERGED_DIM}, VALIDATE={VALIDATE_DIMS}')
    print('DimensionError:', DimensionError.__name__)

    assert DNA_DIM == 768, f"Expected DNA_DIM=768, got {DNA_DIM}"
    assert PROTEIN_DIM == 1280, f"Expected PROTEIN_DIM=1280, got {PROTEIN_DIM}"
    assert MERGED_DIM == 2048, f"Expected MERGED_DIM=2048, got {MERGED_DIM}"
    assert VALIDATE_DIMS == True, f"Expected VALIDATE_DIMS=True, got {VALIDATE_DIMS}"

    print("✓ All constants correct")
    print()


def test_dimension_error_attributes():
    """Test 2: Verify DimensionError attributes"""
    print("=" * 80)
    print("TEST 2: DimensionError Attributes")
    print("=" * 80)

    e = DimensionError(1280, 2560, 'test_tensor', 'test_location')
    print(str(e))

    assert e.expected_dim == 1280
    assert e.actual_dim == 2560
    assert e.tensor_name == 'test_tensor'
    assert e.location == 'test_location'

    print('✓ DimensionError attributes OK')
    print()


def test_merge_data_correct_dimensions():
    """Test 3: merge_data() with CORRECT dimensions produces 2048-dim output"""
    print("=" * 80)
    print("TEST 3: merge_data() with Correct Dimensions")
    print("=" * 80)

    # Create test data with CORRECT dimensions
    dna_data = {
        'nucleotide': ['seq1', 'seq2'],
        'data': [
            {'mean_representation': torch.randn(DNA_DIM).tolist()},
            {'mean_representation': torch.randn(DNA_DIM).tolist()}
        ]
    }
    protein_data = {
        'proteins': ['seq1', 'seq2'],
        'data': [torch.randn(PROTEIN_DIM), torch.randn(PROTEIN_DIM)]
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f1:
        torch.save(dna_data, f1.name)
        dna_file = f1.name

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f2:
        torch.save(protein_data, f2.name)
        protein_file = f2.name

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f3:
        merged_file = f3.name

    try:
        merge_data(dna_file, protein_file, merged_file)
        result = torch.load(merged_file)

        assert result['data'].shape[1] == MERGED_DIM, f'Expected {MERGED_DIM}, got {result["data"].shape[1]}'
        print(f'✓ merge_data() produces {result["data"].shape[1]}-dim features - CORRECT')

    finally:
        os.unlink(dna_file)
        os.unlink(protein_file)
        os.unlink(merged_file)

    print()


def test_merge_data_dimension_mismatch():
    """Test 4: merge_data() with WRONG dimensions raises DimensionError"""
    print("=" * 80)
    print("TEST 4: merge_data() Dimension Mismatch Detection")
    print("=" * 80)

    # Create test data with WRONG protein dimensions (old ESM2 3B size)
    dna_data = {
        'nucleotide': ['seq1'],
        'data': [{'mean_representation': torch.randn(DNA_DIM).tolist()}]
    }
    protein_data = {
        'proteins': ['seq1'],
        'data': [torch.randn(2560)]  # Wrong dim!
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f1:
        torch.save(dna_data, f1.name)
        dna_file = f1.name

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f2:
        torch.save(protein_data, f2.name)
        protein_file = f2.name

    try:
        merge_data(dna_file, protein_file, '/tmp/should_not_exist.pt')
        print('ERROR: Should have raised DimensionError!')
        assert False, "Expected DimensionError was not raised"

    except DimensionError as e:
        print(f'✓ Correctly caught dimension mismatch: {e}')
        assert e.expected_dim == 1280
        assert e.actual_dim == 2560
        print('✓ DimensionError has correct expected_dim and actual_dim')

    finally:
        os.unlink(dna_file)
        os.unlink(protein_file)
        if os.path.exists('/tmp/should_not_exist.pt'):
            os.unlink('/tmp/should_not_exist.pt')

    print()


def main():
    print("\n")
    print("=" * 80)
    print("Phase 3, Plan 01: Dimension Validation Tests")
    print("=" * 80)
    print()

    try:
        test_constants_and_imports()
        test_dimension_error_attributes()
        test_merge_data_correct_dimensions()
        test_merge_data_dimension_mismatch()

        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()

    except Exception as e:
        print("=" * 80)
        print(f"TEST FAILED ✗")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
