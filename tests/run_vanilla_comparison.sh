#!/bin/bash
# Complete vanilla vs refactored comparison workflow
#
# This script:
# 1. Checks if vanilla reference exists
# 2. Generates vanilla reference if needed
# 3. Runs comparison tests
# 4. Generates summary report

set -e  # Exit on error

echo "=========================================="
echo "Vanilla vs Refactored Comparison"
echo "=========================================="
echo ""

REFERENCE_DIR="tests/data/reference_vanilla_output"
GENERATE_SCRIPT="tests/generate_vanilla_reference.sh"

# Check if reference outputs exist
if [ ! -d "$REFERENCE_DIR" ] || [ -z "$(ls -A $REFERENCE_DIR 2>/dev/null)" ]; then
    echo "📋 Vanilla reference outputs not found"
    echo ""

    if [ -f "$GENERATE_SCRIPT" ]; then
        read -p "Generate vanilla reference now? [Y/n] " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            echo ""
            bash "$GENERATE_SCRIPT"
            echo ""
        else
            echo "❌ Cannot run comparison without vanilla reference"
            echo "   Run: ./tests/generate_vanilla_reference.sh"
            exit 1
        fi
    else
        echo "❌ Reference generation script not found: $GENERATE_SCRIPT"
        exit 1
    fi
else
    echo "✅ Vanilla reference outputs found in $REFERENCE_DIR"
    echo ""
    ls -lh "$REFERENCE_DIR"
    echo ""
fi

# Check for pytest
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Install with: pip install pytest"
    exit 1
fi

# Run comparison tests
echo "=========================================="
echo "Running Comparison Tests"
echo "=========================================="
echo ""

# Determine which tests to run
TEST_ARGS=""
if [ "$1" == "embeddings" ]; then
    TEST_ARGS="tests/test_vanilla_comparison.py::TestVanillaEquivalence::test_embeddings_equivalence"
    echo "Running embeddings comparison only..."
elif [ "$1" == "predictions" ]; then
    TEST_ARGS="tests/test_vanilla_comparison.py::TestVanillaEquivalence::test_prediction_output_equivalence"
    echo "Running predictions comparison only..."
elif [ "$1" == "full" ] || [ -z "$1" ]; then
    TEST_ARGS="tests/test_vanilla_comparison.py::TestVanillaEquivalence::test_full_pipeline_equivalence"
    echo "Running full pipeline comparison..."
else
    echo "❌ Invalid argument: $1"
    echo "Usage: $0 [embeddings|predictions|full]"
    exit 1
fi

echo ""

# Run tests with verbose output
if pytest "$TEST_ARGS" -v -s; then
    echo ""
    echo "=========================================="
    echo "✅ Comparison Tests PASSED"
    echo "=========================================="
    echo ""
    echo "Refactored implementation produces mathematically"
    echo "equivalent results to vanilla implementation."
    echo ""

    # Offer to save as golden reference
    if [ ! -d "tests/data/golden_reference" ]; then
        echo "💡 Tip: Save current outputs as golden reference for regression testing:"
        echo "   mkdir -p tests/data/golden_reference"
        echo "   cp -r tests/data/test_with_orfs_output/* tests/data/golden_reference/"
        echo ""
    fi

    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ Comparison Tests FAILED"
    echo "=========================================="
    echo ""
    echo "Review the test output above for details on mismatches."
    echo ""
    echo "Common troubleshooting steps:"
    echo "  1. Run comparison script directly for detailed output:"
    echo "     python tests/compare_vanilla_embeddings.py \\"
    echo "       tests/data/reference_vanilla_output \\"
    echo "       tests/data/test_with_orfs_output/test_with_orfs_nucleotide"
    echo ""
    echo "  2. Check if model checkpoints are identical"
    echo "  3. Verify input files are the same"
    echo "  4. Review tolerance settings in test_vanilla_comparison.py"
    echo ""

    exit 1
fi
