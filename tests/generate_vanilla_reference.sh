#!/bin/bash
# Generate vanilla reference outputs for comparison testing
#
# This script runs the vanilla prediction.py implementation on test data
# and saves the outputs as reference "golden" files for regression testing.
#
# Prerequisites:
#   - prediction.py exists in project root
#   - 500_model.pth checkpoint exists
#   - tests/data/test_with_orfs.fa exists

set -e  # Exit on error

echo "=========================================="
echo "Generating Vanilla Reference Outputs"
echo "=========================================="
echo ""

# Configuration
INPUT_FASTA="tests/data/test_with_orfs.fa"
CHUNK_SIZE="500"
MODEL_CHECKPOINT="500_model.pth"
REFERENCE_DIR="tests/data/reference_vanilla_output"

# Validate prerequisites
if [ ! -f "prediction.py" ]; then
    echo "❌ Error: prediction.py not found in project root"
    exit 1
fi

if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ Error: Model checkpoint not found: $MODEL_CHECKPOINT"
    echo "   Please download or symlink the model checkpoint first"
    exit 1
fi

if [ ! -f "$INPUT_FASTA" ]; then
    echo "❌ Error: Test input not found: $INPUT_FASTA"
    exit 1
fi

# Clean previous vanilla outputs
echo "🧹 Cleaning previous outputs..."
rm -rf tests/data/test_with_orfs_nucleotide/
rm -rf tests/data/test_with_orfs_protein/
rm -f tests/data/test_with_orfs_chunked*.fa
rm -f tests/data/test_with_orfs_identified_*.fa*

# Run vanilla implementation
echo "🚀 Running vanilla prediction.py..."
echo "   Input: $INPUT_FASTA"
echo "   Chunk size: $CHUNK_SIZE"
echo "   Model: $MODEL_CHECKPOINT"
echo ""

python prediction.py "$INPUT_FASTA" "$CHUNK_SIZE" "$MODEL_CHECKPOINT"

# Check if outputs were generated
NUCLEOTIDE_DIR="tests/data/test_with_orfs_nucleotide"
if [ ! -d "$NUCLEOTIDE_DIR" ]; then
    echo "❌ Error: Vanilla pipeline did not create nucleotide output directory"
    exit 1
fi

# Count embedding files
PT_FILES=$(find "$NUCLEOTIDE_DIR" -name "*.pt" | wc -l)
if [ "$PT_FILES" -eq 0 ]; then
    echo "⚠️  Warning: No .pt embedding files generated"
    echo "   This may indicate that no valid ORFs were found"
    echo "   Check if test_with_orfs.fa contains valid ORF sequences"
fi

# Save as reference
echo ""
echo "📦 Saving outputs as reference..."
mkdir -p "$REFERENCE_DIR"

# Vanilla creates three directories:
# 1. test_with_orfs_nucleotide/ - DNABERT-S embeddings
# 2. test_with_orfs_protein/ - ESM-2 embeddings
# 3. test_with_orfs_merged/ - Merged features and predictions

PROTEIN_DIR="tests/data/test_with_orfs_protein"
MERGED_DIR="tests/data/test_with_orfs_merged"

# Copy DNABERT-S embeddings
echo "  Copying DNABERT-S embeddings from nucleotide dir..."
cp -v "$NUCLEOTIDE_DIR"/*.pt "$REFERENCE_DIR/" 2>/dev/null || echo "   (No nucleotide .pt files)"

# Copy ESM-2 embeddings
if [ -d "$PROTEIN_DIR" ]; then
    echo "  Copying ESM-2 embeddings from protein dir..."
    cp -v "$PROTEIN_DIR"/*.pt "$REFERENCE_DIR/" 2>/dev/null || echo "   (No protein .pt files)"
fi

# Copy merged features and predictions
if [ -d "$MERGED_DIR" ]; then
    echo "  Copying merged features and predictions..."
    cp -v "$MERGED_DIR"/*.pt "$REFERENCE_DIR/" 2>/dev/null || echo "   (No merged .pt files)"
    cp -v "$MERGED_DIR"/prediction_results*.* "$REFERENCE_DIR/" 2>/dev/null || echo "   (No prediction files)"
fi

# Summary
echo ""
echo "=========================================="
echo "✅ Vanilla Reference Generation Complete"
echo "=========================================="
echo ""
echo "Reference outputs saved to: $REFERENCE_DIR"
echo ""
ls -lh "$REFERENCE_DIR"
echo ""
echo "Next steps:"
echo "  1. Run comparison tests: pytest tests/test_vanilla_comparison.py -v"
echo "  2. Or run full test suite: pytest tests/ -v"
echo ""
