"""Tests for feature extraction with batched processing"""

import torch
import pytest
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tempfile
from virnucpro.pipeline.features import extract_dnabert_features


@pytest.fixture
def sample_sequences():
    """Create sample DNA sequences for testing"""
    sequences = [
        SeqRecord(Seq("ATGCATGCATGCATGC"), id=f"seq_{i}", description="")
        for i in range(10)
    ]
    return sequences


@pytest.fixture
def temp_fasta_file(sample_sequences):
    """Create temporary FASTA file with sample sequences"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        SeqIO.write(sample_sequences, f, 'fasta')
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def device():
    """Get available device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_batch_processing_produces_correct_embeddings(temp_fasta_file, device):
    """Normal: Batch of 10 sequences produces correct embeddings matching sequential"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "features.pt"

        result = extract_dnabert_features(
            temp_fasta_file,
            output_file,
            device,
            batch_size=4
        )

        assert result == output_file
        assert output_file.exists()

        features = torch.load(output_file)
        assert 'nucleotide' in features
        assert 'data' in features
        assert len(features['nucleotide']) == 10
        assert len(features['data']) == 10

        for item in features['data']:
            assert 'label' in item
            assert 'mean_representation' in item
            assert len(item['mean_representation']) == 768


def test_single_sequence_batch(device):
    """Edge: Single sequence batch works correctly"""
    sequence = [SeqRecord(Seq("ATGCATGCATGCATGC"), id="single_seq", description="")]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        SeqIO.write(sequence, f, 'fasta')
        temp_path = Path(f.name)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "features.pt"

            result = extract_dnabert_features(
                temp_path,
                output_file,
                device,
                batch_size=1
            )

            features = torch.load(output_file)
            assert len(features['nucleotide']) == 1
            assert len(features['data']) == 1
            assert features['data'][0]['label'] == 'single_seq'
    finally:
        temp_path.unlink()


def test_batch_size_larger_than_sequence_count(temp_fasta_file, device):
    """Edge: Batch size larger than sequence count handles gracefully"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "features.pt"

        result = extract_dnabert_features(
            temp_fasta_file,
            output_file,
            device,
            batch_size=1000
        )

        features = torch.load(output_file)
        assert len(features['nucleotide']) == 10
        assert len(features['data']) == 10


def test_empty_input_file(device):
    """Error: Empty input file handled gracefully"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        temp_path = Path(f.name)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "features.pt"

            result = extract_dnabert_features(
                temp_path,
                output_file,
                device,
                batch_size=256
            )

            features = torch.load(output_file)
            assert len(features['nucleotide']) == 0
            assert len(features['data']) == 0
    finally:
        temp_path.unlink()


def test_output_equivalence_batched_vs_sequential(temp_fasta_file, device):
    """Property: Output equivalence - batched(sequences, bs) == sequential(sequences)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        batched_output = Path(tmpdir) / "batched.pt"
        sequential_output = Path(tmpdir) / "sequential.pt"

        extract_dnabert_features(temp_fasta_file, batched_output, device, batch_size=4)
        extract_dnabert_features(temp_fasta_file, sequential_output, device, batch_size=1)

        batched = torch.load(batched_output)
        sequential = torch.load(sequential_output)

        assert batched['nucleotide'] == sequential['nucleotide']
        assert len(batched['data']) == len(sequential['data'])

        for b_item, s_item in zip(batched['data'], sequential['data']):
            assert b_item['label'] == s_item['label']
            b_emb = torch.tensor(b_item['mean_representation'])
            s_emb = torch.tensor(s_item['mean_representation'])
            assert torch.allclose(b_emb, s_emb, rtol=1e-4, atol=1e-6)


def test_batch_size_independence(temp_fasta_file, device):
    """Property: Batch size independence - batched(sequences, bs1) == batched(sequences, bs2)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_bs3 = Path(tmpdir) / "bs3.pt"
        output_bs7 = Path(tmpdir) / "bs7.pt"

        extract_dnabert_features(temp_fasta_file, output_bs3, device, batch_size=3)
        extract_dnabert_features(temp_fasta_file, output_bs7, device, batch_size=7)

        features_bs3 = torch.load(output_bs3)
        features_bs7 = torch.load(output_bs7)

        assert features_bs3['nucleotide'] == features_bs7['nucleotide']

        for item3, item7 in zip(features_bs3['data'], features_bs7['data']):
            emb3 = torch.tensor(item3['mean_representation'])
            emb7 = torch.tensor(item7['mean_representation'])
            assert torch.allclose(emb3, emb7, rtol=1e-4, atol=1e-6)


def test_order_preservation(device):
    """Property: Order preservation - output order matches input order"""
    sequences = [
        SeqRecord(Seq("ATGC" * 4), id=f"seq_{i:03d}", description="")
        for i in range(20)
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        SeqIO.write(sequences, f, 'fasta')
        temp_path = Path(f.name)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "features.pt"

            extract_dnabert_features(temp_path, output_file, device, batch_size=7)

            features = torch.load(output_file)

            for i, (seq_id, data_item) in enumerate(zip(features['nucleotide'], features['data'])):
                expected_id = f"seq_{i:03d}"
                assert seq_id == expected_id
                assert data_item['label'] == expected_id
    finally:
        temp_path.unlink()
