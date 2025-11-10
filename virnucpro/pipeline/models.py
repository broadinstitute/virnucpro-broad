"""PyTorch model and dataset classes"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Union
from pathlib import Path
import logging

logger = logging.getLogger('virnucpro.models')


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier for viral sequence classification.

    Architecture: Input -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Output

    Based on prediction.py:48-71
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_class: int):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension (default: 3328 for DNABERT-S + ESM-2)
            hidden_dim: Hidden layer dimension
            num_class: Number of output classes (2 for binary classification)
        """
        super(MLPClassifier, self).__init__()

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class PredictDataBatchDataset(Dataset):
    """
    PyTorch Dataset for loading merged feature tensors.

    Loads multiple .pt files containing merged DNABERT-S + ESM-2 features
    and provides index-based access across all files.

    Based on prediction.py:20-45
    """

    def __init__(self, file_list: List[Union[str, Path]]):
        """
        Initialize dataset from list of merged feature files.

        Args:
            file_list: List of paths to .pt files containing merged features
        """
        self.file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
        self.ids = []
        self.data = []
        self._load_all_data()

        logger.info(f"Loaded {len(self)} sequences from {len(self.file_list)} files")

    def _load_all_data(self):
        """Load all data from .pt files into memory"""
        for file_path in self.file_list:
            logger.debug(f"Loading {file_path}")
            data_dict = torch.load(file_path)
            data = data_dict['data']
            self.data.append(data)
            ids = data_dict['ids']
            self.ids.extend(ids)

    def __len__(self):
        """Return total number of sequences across all files"""
        return sum(d.size(0) for d in self.data)

    def __getitem__(self, idx):
        """
        Get item by index across all loaded files.

        Args:
            idx: Global index

        Returns:
            Tuple of (data_tensor, sequence_id)
        """
        cumulative_size = 0
        for data in self.data:
            if cumulative_size + data.size(0) > idx:
                index_in_file = idx - cumulative_size
                return data[index_in_file], self.ids[cumulative_size + index_in_file]
            cumulative_size += data.size(0)
        raise IndexError("Index out of range")
