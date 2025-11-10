"""
VirNucPro - Viral Nucleotide and Protein Identifier

A tool for identifying viral sequences using six-frame translation
and deep learning models (DNABERT-S and ESM-2).
"""

__version__ = "2.0.0"
__author__ = "VirNucPro Team"

from virnucpro.core.config import Config
from virnucpro.core.logging_setup import setup_logging

__all__ = ["Config", "setup_logging", "__version__"]
