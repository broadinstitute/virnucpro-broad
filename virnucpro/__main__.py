"""
Entry point for VirNucPro CLI.
Allows execution via: python -m virnucpro
"""

import sys
from virnucpro.cli.main import cli

if __name__ == "__main__":
    sys.exit(cli())
