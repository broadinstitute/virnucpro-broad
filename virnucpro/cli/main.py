"""Main Click CLI group and global options"""

import click
import sys
from pathlib import Path
from virnucpro.core.logging_setup import setup_logging
from virnucpro.core.config import Config
from virnucpro import __version__

# Import command modules
from virnucpro.cli import predict
from virnucpro.cli import profile
from virnucpro.cli import utils


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(version=__version__, prog_name='VirNucPro')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose (DEBUG level) logging')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress console output (errors only)')
@click.option('--log-file', '-l', type=click.Path(),
              help='Path to log file')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.pass_context
def cli(ctx, verbose, quiet, log_file, config):
    """
    VirNucPro - Viral Nucleotide and Protein Identifier

    A production-ready tool for identifying viral sequences using
    six-frame translation and large language models (DNABERT-S and ESM-2).

    Examples:

      # Basic prediction with 500bp model
      python -m virnucpro predict input.fasta

      # Use 300bp model with custom output
      python -m virnucpro predict input.fasta --model-type 300 -o results/

      # Use specific GPU and resume from checkpoint
      python -m virnucpro predict input.fasta --device cuda:1 --resume

      # List available compute devices
      python -m virnucpro utils list-devices

    For detailed help on a command:
      python -m virnucpro COMMAND --help
    """
    # Initialize context object
    ctx.ensure_object(dict)

    # Setup logging
    log_file_path = Path(log_file) if log_file else None
    logger = setup_logging(verbose=verbose, log_file=log_file_path, quiet=quiet)
    ctx.obj['logger'] = logger

    # Load configuration
    try:
        if config:
            cfg = Config.load(Path(config))
            logger.info(f"Loaded configuration from {config}")
        else:
            cfg = Config.load()  # Load default
            logger.debug("Using default configuration")

        ctx.obj['config'] = cfg
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


# Register commands
cli.add_command(predict.predict)
cli.add_command(profile.profile)
cli.add_command(utils.utils)


if __name__ == '__main__':
    cli()
