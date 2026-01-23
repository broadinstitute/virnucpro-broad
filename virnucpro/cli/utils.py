"""Utility CLI commands"""

import click
import sys
from pathlib import Path
import logging

from virnucpro.core.device import list_available_devices
from virnucpro.utils.validation import validate_fasta_file
from virnucpro.core.config import Config

logger = logging.getLogger('virnucpro.cli.utils')


@click.group(name='utils')
def utils():
    """Utility commands for VirNucPro"""
    pass


@utils.command(name='list-devices')
def list_devices_cmd():
    """
    List available compute devices.

    Shows CPU and GPU information including device names,
    memory, and availability status.
    """
    list_available_devices()


@utils.command(name='validate')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--max-errors', '-e', type=int, default=10,
              help='Maximum errors to display')
def validate_cmd(input_file, max_errors):
    """
    Validate FASTA input file.

    Checks for common issues:
      - Duplicate sequence IDs
      - Invalid characters in sequences
      - Ambiguous bases
      - Empty sequences

    Example:
      python -m virnucpro utils validate sequences.fasta
    """
    logger.info(f"Validating {input_file}")

    try:
        is_valid, errors, warnings, stats = validate_fasta_file(
            Path(input_file),
            max_errors=max_errors
        )

        # Print statistics
        print(f"\nFile Statistics:")
        print(f"  Total sequences: {stats['total_sequences']}")
        print(f"  Length range: {stats['min_length']}-{stats['max_length']} bp")
        print(f"  Average length: {stats['avg_length']:.1f} bp")
        print(f"  Sequences with ambiguous bases: {stats['ambiguous_count']}")
        print(f"  Duplicate IDs: {len(stats['duplicate_ids'])}")

        # Print errors
        if errors:
            print(f"\nErrors found ({len(errors)}):")
            for error in errors[:max_errors]:
                print(f"  ✗ {error}")
            if len(errors) > max_errors:
                print(f"  ... and {len(errors) - max_errors} more")

        # Print warnings
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for warning in warnings[:max_errors]:
                print(f"  ⚠ {warning}")
            if len(warnings) > max_errors:
                print(f"  ... and {len(warnings) - max_errors} more")

        if is_valid:
            print("\n✓ File is valid and ready for processing")
            sys.exit(0)
        else:
            print("\n✗ File validation failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


@utils.command(name='generate-config')
@click.option('--output', '-o', type=click.Path(),
              default='virnucpro_config.yaml',
              help='Output configuration file path')
def generate_config_cmd(output):
    """
    Generate a configuration file template.

    Creates a YAML configuration file with default values
    that can be customized and used with --config.

    Example:
      python -m virnucpro utils generate-config -o my_config.yaml
      python -m virnucpro predict input.fasta --config my_config.yaml
    """
    try:
        # Load default config
        config = Config.load()

        # Save to specified location
        output_path = Path(output)
        config.save(output_path)

        print(f"Configuration template saved to {output_path}")
        print(f"\nEdit this file to customize VirNucPro settings,")
        print(f"then use it with: --config {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        sys.exit(1)


@utils.command(name='validate-checkpoints')
@click.argument('checkpoint_dir', type=click.Path(exists=True))
@click.option('--skip-load', is_flag=True,
              help='Skip torch.load validation (faster, less thorough)')
def validate_checkpoints_cmd(checkpoint_dir, skip_load):
    """
    Validate all checkpoints in a directory.

    Checks checkpoint integrity without running the pipeline.
    Reports status and returns appropriate exit code.

    Exit codes:
      0: All checkpoints valid
      1: Some checkpoints failed validation
      3: Checkpoint directory issue

    Example:
      python -m virnucpro utils validate-checkpoints output_dir/checkpoints
    """
    from virnucpro.core.checkpoint_validation import (
        validate_checkpoint_batch,
        load_failed_checkpoints,
        CHECKPOINT_EXIT_CODE
    )

    logger.info(f"Validating checkpoints in: {checkpoint_dir}")

    try:
        checkpoint_path = Path(checkpoint_dir)

        # Find all .pt files
        checkpoint_files = list(checkpoint_path.glob('*.pt'))

        if not checkpoint_files:
            print(f"\nNo checkpoint files (*.pt) found in {checkpoint_dir}")
            sys.exit(CHECKPOINT_EXIT_CODE)

        print(f"\nFound {len(checkpoint_files)} checkpoint file(s)")

        # Validate all checkpoints
        valid_paths, failed_items = validate_checkpoint_batch(
            checkpoint_files,
            skip_load=skip_load,
            logger_instance=logger
        )

        # Report results
        print(f"\nValidation Results:")
        print(f"  Valid:  {len(valid_paths)}")
        print(f"  Failed: {len(failed_items)}")

        if failed_items:
            print(f"\nFailed Checkpoints:")
            for checkpoint_path, error_msg in failed_items:
                error_type = 'CORRUPTED' if 'corrupted:' in error_msg else 'INCOMPATIBLE'
                print(f"  [{error_type}] {checkpoint_path.name}")
                print(f"    Reason: {error_msg}")

        # Check for historical failures
        failed_history = load_failed_checkpoints(checkpoint_path)
        if failed_history:
            print(f"\nHistorical Failures (from failed_checkpoints.txt):")
            print(f"  Total: {len(failed_history)}")
            for path, reason, timestamp in failed_history[-5:]:  # Show last 5
                print(f"  {Path(path).name}: {reason}")
                print(f"    Time: {timestamp}")

        if failed_items:
            print(f"\n✗ Validation failed: {len(failed_items)} checkpoint(s) have issues")
            sys.exit(1)
        else:
            print(f"\n✓ All checkpoints valid")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        sys.exit(CHECKPOINT_EXIT_CODE)
