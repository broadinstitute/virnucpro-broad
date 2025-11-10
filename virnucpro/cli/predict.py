"""Predict command implementation"""

import click
import sys
from pathlib import Path
import logging

from virnucpro.core.device import validate_and_get_device
from virnucpro.core.config import Config

logger = logging.getLogger('virnucpro.cli.predict')


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--model-type', '-m',
              type=click.Choice(['300', '500', 'custom']),
              default='500',
              help='Model type to use (300bp, 500bp, or custom)')
@click.option('--model-path', '-p',
              type=click.Path(exists=True),
              help='Path to custom model file (required if model-type is custom)')
@click.option('--expected-length', '-e',
              type=int,
              help='Expected sequence length (default: matches model type)')
@click.option('--output-dir', '-o',
              type=click.Path(),
              help='Output directory for results (default: {input}_predictions)')
@click.option('--device', '-d',
              type=str,
              default='auto',
              help='Device: "auto", "cpu", "cuda", "cuda:N", or "N" (default: auto)')
@click.option('--batch-size', '-b',
              type=int,
              help='Batch size for prediction (default: from config)')
@click.option('--num-workers', '-w',
              type=int,
              help='Number of data loading workers (default: from config)')
@click.option('--keep-intermediate', '-k',
              is_flag=True,
              help='Keep intermediate files after completion')
@click.option('--resume',
              is_flag=True,
              help='Resume from checkpoint if available')
@click.option('--force', '-f',
              is_flag=True,
              help='Overwrite existing output directory')
@click.option('--no-progress',
              is_flag=True,
              help='Disable progress bars (useful for logging to files)')
@click.pass_context
def predict(ctx, input_file, model_type, model_path, expected_length,
            output_dir, device, batch_size, num_workers,
            keep_intermediate, resume, force, no_progress):
    """
    Predict viral sequences from FASTA input.

    This command processes input sequences through the VirNucPro pipeline:

      1. Chunk sequences to expected length
      2. Six-frame translation to identify ORFs
      3. Extract features using DNABERT-S (DNA) and ESM-2 (protein)
      4. Merge features and predict using MLP classifier
      5. Generate consensus predictions across reading frames

    Examples:

      # Basic prediction
      python -m virnucpro predict sequences.fasta

      # Use 300bp model with GPU 1
      python -m virnucpro predict sequences.fasta -m 300 -d cuda:1

      # Resume interrupted run
      python -m virnucpro predict sequences.fasta --resume

      # Custom model with CPU
      python -m virnucpro predict sequences.fasta -m custom -p my_model.pth -d cpu
    """
    logger = ctx.obj['logger']
    config = ctx.obj['config']

    logger.info(f"VirNucPro Prediction - Input: {input_file}")

    # Validate and prepare parameters
    try:
        # Validate model parameters
        if model_type == 'custom' and not model_path:
            raise click.BadParameter(
                "--model-path is required when using --model-type custom"
            )

        # Set defaults from config
        if not expected_length:
            expected_length = 300 if model_type == '300' else 500

        if not model_path:
            model_path = config.get(f'prediction.models.{model_type}')
            if not model_path or not Path(model_path).exists():
                raise click.FileError(
                    model_path or f"{model_type}_model.pth",
                    "Model file not found. Ensure model file exists in project root."
                )

        # Set output directory
        if not output_dir:
            input_base = Path(input_file).stem
            output_dir = f"{input_base}_predictions"
        output_dir = Path(output_dir)

        # Check output directory
        if output_dir.exists() and not force and not resume:
            if not click.confirm(f"Output directory {output_dir} exists. Overwrite?"):
                logger.info("Prediction cancelled by user")
                sys.exit(0)

        # Get batch size and workers from config if not specified
        if batch_size is None:
            batch_size = config.get('prediction.batch_size', 256)
        if num_workers is None:
            num_workers = config.get('prediction.num_workers', 4)

        # Validate and get device
        fallback_to_cpu = config.get('device.fallback_to_cpu', True)
        device_obj = validate_and_get_device(device, fallback_to_cpu=fallback_to_cpu)

        # Determine cleanup behavior
        auto_cleanup = config.get('files.auto_cleanup', True)
        cleanup = auto_cleanup and not keep_intermediate

        # Log configuration
        logger.info("Configuration:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Expected length: {expected_length}bp")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Device: {device_obj}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Resume: {resume}")
        logger.info(f"  Cleanup intermediate files: {cleanup}")
        logger.info(f"  Progress bars: {'disabled' if no_progress else 'enabled'}")

        # Note: The run_prediction function will be implemented in Phase 4
        # For now, we'll just validate parameters and report configuration
        logger.warning(
            "Prediction pipeline not yet implemented. "
            "This will be completed in Phase 4 of the refactoring."
        )

        logger.info("Parameter validation completed successfully!")
        logger.info("All parameters are valid and ready for prediction pipeline.")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if ctx.obj['logger'].level == logging.DEBUG:
            logger.exception("Detailed error traceback:")
        sys.exit(1)
