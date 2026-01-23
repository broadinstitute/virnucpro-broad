"""Profile command for batch size optimization"""

import click
import sys
import json
from pathlib import Path
import logging

logger = logging.getLogger('virnucpro.cli.profile')


@click.command()
@click.option('--model', '-m',
              type=click.Choice(['dnabert-s', 'esm2']),
              required=True,
              help='Model to profile (dnabert-s or esm2)')
@click.option('--device', '-d',
              type=str,
              default='cuda:0',
              help='Device to profile on (e.g., cuda:0, cuda:1)')
@click.option('--min-batch',
              type=int,
              default=512,
              help='Minimum batch size to test (tokens, default: 512)')
@click.option('--max-batch',
              type=int,
              default=8192,
              help='Maximum batch size to test (tokens, default: 8192)')
@click.option('--step',
              type=int,
              default=512,
              help='Step size for batch size increments (default: 512)')
@click.option('--test-file',
              type=click.Path(exists=True),
              default=None,
              help='Optional FASTA file with test sequences (uses synthetic data if not provided)')
@click.option('--output', '-o',
              type=click.Path(),
              default=None,
              help='Save detailed results to JSON file')
@click.pass_context
def profile(ctx, model, device, min_batch, max_batch, step, test_file, output):
    """
    Profile GPU to find optimal batch sizes.

    This command tests different batch sizes on your GPU to determine the
    optimal configuration for DNABERT-S or ESM-2 processing. It measures:

    - Throughput (sequences per second)
    - Memory usage
    - Maximum batch size before OOM

    The recommended batch size is set to 80% of maximum to leave headroom
    for variation in sequence lengths.

    \b
    Examples:

      # Profile DNABERT-S on GPU 0
      python -m virnucpro profile --model dnabert-s --device cuda:0

      # Profile ESM-2 with custom range
      python -m virnucpro profile --model esm2 --min-batch 1024 --max-batch 4096

      # Use real sequences for profiling
      python -m virnucpro profile --model dnabert-s --test-file sequences.fasta

      # Save detailed results to JSON
      python -m virnucpro profile --model esm2 --output profile_results.json

    \b
    Note:
      - Profiling requires a CUDA-enabled GPU
      - BF16 precision is automatically used on Ampere+ GPUs (RTX 30xx/40xx, A100)
      - Other processes using the GPU may affect results
    """
    logger = ctx.obj['logger']

    logger.info(f"VirNucPro Batch Size Profiler")
    logger.info(f"Model: {model}")
    logger.info(f"Device: {device}")

    # Validate device is CUDA
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA not available. Profiling requires a GPU.")
        sys.exit(1)

    # Validate device exists
    try:
        device_obj = torch.device(device)
        if device_obj.type != 'cuda':
            logger.error(f"Device {device} is not a CUDA device. Use format: cuda:0, cuda:1, etc.")
            sys.exit(1)

        device_id = device_obj.index if device_obj.index is not None else 0
        if device_id >= torch.cuda.device_count():
            logger.error(f"Device {device} not found. Available devices: {torch.cuda.device_count()}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Invalid device {device}: {e}")
        sys.exit(1)

    # Convert test_file to Path if provided
    test_file_path = Path(test_file) if test_file else None

    try:
        # Import profiling functions
        from virnucpro.pipeline.profiler import profile_dnabert_batch_size, profile_esm_batch_size

        # Run profiling
        logger.info(f"\nStarting profiling (this may take a few minutes)...")
        logger.info(f"Testing batch sizes from {min_batch} to {max_batch} (step: {step})\n")

        if model == 'dnabert-s':
            results = profile_dnabert_batch_size(
                device=device,
                test_sequence_file=test_file_path,
                min_batch=min_batch,
                max_batch=max_batch,
                step=step
            )
        else:  # esm2
            results = profile_esm_batch_size(
                device=device,
                test_sequence_file=test_file_path,
                min_batch=min_batch,
                max_batch=max_batch,
                step=step
            )

        # Display results
        display_results(results, model, logger)

        # Save to JSON if requested
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nDetailed results saved to: {output_path}")

        logger.info("\nProfiling complete!")

        # Return success
        sys.exit(0)

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        if ctx.obj['logger'].level == logging.DEBUG:
            logger.exception("Detailed error traceback:")
        sys.exit(1)


def display_results(results: dict, model_name: str, logger):
    """
    Display profiling results in user-friendly format.

    Args:
        results: Results dictionary from profiling function
        model_name: Name of model profiled
        logger: Logger instance
    """
    logger.info("\n" + "="*70)
    logger.info(f"  {model_name.upper()} PROFILING RESULTS")
    logger.info("="*70)

    logger.info(f"\nDevice: {results['device']}")
    logger.info(f"BF16 Precision: {'Enabled' if results['bf16_enabled'] else 'Disabled'}")

    logger.info(f"\n{'Batch Size':<15} {'Throughput':<20} {'Memory':<15} {'Status':<10}")
    logger.info("-"*70)

    # Display throughput curve
    for (batch_size, throughput), (_, memory) in zip(
        results['throughput_curve'],
        results['memory_curve']
    ):
        logger.info(f"{batch_size:<15} {f'{throughput:.1f} seq/s':<20} {f'{memory:.2f} GB':<15} {'✓ OK':<10}")

    # Check if we hit OOM
    if len(results['throughput_curve']) < (results['max_batch_size'] - 512) // 512:
        logger.info(f"{'> ' + str(results['max_batch_size'] + 512):<15} {'N/A':<20} {'N/A':<15} {'✗ OOM':<10}")

    logger.info("="*70)

    # Recommendations
    logger.info(f"\nRECOMMENDATIONS:")
    logger.info(f"  Maximum batch size: {results['max_batch_size']} tokens")
    logger.info(f"  Recommended batch size: {results['optimal_batch_size']} tokens")
    logger.info(f"\n  Use --{model_name.replace('-s', '')}-batch-size={results['optimal_batch_size']} for best performance")

    # Display throughput chart
    if results['throughput_curve']:
        logger.info(f"\nTHROUGHPUT CHART:")
        max_throughput = max(t for _, t in results['throughput_curve'])

        for batch_size, throughput in results['throughput_curve']:
            bar_length = int((throughput / max_throughput) * 40)
            bar = '█' * bar_length
            logger.info(f"  {batch_size:>5} tokens | {bar:<40} {throughput:>6.1f} seq/s")

    logger.info("\n" + "="*70)
