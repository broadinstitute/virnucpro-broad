"""Progress reporting utilities using tqdm"""

from tqdm import tqdm
from typing import Optional, Iterable, Any
import logging
import sys

logger = logging.getLogger('virnucpro.progress')


class ProgressReporter:
    """
    Wrapper for tqdm progress bars that integrates with logging.

    Ensures progress bars don't interfere with log messages and
    provides consistent styling across the application.
    """

    def __init__(self, disable: bool = False):
        """
        Initialize progress reporter.

        Args:
            disable: If True, disable all progress bars (for quiet mode or CI)
        """
        self.disable = disable

    def create_bar(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = 'it',
        leave: bool = True,
        **kwargs
    ) -> tqdm:
        """
        Create a tqdm progress bar.

        Args:
            iterable: Optional iterable to wrap
            total: Total number of iterations (if iterable is None)
            desc: Description prefix for progress bar
            unit: Unit name (default: 'it')
            leave: Keep progress bar after completion
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar object

        Example:
            >>> reporter = ProgressReporter()
            >>> for item in reporter.create_bar(items, desc="Processing"):
            ...     process(item)
        """
        # Configure tqdm to work with logging
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            disable=self.disable,
            file=sys.stdout,
            ncols=100,  # Fixed width for consistent display
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            **kwargs
        )

    def create_file_bar(
        self,
        total_files: int,
        desc: str = "Processing files",
        **kwargs
    ) -> tqdm:
        """
        Create a progress bar specifically for file processing.

        Args:
            total_files: Total number of files to process
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar
        """
        return self.create_bar(
            total=total_files,
            desc=desc,
            unit='file',
            **kwargs
        )

    def create_sequence_bar(
        self,
        total_sequences: int,
        desc: str = "Processing sequences",
        **kwargs
    ) -> tqdm:
        """
        Create a progress bar for sequence processing.

        Args:
            total_sequences: Total number of sequences
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar
        """
        return self.create_bar(
            total=total_sequences,
            desc=desc,
            unit='seq',
            **kwargs
        )

    def create_stage_bar(
        self,
        stages: int = 1,
        desc: str = "Pipeline stages",
        **kwargs
    ) -> tqdm:
        """
        Create a progress bar for pipeline stages.

        Args:
            stages: Number of stages
            desc: Description
            **kwargs: Additional tqdm arguments

        Returns:
            tqdm progress bar
        """
        return self.create_bar(
            total=stages,
            desc=desc,
            unit='stage',
            **kwargs
        )

    @staticmethod
    def write_above_bar(message: str):
        """
        Write a message above the current progress bar.

        Uses tqdm.write() to ensure message appears above progress bar
        rather than interfering with it.

        Args:
            message: Message to display
        """
        tqdm.write(message)


# Helper functions for common progress patterns
def process_with_progress(
    items: Iterable[Any],
    process_fn: callable,
    desc: str = "Processing",
    unit: str = "it",
    disable: bool = False
) -> list:
    """
    Process items with automatic progress bar.

    Args:
        items: Items to process
        process_fn: Function to apply to each item
        desc: Progress bar description
        unit: Unit name
        disable: Disable progress bar

    Returns:
        List of processed results
    """
    reporter = ProgressReporter(disable=disable)
    results = []

    with reporter.create_bar(items, desc=desc, unit=unit) as pbar:
        for item in pbar:
            result = process_fn(item)
            results.append(result)

    return results
