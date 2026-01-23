"""Rich-based dashboard for live multi-GPU progress tracking"""

import sys
import logging
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger('virnucpro.pipeline.dashboard')

# Check if rich is available (optional dependency)
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TaskID,
    )
    from rich.live import Live
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("rich library not available, falling back to simple logging")


class MultiGPUDashboard:
    """
    Live progress dashboard for multi-GPU processing with fallback to logging.

    Displays concurrent progress bars for each GPU, showing files processed,
    current status, and elapsed time. Automatically falls back to simple
    logging in non-TTY environments or when rich is not installed.

    Example:
        >>> dashboard = MultiGPUDashboard(4, {0: 10, 1: 10, 2: 12, 3: 8})
        >>> dashboard.start()
        >>> dashboard.update(0, files_completed=1)
        >>> dashboard.set_status(0, "Processing sample_123.fa")
        >>> dashboard.complete(0)
        >>> dashboard.complete_all()
    """

    def __init__(self, num_gpus: int, total_files_per_gpu: Dict[int, int]):
        """
        Initialize multi-GPU dashboard.

        Args:
            num_gpus: Number of GPUs being used
            total_files_per_gpu: Dictionary mapping GPU ID -> total files to process
        """
        self.num_gpus = num_gpus
        self.total_files_per_gpu = total_files_per_gpu
        self.start_time: Optional[datetime] = None
        self.completed_per_gpu: Dict[int, int] = {gpu_id: 0 for gpu_id in range(num_gpus)}

        # Check if we should use rich display
        self.use_rich = RICH_AVAILABLE and sys.stdout.isatty()

        if self.use_rich:
            self.console = Console()
            self.progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            )
            self.gpu_tasks: Dict[int, TaskID] = {}
            self.live: Optional[Live] = None
        else:
            logger.info("Using simple logging mode (non-TTY or rich not available)")
            self.console = None
            self.progress = None
            self.gpu_tasks = {}
            self.live = None

    def start(self):
        """Initialize and start the progress display."""
        self.start_time = datetime.now()

        if self.use_rich:
            # Create progress tasks for each GPU
            for gpu_id in range(self.num_gpus):
                total_files = self.total_files_per_gpu.get(gpu_id, 0)
                task_id = self.progress.add_task(
                    f"[cyan]GPU {gpu_id}",
                    total=total_files
                )
                self.gpu_tasks[gpu_id] = task_id

            # Start Live display
            self.live = Live(self.progress, console=self.console, refresh_per_second=4)
            self.live.start()
            logger.info(f"Started dashboard for {self.num_gpus} GPUs")
        else:
            # Log start in simple mode
            total_files = sum(self.total_files_per_gpu.values())
            logger.info(f"Started processing {total_files} files across {self.num_gpus} GPUs")
            for gpu_id in range(self.num_gpus):
                files = self.total_files_per_gpu.get(gpu_id, 0)
                logger.info(f"  GPU {gpu_id}: {files} files")

    def update(self, gpu_id: int, files_completed: int = 1):
        """
        Update progress for a specific GPU.

        Args:
            gpu_id: GPU device ID
            files_completed: Number of files completed (default: 1)
        """
        if gpu_id not in self.completed_per_gpu:
            logger.warning(f"Invalid GPU ID {gpu_id}")
            return

        self.completed_per_gpu[gpu_id] += files_completed

        if self.use_rich:
            task_id = self.gpu_tasks.get(gpu_id)
            if task_id is not None:
                self.progress.update(task_id, advance=files_completed)
        else:
            # Log progress in simple mode
            total = self.total_files_per_gpu.get(gpu_id, 0)
            completed = self.completed_per_gpu[gpu_id]
            percent = (completed / total * 100.0) if total > 0 else 0.0
            logger.info(f"GPU {gpu_id}: {completed}/{total} files ({percent:.1f}%)")

    def set_status(self, gpu_id: int, status_text: str):
        """
        Update status text for a specific GPU.

        Args:
            gpu_id: GPU device ID
            status_text: Status message (e.g., "Processing file_123.fa")
        """
        if self.use_rich:
            task_id = self.gpu_tasks.get(gpu_id)
            if task_id is not None:
                self.progress.update(task_id, description=f"[cyan]GPU {gpu_id} - {status_text}")
        else:
            logger.info(f"GPU {gpu_id}: {status_text}")

    def complete(self, gpu_id: int):
        """
        Mark a GPU's work as complete.

        Args:
            gpu_id: GPU device ID
        """
        if self.use_rich:
            task_id = self.gpu_tasks.get(gpu_id)
            if task_id is not None:
                total = self.total_files_per_gpu.get(gpu_id, 0)
                # Ensure task shows 100%
                self.progress.update(task_id, completed=total)
                self.progress.update(task_id, description=f"[green]GPU {gpu_id} - Complete")
        else:
            total = self.total_files_per_gpu.get(gpu_id, 0)
            logger.info(f"GPU {gpu_id}: Complete ({total}/{total} files)")

    def complete_all(self):
        """Mark all GPU tasks as complete and stop the display."""
        # Mark all GPUs complete
        for gpu_id in range(self.num_gpus):
            if gpu_id not in self.completed_per_gpu:
                continue

            if self.use_rich:
                task_id = self.gpu_tasks.get(gpu_id)
                if task_id is not None:
                    total = self.total_files_per_gpu.get(gpu_id, 0)
                    self.progress.update(task_id, completed=total)

        # Stop live display
        if self.live is not None:
            self.live.stop()

        # Log summary
        summary = self.get_summary()
        elapsed = summary['elapsed_seconds']
        total_files = summary['total_files']
        throughput = summary['throughput']

        if self.use_rich:
            logger.info(f"All GPUs complete: {total_files} files in {elapsed:.1f}s ({throughput:.2f} files/sec)")
        else:
            logger.info(f"Processing complete: {total_files} files in {elapsed:.1f}s ({throughput:.2f} files/sec)")

    def get_summary(self) -> Dict:
        """
        Get processing summary statistics.

        Returns:
            Dictionary with:
                - total_files: Total files processed
                - completed: Files completed so far
                - elapsed_seconds: Time elapsed since start
                - throughput: Files per second
        """
        total_files = sum(self.total_files_per_gpu.values())
        completed = sum(self.completed_per_gpu.values())

        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
        else:
            elapsed = 0.0

        throughput = completed / elapsed if elapsed > 0 else 0.0

        return {
            'total_files': total_files,
            'completed': completed,
            'elapsed_seconds': elapsed,
            'throughput': throughput
        }


def create_progress_display(file_assignments: Dict[int, List[Path]]) -> MultiGPUDashboard:
    """
    Create a dashboard from file assignments.

    Helper function to create a configured MultiGPUDashboard instance
    from the output of file assignment functions.

    Args:
        file_assignments: Dictionary mapping GPU ID -> list of file paths

    Returns:
        Configured MultiGPUDashboard instance ready to start

    Example:
        >>> files = {0: [Path('a.fa'), Path('b.fa')], 1: [Path('c.fa')]}
        >>> dashboard = create_progress_display(files)
        >>> dashboard.start()
    """
    num_gpus = len(file_assignments)
    total_files_per_gpu = {
        gpu_id: len(file_list)
        for gpu_id, file_list in file_assignments.items()
    }

    return MultiGPUDashboard(num_gpus, total_files_per_gpu)


def monitor_progress(progress_queue, dashboard: MultiGPUDashboard, stop_event: threading.Event):
    """
    Monitor progress queue and update dashboard.

    Runs in a background thread, consuming progress events from workers
    and updating the dashboard accordingly. Supports both TTY and non-TTY modes.

    Args:
        progress_queue: Multiprocessing queue receiving progress events
        dashboard: Dashboard instance to update
        stop_event: Event to signal monitoring thread to stop

    Progress event format:
        {
            'gpu_id': int,      # GPU device ID
            'file': str,        # File path
            'status': str       # 'complete' or 'failed'
        }
    """
    logger.debug("Progress monitor thread started")

    try:
        while not stop_event.is_set():
            try:
                # Get progress event with timeout to allow checking stop_event
                event = progress_queue.get(timeout=0.5)

                gpu_id = event.get('gpu_id')
                file_path = event.get('file')
                status = event.get('status')

                # Update dashboard based on event
                if status == 'complete':
                    dashboard.update(gpu_id, files_completed=1)
                elif status == 'failed':
                    # Still count as progress (attempted)
                    dashboard.update(gpu_id, files_completed=1)
                    logger.warning(f"GPU {gpu_id}: Failed to process {file_path}")

            except queue.Empty:
                # Timeout - continue loop to check stop_event
                continue

    except Exception as e:
        logger.exception("Error in progress monitor thread")

    logger.debug("Progress monitor thread stopped")
