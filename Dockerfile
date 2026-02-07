# VirNucPro: FastESM2 Migration Docker Environment
# Base: NVIDIA PyTorch 25.09 with native GB10 (sm_121) support
# PyTorch: 2.9.0a0+50eac811a6.nv25.09 | CUDA: 13.0

FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set working directory
WORKDIR /workspace

# Copy requirements first for better layer caching
COPY requirements-docker.txt /tmp/requirements-docker.txt

# Install Python dependencies
# Note: PyTorch and CUDA are already included in base image
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt

# Copy project files
COPY . /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Default command: run validation
CMD ["python", "scripts/validate_environment.py"]
