"""Pipeline modules for VirNucPro prediction workflow"""

from virnucpro.pipeline.async_inference import (
    AsyncInferenceRunner,
    InferenceResult,
    run_async_inference,
)

__all__ = [
    'AsyncInferenceRunner',
    'InferenceResult',
    'run_async_inference',
]
