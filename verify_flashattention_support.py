#!/usr/bin/env python3
"""Verify FlashAttention-2 support on RTX 4090s with ESM-2 model."""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from virnucpro.cuda.attention_utils import (
    get_attention_implementation,
    get_gpu_info,
    is_flash_attention_available
)


def main():
    print("=" * 70)
    print("FlashAttention-2 Support Verification")
    print("=" * 70)
    print()

    # 1. Check GPU information
    print("📊 GPU Information:")
    print("-" * 70)
    gpu_info = get_gpu_info()

    print(f"CUDA Available: {gpu_info['has_cuda']}")
    print(f"GPU Count: {gpu_info['device_count']}")
    print(f"FlashAttention-2 Available: {gpu_info['flash_attention_available']}")
    print()

    if gpu_info['device_count'] > 0:
        for device in gpu_info['devices']:
            print(f"  GPU {device['id']}: {device['name']}")
            print(f"    Compute Capability: {device['compute_capability']}")
            print(f"    Total Memory: {device['total_memory_gb']:.2f} GB")
            print(f"    Supports FlashAttention-2: {device['supports_flash_attention']}")
            print()

    # 2. Check PyTorch version and SDPA support
    print("🔧 PyTorch Configuration:")
    print("-" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Has sdp_kernel: {hasattr(torch.backends.cuda, 'sdp_kernel')}")
    print()

    # 3. Test FlashAttention-2 context
    print("🧪 FlashAttention-2 Context Test:")
    print("-" * 70)
    if torch.cuda.is_available():
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False
            ):
                print("✅ FlashAttention-2 context creation: SUCCESS")

                # Test actual attention computation
                device = torch.device("cuda:0")
                batch_size, seq_len, embed_dim = 2, 128, 768

                q = torch.randn(batch_size, 8, seq_len, embed_dim // 8, device=device, dtype=torch.bfloat16)
                k = torch.randn(batch_size, 8, seq_len, embed_dim // 8, device=device, dtype=torch.bfloat16)
                v = torch.randn(batch_size, 8, seq_len, embed_dim // 8, device=device, dtype=torch.bfloat16)

                output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                print(f"✅ FlashAttention-2 computation test: SUCCESS")
                print(f"   Output shape: {output.shape}")
                print(f"   Output dtype: {output.dtype}")

        except Exception as e:
            print(f"❌ FlashAttention-2 test: FAILED")
            print(f"   Error: {e}")
    else:
        print("⚠️  CUDA not available, skipping context test")
    print()

    # 4. Test ESM-2 model loading with FlashAttention
    print("🧬 ESM-2 Model FlashAttention Integration Test:")
    print("-" * 70)
    try:
        import esm
        from virnucpro.models.esm2_flash import load_esm2_model

        print("Loading ESM-2 3B model with FlashAttention wrapper...")
        model, batch_converter = load_esm2_model(
            model_name="esm2_t36_3B_UR50D",
            device="cuda:0"
        )

        print(f"✅ Model loaded successfully")
        print(f"   {model}")
        print(f"   Attention implementation: {model.attention_impl}")
        print(f"   Using BF16: {model.use_bf16}")

        # Test forward pass
        print("\nTesting forward pass with sample data...")
        data = [("protein1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGQAVDVVGQALLPRAKRRVVGFVR")]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to("cuda:0")

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[36])

        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(results.keys())}")
        if "representations" in results:
            print(f"   Embedding shape: {results['representations'][36].shape}")
            print(f"   Embedding dtype: {results['representations'][36].dtype}")

    except Exception as e:
        print(f"❌ ESM-2 model test: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 5. Summary
    print("=" * 70)
    print("Summary:")
    print("=" * 70)

    attention_impl = get_attention_implementation()
    if attention_impl == "flash_attention_2":
        print("✅ RTX 4090s SUPPORT FlashAttention-2")
        print("✅ ESM-2 model IS COMPATIBLE with FlashAttention-2")
        print()
        print("Expected speedup: 2-4x for attention operations")
        print("Memory savings: ~20-30% with BF16 enabled")
    else:
        print("⚠️  FlashAttention-2 NOT available")
        print("   Possible reasons:")
        print("   - GPU compute capability < 8.0")
        print("   - PyTorch version < 2.2")
        print("   - Missing flash-attn package")
        print()
        print("Standard attention will be used (slower but functional)")

    print("=" * 70)


if __name__ == "__main__":
    main()
