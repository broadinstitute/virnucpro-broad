#!/usr/bin/env python3
"""Quick verification that BF16 dtype fix works for DNABERT-S mean pooling."""

import torch

def test_mean_pooling_bf16():
    """Test that mean pooling works with BF16 hidden_states and int64 attention_mask."""

    # Simulate BF16 hidden states (from model output)
    batch_size = 2
    seq_len = 10
    hidden_dim = 768

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)

    # Set some positions to 0 (padding)
    attention_mask[0, 8:] = 0
    attention_mask[1, 9:] = 0

    print(f"hidden_states dtype: {hidden_states.dtype}")
    print(f"attention_mask dtype: {attention_mask.dtype}")

    # OLD CODE (should fail):
    try:
        embedding_means_old = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        print("❌ OLD CODE: Unexpectedly succeeded (dtype mismatch should fail)")
    except RuntimeError as e:
        print(f"✓ OLD CODE: Failed as expected with: {str(e)[:80]}")

    # NEW CODE (should work):
    try:
        # Convert attention_mask to match hidden_states dtype (for BF16 compatibility)
        attention_mask_typed = attention_mask.to(hidden_states.dtype)
        embedding_means = (hidden_states * attention_mask_typed.unsqueeze(-1)).sum(dim=1) / attention_mask_typed.sum(dim=1, keepdim=True)

        print(f"✓ NEW CODE: Success! embedding_means shape: {embedding_means.shape}, dtype: {embedding_means.dtype}")

        # Verify the result makes sense
        assert embedding_means.shape == (batch_size, hidden_dim)
        assert embedding_means.dtype == torch.bfloat16
        print("✓ All assertions passed")

    except Exception as e:
        print(f"❌ NEW CODE: Failed with: {e}")
        return False

    return True

if __name__ == "__main__":
    print("Testing BF16 mean pooling fix for DNABERT-S...")
    print("=" * 80)

    if test_mean_pooling_bf16():
        print("\n" + "=" * 80)
        print("✓ VERIFICATION PASSED: BF16 dtype fix works correctly")
        print("=" * 80)
        exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        exit(1)
