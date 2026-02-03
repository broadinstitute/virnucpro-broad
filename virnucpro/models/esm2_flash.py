"""ESM-2 model wrapper with FlashAttention-2 integration.

This module provides a wrapper around the fair-esm ESM-2 model that automatically
uses FlashAttention-2 on compatible GPUs (Ampere+) with transparent fallback to
standard attention on older hardware.

FlashAttention-2 provides 2-4x attention speedup for transformer models without
changing model outputs or accuracy.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any, Literal
import logging
import esm

from virnucpro.cuda.attention_utils import (
    get_attention_implementation,
    configure_flash_attention
)
from virnucpro.models.packed_attention import (
    create_position_ids_packed,
    flash_attn_varlen_wrapper,
    FLASH_ATTN_AVAILABLE,
)

logger = logging.getLogger('virnucpro.models.esm2_flash')


class ESM2WithFlashAttention(nn.Module):
    """
    Wrapper for ESM-2 model with FlashAttention-2 optimization.

    This wrapper automatically detects GPU capabilities and uses FlashAttention-2
    when available (Ampere+ GPUs with PyTorch 2.2+), falling back gracefully to
    standard attention on older hardware.

    The wrapper maintains full compatibility with the original ESM-2 model interface
    while providing 2-4x attention speedup on compatible GPUs.

    Attributes:
        model: The underlying ESM-2 model
        attention_impl: Current attention implementation ("flash_attention_2" or "standard_attention")
        device: Device the model is running on

    Example:
        >>> model = ESM2WithFlashAttention(base_model, device="cuda:0")
        >>> # Model automatically uses FlashAttention-2 if available
        >>> output = model(tokens)
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: torch.device,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize ESM-2 wrapper with FlashAttention-2 optimization.

        Args:
            base_model: Base ESM-2 model from esm.pretrained
            device: Device to run model on
            logger_instance: Optional logger for configuration messages
        """
        super().__init__()

        self.model = base_model
        self.device = device
        log = logger_instance if logger_instance is not None else logger

        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()

        # Detect and configure attention implementation
        self.attention_impl = get_attention_implementation()

        # Configure FlashAttention-2 if available
        self.model = configure_flash_attention(self.model, log)

        # EXPERIMENTAL: Force FP32 to test vanilla compatibility
        # Original code checked capability[0] >= 8 for BF16
        self.use_bf16 = False
        log.info("EXPERIMENTAL: Forcing FP32 precision (BF16 disabled for vanilla compatibility test)")

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: Optional[list] = None,
        return_contacts: bool = False
    ) -> dict:
        """
        Forward pass through ESM-2 model with optimal attention implementation.

        For FlashAttention-2, uses PyTorch's scaled_dot_product_attention context manager
        to ensure flash kernels are used. For standard attention, uses default behavior.

        Args:
            tokens: Input token IDs, shape (batch_size, seq_len)
            repr_layers: List of layer indices to return representations for
            return_contacts: Whether to return contact predictions

        Returns:
            Dictionary with model outputs including representations

        Example:
            >>> with torch.no_grad():
            ...     output = model(batch_tokens, repr_layers=[36])
            ...     embeddings = output["representations"][36]
        """
        if repr_layers is None:
            repr_layers = [36]  # Default to final layer for ESM-2

        # Use FlashAttention-2 context manager if available
        if self.attention_impl == "flash_attention_2":
            # Enable only FlashAttention kernel, disable fallbacks
            # Use new API if available (PyTorch 2.5+), fall back to deprecated API (PyTorch 2.2-2.4)
            try:
                if hasattr(torch.nn.attention, 'sdpa_kernel') and hasattr(torch.nn.attention, 'SDPBackend'):
                    from torch.nn.attention import SDPBackend
                    # New API uses SDPBackend enum
                    with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                        return self.model(
                            tokens,
                            repr_layers=repr_layers,
                            return_contacts=return_contacts
                        )
                else:
                    # Old deprecated API
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=False
                    ):
                        return self.model(
                            tokens,
                            repr_layers=repr_layers,
                            return_contacts=return_contacts
                        )
            except Exception:
                # If FlashAttention context fails, fall back to standard path
                return self.model(
                    tokens,
                    repr_layers=repr_layers,
                    return_contacts=return_contacts
                )
        else:
            # Standard attention path
            return self.model(
                tokens,
                repr_layers=repr_layers,
                return_contacts=return_contacts
            )

    def forward_packed(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        repr_layers: Optional[list] = None,
    ) -> dict:
        """
        Forward pass for packed sequences using FlashAttention varlen.

        This method processes packed batches (1D concatenated tokens + cu_seqlens)
        through ESM-2 using FlashAttention's varlen API. Packed format enables
        2-3x throughput improvement over padded batches by eliminating padding tokens.

        Args:
            input_ids: 1D tensor of concatenated token IDs [total_tokens]
            cu_seqlens: Cumulative sequence lengths [num_sequences + 1], dtype=int32
                        Format: [0, len1, len1+len2, ...] where each element is
                        the starting index of a sequence in input_ids
            max_seqlen: Maximum sequence length in batch
            repr_layers: Layer indices to return representations for (default: [36])

        Returns:
            Dictionary with 'representations' key containing layer outputs
            Format: {'representations': {layer_idx: tensor}}
            Tensor shape: [total_tokens, hidden_dim] (packed format)

        Example:
            >>> # Three sequences: lengths [3, 4, 3]
            >>> input_ids = torch.tensor([1,2,3, 4,5,6,7, 8,9,10], device='cuda')
            >>> cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32, device='cuda')
            >>> output = model.forward_packed(input_ids, cu_seqlens, max_seqlen=4)
            >>> embeddings = output['representations'][36]  # Shape: [10, 2560]
        """
        if repr_layers is None:
            repr_layers = [36]  # Default to final layer for ESM-2 3B

        # Check FlashAttention availability - use fallback if not available
        if not FLASH_ATTN_AVAILABLE:
            logger.warning(
                "FlashAttention not available. Using fallback unpack/repack strategy. "
                "Install flash-attn for 2-3x speedup: pip install flash-attn --no-build-isolation"
            )
            return self._forward_packed_fallback(input_ids, cu_seqlens, max_seqlen, repr_layers)

        # FlashAttention path
        batch_size = len(cu_seqlens) - 1

        # Create position IDs that reset at each sequence boundary
        position_ids = create_position_ids_packed(cu_seqlens)

        # Get token embeddings - shape: [total_tokens, hidden_dim]
        embeddings = self.model.embed_tokens(input_ids)

        # FlashAttention requires FP16/BF16 - auto-convert if needed
        if embeddings.dtype not in [torch.float16, torch.bfloat16]:
            logger.warning(
                f"Model loaded in {embeddings.dtype}, but FlashAttention requires FP16/BF16. "
                "Auto-converting to BF16 for packed inference. "
                "For best performance, load model with: model.to(dtype=torch.bfloat16)"
            )
            # Convert model to BF16 (better numerical stability than FP16 for this model)
            self.model = self.model.to(dtype=torch.bfloat16)
            embeddings = self.model.embed_tokens(input_ids)

        # NOTE: ESM-2 uses Rotary Position Embeddings (RoPE), not learned position embeddings
        # Position information is applied inside attention (to Q/K), not as a separate embedding
        # The position_ids are passed to each layer and applied during attention computation

        # Process through transformer layers
        hidden_states = embeddings
        representations = {}

        for layer_idx, layer in enumerate(self.model.layers):
            hidden_states = self._layer_forward_packed(
                layer, hidden_states, cu_seqlens, max_seqlen, position_ids
            )

            # Store representation if this layer was requested
            if layer_idx in repr_layers:
                representations[layer_idx] = hidden_states.clone()

        # Apply final layer norm
        hidden_states = self.model.emb_layer_norm_after(hidden_states)

        # Store final layer if requested (common case for ESM-2 3B is layer 36)
        if 36 in repr_layers or len(self.model.layers) in repr_layers:
            layer_idx = len(self.model.layers)
            representations[layer_idx] = hidden_states

        return {'representations': representations}

    def _layer_forward_packed(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single transformer layer forward pass for packed sequences.

        This method processes a single ESM-2 transformer layer using FlashAttention
        varlen for the attention computation. It handles the self-attention and
        feed-forward sublayers with residual connections.

        ESM-2 uses Rotary Position Embeddings (RoPE) which are applied to Q and K
        inside the attention computation.

        Args:
            layer: ESM-2 TransformerEncoderLayer
            hidden_states: [total_tokens, hidden_dim]
            cu_seqlens: [batch_size + 1] cumulative sequence lengths
            max_seqlen: Maximum sequence length in batch
            position_ids: [total_tokens] position indices that reset at sequence boundaries

        Returns:
            hidden_states: [total_tokens, hidden_dim] after layer processing
        """
        # Self-attention sublayer with residual
        residual = hidden_states
        hidden_states = layer.self_attn_layer_norm(hidden_states)

        # Compute Q, K, V from self-attention
        # ESM-2 uses separate projection layers (q_proj, k_proj, v_proj), not combined in_proj_weight
        hidden_dim = layer.self_attn.embed_dim
        num_heads = layer.self_attn.num_heads
        head_dim = hidden_dim // num_heads

        # Project to Q, K, V using separate linear layers
        # Shape: [total_tokens, hidden_dim] -> [total_tokens, num_heads, head_dim]
        q = layer.self_attn.q_proj(hidden_states).reshape(-1, num_heads, head_dim)
        k = layer.self_attn.k_proj(hidden_states).reshape(-1, num_heads, head_dim)
        v = layer.self_attn.v_proj(hidden_states).reshape(-1, num_heads, head_dim)

        # Apply Rotary Position Embeddings (RoPE) to Q and K
        # ESM-2's rot_emb applies position information to query and key
        if hasattr(layer.self_attn, 'rot_emb') and layer.self_attn.rot_emb is not None:
            # rot_emb takes q, k and applies rotary embeddings based on position_ids
            # Expected signature: rot_emb(q, k) where positions come from sequence structure
            # For packed sequences, we need to pass position_ids that reset at boundaries
            q, k = layer.self_attn.rot_emb(q, k, position_ids)

        # FlashAttention varlen - automatically prevents cross-sequence attention
        attn_output = flash_attn_varlen_wrapper(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,  # No dropout for inference
            causal=False,   # Bidirectional attention for ESM/BERT
        )

        # Reshape back: [total_tokens, num_heads, head_dim] -> [total_tokens, hidden_dim]
        attn_output = attn_output.reshape(-1, hidden_dim)

        # Apply output projection
        hidden_states = layer.self_attn.out_proj(attn_output)

        # Residual connection
        hidden_states = residual + hidden_states

        # Feed-forward sublayer with residual
        residual = hidden_states
        hidden_states = layer.final_layer_norm(hidden_states)
        hidden_states = layer.fc1(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = layer.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_packed_fallback(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        repr_layers: Optional[list] = None,
    ) -> dict:
        """
        Fallback forward pass for packed sequences when FlashAttention unavailable.

        This method unpacks the 1D input_ids to 2D padded format, runs the standard
        forward pass, and repacks the output. Less efficient than FlashAttention varlen
        but ensures correctness on systems without flash-attn (e.g., older GPUs, CI).

        Args:
            input_ids: 1D tensor of concatenated token IDs [total_tokens]
            cu_seqlens: Cumulative sequence lengths [num_sequences + 1]
            max_seqlen: Maximum sequence length in batch
            repr_layers: Layer indices to return representations for

        Returns:
            Dictionary with 'representations' key containing packed layer outputs
        """
        batch_size = len(cu_seqlens) - 1

        # Unpack 1D input_ids to 2D padded tensor
        # Shape: [batch_size, max_seqlen]
        padded_tokens = torch.zeros(
            batch_size, max_seqlen,
            dtype=input_ids.dtype,
            device=input_ids.device
        )

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            padded_tokens[i, :seq_len] = input_ids[start:end]

        # Run standard forward pass with padded format
        output = self.forward(padded_tokens, repr_layers=repr_layers)

        # Repack output to 1D using cu_seqlens boundaries
        representations = {}
        for layer_idx, padded_repr in output['representations'].items():
            # Shape: [batch_size, max_seqlen, hidden_dim]
            total_tokens = cu_seqlens[-1].item()
            hidden_dim = padded_repr.shape[-1]

            packed_repr = torch.zeros(
                total_tokens, hidden_dim,
                dtype=padded_repr.dtype,
                device=padded_repr.device
            )

            for i in range(batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_len = end - start

                packed_repr[start:end] = padded_repr[i, :seq_len]

            representations[layer_idx] = packed_repr

        return {'representations': representations}

    def __repr__(self) -> str:
        """String representation showing attention implementation."""
        return (
            f"ESM2WithFlashAttention("
            f"attention={self.attention_impl}, "
            f"device={self.device}, "
            f"dtype={'bfloat16' if self.use_bf16 else 'float32'})"
        )


def load_esm2_model(
    model_name: Literal[
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D"
    ] = "esm2_t36_3B_UR50D",
    device: str = "cuda",
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[ESM2WithFlashAttention, Any]:
    """
    Load ESM-2 model with FlashAttention-2 optimization.

    This function loads an ESM-2 model and wraps it with FlashAttention-2 support,
    automatically detecting GPU capabilities and configuring optimal attention
    implementation.

    Supported models:
    - esm2_t33_650M_UR50D: 650M parameters, 33 layers
    - esm2_t36_3B_UR50D: 3B parameters, 36 layers (default)
    - esm2_t48_15B_UR50D: 15B parameters, 48 layers

    Args:
        model_name: Name of ESM-2 model variant to load
        device: Device to load model on (e.g., "cuda", "cuda:0", "cpu")
        logger_instance: Optional logger for configuration messages

    Returns:
        Tuple of (model, batch_converter):
        - model: ESM2WithFlashAttention wrapper
        - batch_converter: ESM alphabet batch converter for tokenization

    Example:
        >>> model, batch_converter = load_esm2_model("esm2_t33_650M_UR50D", "cuda:0")
        >>> # Model automatically uses FlashAttention-2 if available
        >>> sequences = [("protein1", "MKTAYIAK"), ("protein2", "VLSPADKTNV")]
        >>> labels, strs, tokens = batch_converter(sequences)
        >>> with torch.no_grad():
        ...     results = model(tokens.to("cuda:0"), repr_layers=[33])
    """
    log = logger_instance if logger_instance is not None else logger

    log.info(f"Loading ESM-2 model: {model_name}")

    # Load base ESM-2 model using fair-esm
    model_loader = getattr(esm.pretrained, model_name)
    base_model, alphabet = model_loader()

    log.info(f"Model loaded: {model_name}")

    # Convert device string to torch.device
    device_obj = torch.device(device)

    # Wrap with FlashAttention-2 support
    model = ESM2WithFlashAttention(
        base_model,
        device_obj,
        logger_instance=log
    )

    # Get batch converter from alphabet
    batch_converter = alphabet.get_batch_converter()

    log.info(f"ESM-2 ready: {model}")

    return model, batch_converter


def get_esm2_embeddings(
    model: ESM2WithFlashAttention,
    batch_converter: Any,
    sequences: list,
    layer: int = 36,
    truncation_length: int = 1024
) -> Tuple[list, list]:
    """
    Extract embeddings from ESM-2 model for a batch of sequences.

    This is a convenience function for extracting mean-pooled sequence embeddings.

    Args:
        model: ESM2WithFlashAttention model
        batch_converter: ESM alphabet batch converter
        sequences: List of (id, sequence) tuples
        layer: Layer index to extract representations from (default: 36 for 3B model)
        truncation_length: Maximum sequence length (sequences truncated beyond this)

    Returns:
        Tuple of (ids, embeddings):
        - ids: List of sequence IDs
        - embeddings: List of embedding tensors (mean-pooled, CPU)

    Example:
        >>> sequences = [("seq1", "MKTAYIAK"), ("seq2", "VLSPADKTNV")]
        >>> ids, embeddings = get_esm2_embeddings(model, batch_converter, sequences)
        >>> # embeddings[0].shape == (2560,) for 3B model
    """
    # Truncate sequences if needed
    truncated_sequences = []
    for seq_id, seq_str in sequences:
        if len(seq_str) > truncation_length:
            seq_str = seq_str[:truncation_length]
        truncated_sequences.append((seq_id, seq_str))

    # Convert batch
    labels, strs, tokens = batch_converter(truncated_sequences)
    tokens = tokens.to(model.device)

    # Forward pass
    with torch.no_grad():
        results = model(tokens, repr_layers=[layer])
        representations = results["representations"][layer]

    # Extract mean-pooled embeddings
    ids = []
    embeddings = []

    for i, (seq_id, seq_str) in enumerate(truncated_sequences):
        # Mean pool positions 1 to len+1 (skip BOS token)
        seq_len = min(len(seq_str), truncation_length)
        embedding = representations[i, 1:seq_len + 1].mean(dim=0)

        # Convert to FP32 and move to CPU for storage
        embedding = embedding.float().cpu()

        ids.append(seq_id)
        embeddings.append(embedding)

    return ids, embeddings
