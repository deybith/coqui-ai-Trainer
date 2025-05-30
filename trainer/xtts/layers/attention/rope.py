"""
Enhanced XTTS Rotary Position Embedding (RoPE) Implementation

This module implements Rotary Position Embeddings for XTTS,
providing better positional encoding for long audio sequences.

Key features:
- Relative position information
- Better extrapolation to longer sequences
- Improved temporal modeling for audio
- Integration with attention mechanisms

Based on:
- RoFormer: Enhanced Transformer with Rotary Position Embedding
- Optimized implementations for audio processing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from einops import rearrange, repeat


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation.
    
    Applies rotary transformations to query and key embeddings
    to encode relative positional information.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        rope_type: str = "default",  # "default", "linear", "dynamic"
        interpolation_factor: float = 1.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        self.interpolation_factor = interpolation_factor
        
        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cosine and sine values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequencies for rotary embedding."""
        if self.rope_type == "linear":
            # Linear scaling for better extrapolation
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            inv_freq = inv_freq / self.scaling_factor
        elif self.rope_type == "dynamic":
            # Dynamic scaling based on sequence length
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
        else:
            # Default RoPE
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
        
        return inv_freq
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cosine and sine values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            
            # Create position indices
            t = torch.arange(self._seq_len_cached, device=device, dtype=dtype)
            
            if self.rope_type == "dynamic" and seq_len > self.max_position_embeddings:
                # Dynamic scaling for sequences longer than training
                scale = seq_len / self.max_position_embeddings
                t = t / scale
            
            # Compute frequencies
            freqs = torch.outer(t, self.inv_freq)
            
            # Apply interpolation if needed
            if self.interpolation_factor != 1.0:
                freqs = freqs / self.interpolation_factor
            
            # Compute cosine and sine
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute rotary embeddings.
        
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, heads, seq_len, dim_head]
            seq_len: Sequence length (optional, inferred from x if not provided)
            position_ids: Custom position indices (optional)
            
        Returns:
            Tuple of (cos, sin) tensors for rotary transformation
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Update cache if necessary
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        if position_ids is not None:
            # Use custom position indices
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]
        else:
            # Use sequential position indices
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input tensor.
    
    Args:
        x: Input tensor [..., dim]
        
    Returns:
        Rotated tensor [..., dim]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, heads, seq_len, dim_head]
        k: Key tensor [batch, heads, seq_len, dim_head]
        cos: Cosine values [seq_len, dim_head]
        sin: Sine values [seq_len, dim_head]
        position_ids: Custom position indices (optional)
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
        
    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Ensure cos and sin have the right shape for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotary transformation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RoPEAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embedding.
    
    Drop-in replacement for standard attention with better positional encoding.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        rope_scaling: Optional[dict] = None,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embedding
        rope_config = rope_scaling or {}
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            **rope_config
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with RoPE attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key-value pairs
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value caching
            
        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        query_states = self.q_proj(x)
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key-value pairs for caching
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if len(past_key_value) == 2:
                kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary position embedding
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Concatenate past key-value pairs
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class RoPEMultiHeadAttention(nn.Module):
    """
    Enhanced multi-head attention with RoPE for XTTS integration.
    
    Includes additional features for TTS-specific requirements.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 8192,  # Longer for audio sequences
        rope_scaling: Optional[dict] = None,
        use_bias: bool = True,
        attention_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_flash_attention: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        
        # Core attention
        self.attention = RoPEAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
            rope_scaling=rope_scaling,
            bias=use_bias,
        )
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional: integrate with Flash Attention if available
        if use_flash_attention:
            try:
                from trainer.xtts.layers.attention.flash_attention import FlashAttention2
                self.flash_attn = FlashAttention2(
                    dim=d_model,
                    heads=num_heads,
                    dropout=attention_dropout,
                    **kwargs
                )
            except ImportError:
                print("⚠️  Flash Attention not available. Using standard RoPE attention.")
                self.flash_attn = None
        else:
            self.flash_attn = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with enhanced RoPE attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key-value pairs
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value caching
            
        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        residual = hidden_states
        
        # Pre-normalization
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply attention (RoPE or Flash+RoPE)
        if self.flash_attn is not None and not output_attentions and not use_cache:
            # Use Flash Attention for efficiency (when possible)
            attention_output = self.flash_attn(hidden_states, mask=attention_mask)
            outputs = (attention_output, None, None)
        else:
            # Use standard RoPE attention
            outputs = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        attention_output = outputs[0]
        
        # Residual connection and dropout
        attention_output = self.dropout(attention_output)
        hidden_states = residual + attention_output
        
        # Post-normalization
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        return (hidden_states,) + outputs[1:]


def create_rope_attention_layer(
    d_model: int,
    num_heads: int,
    max_position_embeddings: int = 8192,
    rope_type: str = "default",
    scaling_factor: float = 1.0,
    use_flash_attention: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create RoPE attention layers.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_position_embeddings: Maximum sequence length
        rope_type: Type of RoPE scaling ("default", "linear", "dynamic")
        scaling_factor: Scaling factor for RoPE
        use_flash_attention: Whether to integrate with Flash Attention
        **kwargs: Additional layer arguments
        
    Returns:
        RoPE attention layer ready for use
    """
    rope_scaling = {
        "rope_type": rope_type,
        "scaling_factor": scaling_factor,
    }
    
    return RoPEMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=rope_scaling,
        use_flash_attention=use_flash_attention,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test RoPE implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    seq_len = 1024
    d_model = 512
    num_heads = 8
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"🔄 Testing Rotary Position Embedding (RoPE)")
    print(f"Input shape: {x.shape}")
    
    # Test basic RoPE
    rope_emb = RotaryEmbedding(dim=d_model // num_heads).to(device)
    
    with torch.no_grad():
        cos, sin = rope_emb(x)
        print(f"RoPE cos/sin shapes: {cos.shape}, {sin.shape}")
    
    # Test RoPE attention
    rope_attn = create_rope_attention_layer(
        d_model=d_model,
        num_heads=num_heads,
        max_position_embeddings=2048,
        rope_type="linear",
        scaling_factor=1.0
    ).to(device)
    
    with torch.no_grad():
        outputs = rope_attn(x)
        output = outputs[0]
        print(f"RoPE attention output shape: {output.shape}")
    
    # Test with Flash Attention integration
    try:
        rope_flash_attn = create_rope_attention_layer(
            d_model=d_model,
            num_heads=num_heads,
            use_flash_attention=True
        ).to(device)
        
        with torch.no_grad():
            outputs = rope_flash_attn(x)
            output = outputs[0]
            print(f"RoPE + Flash attention output shape: {output.shape}")
            print(f"✅ RoPE with Flash Attention integration successful!")
    except Exception as e:
        print(f"⚠️  RoPE + Flash Attention integration not available: {e}")
    
    print(f"✅ Rotary Position Embedding test completed successfully!")
