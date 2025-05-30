"""
Streaming Architecture for Real-Time XTTS Inference

This module implements streaming-friendly components that enable real-time
text-to-speech generation with minimal latency and memory usage.

Key Features:
- Chunked processing with overlapping windows
- Streaming transformer layers
- Dynamic attention with limited context
- Real-time quality monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)


class StreamingBuffer:
    """
    Circular buffer for streaming audio processing.
    
    Manages overlapping chunks of audio data for continuous processing
    while maintaining temporal coherence.
    """
    
    def __init__(
        self,
        buffer_size: int = 4096,
        chunk_size: int = 1024,
        overlap_size: int = 256,
        device: torch.device = torch.device('cpu'),
    ):
        self.buffer_size = buffer_size
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.device = device
        
        # Initialize buffer
        self.buffer = deque(maxlen=buffer_size)
        self.current_chunk = None
        self.overlap_buffer = None
        
    def add_chunk(self, chunk: torch.Tensor) -> bool:
        """
        Add new chunk to buffer.
        
        Args:
            chunk: New audio chunk [chunk_size, features]
            
        Returns:
            bool: True if ready for processing
        """
        self.buffer.append(chunk)
        
        # Check if we have enough data for processing
        total_samples = sum(c.shape[0] for c in self.buffer)
        return total_samples >= self.chunk_size
    
    def get_processing_chunk(self) -> Optional[torch.Tensor]:
        """
        Get next chunk for processing with proper overlap handling.
        
        Returns:
            Processing chunk with overlap or None if not ready
        """
        if len(self.buffer) == 0:
            return None
            
        # Concatenate available buffer data
        buffer_data = torch.cat(list(self.buffer), dim=0)
        
        if buffer_data.shape[0] < self.chunk_size:
            return None
            
        # Extract chunk with overlap
        if self.overlap_buffer is not None:
            # Combine with previous overlap
            chunk_data = torch.cat([self.overlap_buffer, buffer_data[:self.chunk_size]], dim=0)
        else:
            chunk_data = buffer_data[:self.chunk_size + self.overlap_size]
            
        # Update overlap buffer for next iteration
        self.overlap_buffer = buffer_data[self.chunk_size - self.overlap_size:self.chunk_size]
        
        # Remove processed data from buffer
        remaining_data = buffer_data[self.chunk_size:]
        self.buffer.clear()
        if remaining_data.shape[0] > 0:
            self.buffer.append(remaining_data)
            
        return chunk_data
    
    def reset(self):
        """Reset buffer state."""
        self.buffer.clear()
        self.overlap_buffer = None


class StreamingMultiHeadAttention(nn.Module):
    """
    Streaming-friendly multi-head attention with limited context window.
    
    Implements efficient attention computation for real-time processing
    by maintaining a sliding window of past context.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_window: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context_window = context_window
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Context buffers for streaming
        self.register_buffer('key_buffer', torch.zeros(1, context_window, embed_dim))
        self.register_buffer('value_buffer', torch.zeros(1, context_window, embed_dim))
        self.buffer_position = 0
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        streaming: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming mode.
        
        Args:
            query: Query tensor [B, T, D]
            key: Key tensor [B, T, D] (optional, defaults to query)
            value: Value tensor [B, T, D] (optional, defaults to query)
            streaming: Enable streaming mode with context buffer
            
        Returns:
            Attention output [B, T, D]
        """
        B, T, D = query.shape
        
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        if streaming:
            # Update context buffers
            self._update_buffers(k, v)
            # Use buffered context for attention
            k_full = self.key_buffer[:, :self.buffer_position]
            v_full = self.value_buffer[:, :self.buffer_position]
        else:
            k_full = k
            v_full = v
            
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_full = k_full.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_full = v_full.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
        
        # Apply causal mask for autoregressive generation
        if streaming:
            seq_len = k_full.shape[-2]
            causal_mask = torch.triu(torch.ones(T, seq_len, device=query.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_full)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(attn_output)
        
        return output
    
    def _update_buffers(self, key: torch.Tensor, value: torch.Tensor):
        """Update key-value buffers for streaming."""
        B, T, D = key.shape
        
        # Expand buffers if needed
        if self.key_buffer.shape[0] < B:
            self.key_buffer = self.key_buffer.expand(B, -1, -1).contiguous()
            self.value_buffer = self.value_buffer.expand(B, -1, -1).contiguous()
        
        # Add new key-value pairs to buffer
        if self.buffer_position + T <= self.context_window:
            self.key_buffer[:B, self.buffer_position:self.buffer_position + T] = key
            self.value_buffer[:B, self.buffer_position:self.buffer_position + T] = value
            self.buffer_position += T
        else:
            # Shift buffer and add new data
            shift_amount = T
            self.key_buffer[:B, :-shift_amount] = self.key_buffer[:B, shift_amount:]
            self.value_buffer[:B, :-shift_amount] = self.value_buffer[:B, shift_amount:]
            self.key_buffer[:B, -shift_amount:] = key
            self.value_buffer[:B, -shift_amount:] = value
            self.buffer_position = min(self.buffer_position, self.context_window)
    
    def reset_buffers(self):
        """Reset context buffers for new sequence."""
        self.key_buffer.zero_()
        self.value_buffer.zero_()
        self.buffer_position = 0


class StreamingTransformerBlock(nn.Module):
    """
    Streaming-friendly transformer block with efficient memory usage.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        context_window: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = StreamingMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            context_window=context_window,
            dropout=dropout,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        """Forward pass with optional streaming mode."""
        # Self-attention with residual connection
        attn_out = self.self_attn(x, streaming=streaming)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
    
    def reset_streaming_state(self):
        """Reset streaming state for new sequence."""
        self.self_attn.reset_buffers()


class StreamingXTTSDecoder(nn.Module):
    """
    Streaming-capable XTTS decoder for real-time inference.
    
    This decoder implements chunked processing with minimal latency
    while maintaining high-quality audio generation.
    """
    
    def __init__(
        self,
        vocab_size: int = 8194,
        embed_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        ff_dim: int = 4096,
        context_window: int = 512,
        chunk_size: int = 128,
        overlap_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_window * 2, embed_dim)
        
        # Streaming transformer layers
        self.layers = nn.ModuleList([
            StreamingTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                context_window=context_window,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Streaming buffer
        self.streaming_buffer = StreamingBuffer(
            buffer_size=context_window,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )
        
        # Position tracking for streaming
        self.current_position = 0
        
    def forward(
        self,
        input_ids: torch.Tensor,
        streaming: bool = False,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional streaming mode.
        
        Args:
            input_ids: Input token IDs [B, T]
            streaming: Enable streaming mode
            return_hidden_states: Return intermediate hidden states
            
        Returns:
            Dictionary with logits and optional hidden states
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        if streaming:
            positions = torch.arange(
                self.current_position,
                self.current_position + T,
                device=device
            ).unsqueeze(0).expand(B, -1)
            self.current_position += T
        else:
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        hidden_states = self.dropout(token_embeds + pos_embeds)
        
        # Store intermediate hidden states if requested
        all_hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, streaming=streaming)
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        result = {'logits': logits}
        if return_hidden_states:
            result['hidden_states'] = all_hidden_states
            
        return result
    
    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate tokens in streaming mode with minimal latency.
        
        Args:
            input_ids: Initial input tokens [B, T]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token sequence
        """
        self.reset_streaming_state()
        
        B = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # Get next token logits
            with torch.no_grad():
                outputs = self.forward(generated[:, -1:], streaming=True)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits.masked_fill_(indices_to_remove, float('-inf'))
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end-of-sequence token (assuming 0 is EOS)
                if (next_token == 0).all():
                    break
        
        return generated
    
    def reset_streaming_state(self):
        """Reset all streaming state for new sequence."""
        for layer in self.layers:
            layer.reset_streaming_state()
        self.streaming_buffer.reset()
        self.current_position = 0


class RealTimeQualityMonitor(nn.Module):
    """
    Real-time quality monitoring for streaming TTS.
    
    Provides online assessment of audio quality metrics to enable
    adaptive quality control during streaming generation.
    """
    
    def __init__(
        self,
        feature_dim: int = 80,
        hidden_dim: int = 256,
        quality_threshold: float = 0.7,
    ):
        super().__init__()
        self.quality_threshold = quality_threshold
        
        # Quality assessment network
        self.quality_net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Running statistics
        self.register_buffer('quality_history', torch.zeros(100))
        self.history_position = 0
        
    def forward(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Assess quality of current audio features.
        
        Args:
            features: Audio features [B, T, D]
            
        Returns:
            Quality metrics dictionary
        """
        # Transpose for conv1d
        features_conv = features.transpose(1, 2)
        
        # Compute quality score
        quality_score = self.quality_net(features_conv).mean().item()
        
        # Update history
        self.quality_history[self.history_position] = quality_score
        self.history_position = (self.history_position + 1) % self.quality_history.shape[0]
        
        # Compute metrics
        avg_quality = self.quality_history.mean().item()
        quality_trend = (quality_score - avg_quality) / (avg_quality + 1e-8)
        
        return {
            'current_quality': quality_score,
            'average_quality': avg_quality,
            'quality_trend': quality_trend,
            'needs_adjustment': quality_score < self.quality_threshold,
        }


# Factory functions
def create_streaming_decoder(
    vocab_size: int = 8194,
    embed_dim: int = 1024,
    num_layers: int = 12,
    num_heads: int = 16,
    context_window: int = 512,
    chunk_size: int = 128,
) -> StreamingXTTSDecoder:
    """Create streaming decoder with optimal settings."""
    return StreamingXTTSDecoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        context_window=context_window,
        chunk_size=chunk_size,
    )


def create_quality_monitor(
    feature_dim: int = 80,
    quality_threshold: float = 0.7,
) -> RealTimeQualityMonitor:
    """Create real-time quality monitor."""
    return RealTimeQualityMonitor(
        feature_dim=feature_dim,
        quality_threshold=quality_threshold,
    )
