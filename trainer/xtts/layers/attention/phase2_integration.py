"""
Enhanced XTTS Phase 2 Integration Layer

This module integrates all Phase 2 enhancements into a unified architecture:
- Mamba/State Space Models for linear complexity
- Flash Attention 2.0 for optimized computation  
- Rotary Position Embedding (RoPE) for better positional encoding
- Mixture of Experts (MoE) for scalable specialization

Key features:
- Modular architecture allowing selective enhancement usage
- Hybrid approaches combining multiple techniques
- Optimized for TTS-specific requirements
- Backward compatibility with Phase 1 components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass

# Import Phase 2 components
from .mamba import MambaBlock, MambaEncoder, HybridMambaAttention, create_mamba_layer
from .flash_attention import FlashAttention2, FlashMultiHeadAttention, create_flash_attention_layer
from .rope import RoPEMultiHeadAttention, create_rope_attention_layer
from .mixture_of_experts import MoELayer, create_moe_layer


@dataclass
class Phase2Config:
    """Configuration for Phase 2 enhancements."""
    
    # Model dimensions
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    
    # Mamba configuration
    use_mamba: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # Flash Attention configuration
    use_flash_attention: bool = True
    flash_block_size: int = 256
    flash_causal: bool = False
    
    # RoPE configuration
    use_rope: bool = True
    rope_max_position_embeddings: int = 8192
    rope_type: str = "linear"  # "default", "linear", "dynamic"
    rope_scaling_factor: float = 1.0
    
    # MoE configuration
    use_moe: bool = True
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_capacity: Optional[int] = None
    moe_auxiliary_loss_factor: float = 0.01
    
    # Hybrid configuration
    use_hybrid_layers: bool = True
    hybrid_mamba_attention_ratio: float = 0.6  # 60% Mamba, 40% Attention
    
    # General
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    use_pre_norm: bool = True
    
    # Specialization
    language_specialization: bool = False
    supported_languages: Optional[List[str]] = None


class Phase2EnhancedLayer(nn.Module):
    """
    Individual enhanced layer combining Phase 2 techniques.
    
    Can selectively use Mamba, Flash Attention, RoPE, and MoE
    based on configuration.
    """
    
    def __init__(
        self,
        config: Phase2Config,
        layer_idx: int = 0,
        is_decoder: bool = True,
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.is_decoder = is_decoder
        self.d_model = config.d_model
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # Attention mechanism selection
        self._build_attention_layer()
        
        # Feed-forward layer selection
        self._build_feedforward_layer()
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Auxiliary loss tracking
        self.auxiliary_losses = {}
    
    def _build_attention_layer(self):
        """Build the attention layer based on configuration."""
        if self.config.use_hybrid_layers and self.config.use_mamba:
            # Hybrid Mamba-Attention layer
            mamba_config = {
                'd_state': self.config.mamba_d_state,
                'd_conv': self.config.mamba_d_conv,
                'expand': self.config.mamba_expand,
                'layer_idx': self.layer_idx,
            }
            
            attention_config = {
                'num_heads': self.config.num_heads,
                'dropout': self.config.dropout,
            }
            
            self.attention = HybridMambaAttention(
                d_model=self.config.d_model,
                mamba_config=mamba_config,
                attention_config=attention_config,
                mix_ratio=self.config.hybrid_mamba_attention_ratio,
                use_gating=True,
            )
            
        elif self.config.use_mamba:
            # Pure Mamba layer
            self.attention = MambaBlock(
                d_model=self.config.d_model,
                d_state=self.config.mamba_d_state,
                d_conv=self.config.mamba_d_conv,
                expand=self.config.mamba_expand,
                layer_idx=self.layer_idx,
            )
            
        elif self.config.use_rope and self.config.use_flash_attention:
            # RoPE + Flash Attention
            self.attention = create_rope_attention_layer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                max_position_embeddings=self.config.rope_max_position_embeddings,
                rope_type=self.config.rope_type,
                scaling_factor=self.config.rope_scaling_factor,
                use_flash_attention=True,
                dropout=self.config.dropout,
            )
            
        elif self.config.use_flash_attention:
            # Flash Attention only
            self.attention = create_flash_attention_layer(
                d_model=self.config.d_model,
                n_heads=self.config.num_heads,
                dropout=self.config.dropout,
                causal=self.config.flash_causal,
                block_size=self.config.flash_block_size,
            )
            
        elif self.config.use_rope:
            # RoPE attention only
            self.attention = create_rope_attention_layer(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                max_position_embeddings=self.config.rope_max_position_embeddings,
                rope_type=self.config.rope_type,
                scaling_factor=self.config.rope_scaling_factor,
                dropout=self.config.dropout,
            )
            
        else:
            # Standard multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=self.config.d_model,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                batch_first=True,
            )
    
    def _build_feedforward_layer(self):
        """Build the feed-forward layer based on configuration."""
        if self.config.use_moe:
            # Mixture of Experts
            languages = None
            if self.config.language_specialization and self.config.supported_languages:
                languages = self.config.supported_languages
            
            self.feed_forward = create_moe_layer(
                d_model=self.config.d_model,
                num_experts=self.config.moe_num_experts,
                top_k=self.config.moe_top_k,
                d_ff=self.config.d_ff,
                specialization_type="language" if self.config.language_specialization else None,
                languages=languages,
                dropout=self.config.dropout,
                router_config={
                    'auxiliary_loss_factor': self.config.moe_auxiliary_loss_factor,
                }
            )
        else:
            # Standard feed-forward network
            self.feed_forward = nn.Sequential(
                nn.Linear(self.config.d_model, self.config.d_ff),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.d_ff, self.config.d_model),
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Forward pass through the enhanced layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cached key-value pairs
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (hidden_states, attention_weights, present_key_value, auxiliary_losses)
        """
        residual = hidden_states
        auxiliary_losses = {}
        
        # Pre-normalization
        if self.config.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Attention layer
        attention_output = None
        attention_weights = None
        present_key_value = None
        
        if isinstance(self.attention, (MambaBlock, HybridMambaAttention)):
            # Mamba-based attention
            if isinstance(self.attention, MambaBlock):
                attention_output, residual = self.attention(hidden_states, residual=residual)
            else:
                attention_output = self.attention(hidden_states, mask=attention_mask)
        
        elif hasattr(self.attention, '__class__') and 'RoPE' in self.attention.__class__.__name__:
            # RoPE attention
            outputs = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            attention_output = outputs[0]
            if len(outputs) > 1:
                attention_weights = outputs[1]
            if len(outputs) > 2:
                present_key_value = outputs[2]
        
        elif isinstance(self.attention, (FlashAttention2, FlashMultiHeadAttention)):
            # Flash Attention
            if hasattr(self.attention, 'forward') and 'mask' in self.attention.forward.__code__.co_varnames:
                attention_output = self.attention(hidden_states, mask=attention_mask)
            else:
                attention_output = self.attention(hidden_states)
        
        else:
            # Standard attention
            attention_output, attention_weights = self.attention(
                hidden_states, hidden_states, hidden_states,
                attn_mask=attention_mask,
                need_weights=output_attentions,
            )
        
        # Residual connection and dropout
        if attention_output is not None:
            attention_output = self.dropout(attention_output)
            if not isinstance(self.attention, MambaBlock):  # Mamba handles residual internally
                hidden_states = residual + attention_output
        
        # Post-attention normalization
        if not self.config.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # Feed-forward layer
        residual = hidden_states
        if self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        if isinstance(self.feed_forward, MoELayer):
            # MoE feed-forward
            ff_output, moe_aux_losses = self.feed_forward(
                hidden_states, 
                attention_mask=attention_mask,
                return_auxiliary_loss=True
            )
            auxiliary_losses.update(moe_aux_losses)
        else:
            # Standard feed-forward
            ff_output = self.feed_forward(hidden_states)
        
        # Residual connection
        ff_output = self.dropout(ff_output)
        hidden_states = residual + ff_output
        
        # Post-FF normalization
        if not self.config.use_pre_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        
        return hidden_states, attention_weights, present_key_value, auxiliary_losses


class Phase2EnhancedEncoder(nn.Module):
    """
    Multi-layer encoder with Phase 2 enhancements.
    
    Stacks multiple Phase2EnhancedLayer modules to create
    a complete enhanced encoder.
    """
    
    def __init__(
        self,
        config: Phase2Config,
        is_decoder: bool = False,
    ):
        super().__init__()
        
        self.config = config
        self.is_decoder = is_decoder
        
        # Embedding layers (if needed)
        self.embed_positions = None
        if not config.use_rope:
            # Use learnable position embeddings if not using RoPE
            self.embed_positions = nn.Embedding(
                config.rope_max_position_embeddings, 
                config.d_model
            )
        
        # Enhanced layers
        self.layers = nn.ModuleList([
            Phase2EnhancedLayer(
                config=config,
                layer_idx=i,
                is_decoder=is_decoder,
            )
            for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """
        Forward pass through the enhanced encoder.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional cached key-value pairs
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Enhanced encoder outputs
        """
        # Add position embeddings if not using RoPE
        if self.embed_positions is not None:
            if position_ids is None:
                position_ids = torch.arange(
                    hidden_states.shape[1], 
                    device=hidden_states.device,
                    dtype=torch.long
                ).unsqueeze(0)
            position_embeddings = self.embed_positions(position_ids)
            hidden_states = hidden_states + position_embeddings
        
        # Prepare outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_auxiliary_losses = {}
        next_decoder_cache = () if use_cache else None
        
        # Process through layers
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions and layer_outputs[1] is not None:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            if use_cache and layer_outputs[2] is not None:
                next_decoder_cache = next_decoder_cache + (layer_outputs[2],)
            
            # Collect auxiliary losses
            if layer_outputs[3]:
                for loss_name, loss_value in layer_outputs[3].items():
                    if loss_name not in all_auxiliary_losses:
                        all_auxiliary_losses[loss_name] = []
                    all_auxiliary_losses[loss_name].append(loss_value)
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Aggregate auxiliary losses
        aggregated_auxiliary_losses = {}
        for loss_name, loss_list in all_auxiliary_losses.items():
            if loss_list:
                aggregated_auxiliary_losses[loss_name] = torch.stack(loss_list).mean()
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_decoder_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
                "auxiliary_losses": aggregated_auxiliary_losses,
            }
        else:
            return (
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_attentions,
                aggregated_auxiliary_losses,
            )


def create_phase2_encoder(
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    use_mamba: bool = True,
    use_flash_attention: bool = True,
    use_rope: bool = True,
    use_moe: bool = True,
    language_specialization: bool = False,
    supported_languages: Optional[List[str]] = None,
    **kwargs
) -> Phase2EnhancedEncoder:
    """
    Factory function to create Phase 2 enhanced encoder.
    
    Args:
        d_model: Model dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        use_mamba: Whether to use Mamba/SSM
        use_flash_attention: Whether to use Flash Attention
        use_rope: Whether to use RoPE
        use_moe: Whether to use MoE
        language_specialization: Whether to use language-specific experts
        supported_languages: List of supported languages
        **kwargs: Additional configuration options
        
    Returns:
        Phase 2 enhanced encoder
    """
    config = Phase2Config(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        use_mamba=use_mamba,
        use_flash_attention=use_flash_attention,
        use_rope=use_rope,
        use_moe=use_moe,
        language_specialization=language_specialization,
        supported_languages=supported_languages,
        **kwargs
    )
    
    return Phase2EnhancedEncoder(config)


# Example usage and testing
if __name__ == "__main__":
    # Test Phase 2 integration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    seq_len = 512
    d_model = 512
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"🚀 Testing Phase 2 Enhanced Architecture")
    print(f"Input shape: {x.shape}")
    
    # Test with all enhancements
    enhanced_encoder = create_phase2_encoder(
        d_model=d_model,
        num_layers=4,
        num_heads=8,
        d_ff=2048,
        use_mamba=True,
        use_flash_attention=True,
        use_rope=True,
        use_moe=True,
        language_specialization=True,
        supported_languages=["en", "es", "fr", "de"],
    ).to(device)
    
    with torch.no_grad():
        outputs = enhanced_encoder(x, return_dict=True)
        
        print(f"Enhanced encoder output shape: {outputs['last_hidden_state'].shape}")
        if outputs['auxiliary_losses']:
            print(f"Auxiliary losses: {list(outputs['auxiliary_losses'].keys())}")
    
    # Test individual enhancement combinations
    print(f"\n🔍 Testing individual enhancement combinations:")
    
    # Mamba only
    mamba_encoder = create_phase2_encoder(
        d_model=d_model,
        num_layers=2,
        use_mamba=True,
        use_flash_attention=False,
        use_rope=False,
        use_moe=False,
    ).to(device)
    
    with torch.no_grad():
        outputs = mamba_encoder(x, return_dict=True)
        print(f"✅ Mamba-only encoder: {outputs['last_hidden_state'].shape}")
    
    # Flash Attention + RoPE
    flash_rope_encoder = create_phase2_encoder(
        d_model=d_model,
        num_layers=2,
        use_mamba=False,
        use_flash_attention=True,
        use_rope=True,
        use_moe=False,
    ).to(device)
    
    with torch.no_grad():
        outputs = flash_rope_encoder(x, return_dict=True)
        print(f"✅ Flash Attention + RoPE encoder: {outputs['last_hidden_state'].shape}")
    
    # MoE only
    moe_encoder = create_phase2_encoder(
        d_model=d_model,
        num_layers=2,
        use_mamba=False,
        use_flash_attention=False,
        use_rope=False,
        use_moe=True,
        moe_num_experts=4,
        moe_top_k=2,
    ).to(device)
    
    with torch.no_grad():
        outputs = moe_encoder(x, return_dict=True)
        print(f"✅ MoE-only encoder: {outputs['last_hidden_state'].shape}")
        if outputs['auxiliary_losses']:
            print(f"   MoE auxiliary losses: {list(outputs['auxiliary_losses'].keys())}")
    
    print(f"\n✅ Phase 2 Enhanced Architecture test completed successfully!")
