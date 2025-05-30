"""
Enhanced XTTS Mixture of Experts (MoE) Implementation

This module implements Mixture of Experts for XTTS, enabling
scalable specialization for different languages, voices, and styles.

Key features:
- Scalable model capacity without increasing inference cost
- Language-specific and style-specific experts
- Advanced routing mechanisms with load balancing
- Sparsely activated experts for efficiency

Based on:
- Switch Transformer: Scaling to Trillion Parameter Models
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- PaLM-2: Mixture-of-Experts improvements
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from einops import rearrange, reduce
import numpy as np


class Expert(nn.Module):
    """
    Individual expert network.
    
    Can be specialized for different languages, voices, or styles.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        bias: bool = True,
        expert_type: str = "ffn",  # "ffn", "conv", "attention"
        specialization: Optional[str] = None,  # "language", "voice", "style"
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.expert_type = expert_type
        self.specialization = specialization
        
        if expert_type == "ffn":
            # Standard feed-forward expert
            self.wi = nn.Linear(d_model, d_ff, bias=bias)
            self.wo = nn.Linear(d_ff, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            
            # Activation function
            if activation == "gelu":
                self.activation = F.gelu
            elif activation == "relu":
                self.activation = F.relu
            elif activation == "swish":
                self.activation = F.silu
            else:
                raise ValueError(f"Unknown activation: {activation}")
                
        elif expert_type == "conv":
            # Convolutional expert for local patterns
            self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1, bias=bias)
            self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=3, padding=1, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.activation = F.gelu
            
        elif expert_type == "attention":
            # Attention-based expert
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=d_model // 64,
                dropout=dropout,
                batch_first=True
            )
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=bias)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")
        
        # Specialization metadata
        self.metadata = {
            "expert_type": expert_type,
            "specialization": specialization,
            "d_model": d_model,
            "d_ff": d_ff
        }
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the expert.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            **kwargs: Additional arguments for specific expert types
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if self.expert_type == "ffn":
            # Standard feed-forward network
            hidden = self.activation(self.wi(x))
            hidden = self.dropout(hidden)
            output = self.wo(hidden)
            return output
            
        elif self.expert_type == "conv":
            # Convolutional processing
            # Transpose for conv1d: [batch, d_model, seq_len]
            x_conv = x.transpose(-1, -2)
            hidden = self.activation(self.conv1(x_conv))
            hidden = self.dropout(hidden)
            output = self.conv2(hidden)
            # Transpose back: [batch, seq_len, d_model]
            return output.transpose(-1, -2)
            
        elif self.expert_type == "attention":
            # Attention-based expert
            # Self-attention
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed-forward
            ffn_out = self.ffn(x)
            output = self.norm2(x + self.dropout(ffn_out))
            return output


class TopKRouter(nn.Module):
    """
    Top-K routing mechanism for selecting experts.
    
    Routes each token to the top-k experts based on learned routing weights.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        router_bias: bool = False,
        router_jitter_noise: float = 0.0,
        router_ignore_padding_tokens: bool = True,
        use_auxiliary_loss: bool = True,
        auxiliary_loss_factor: float = 0.01,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_loss_factor = auxiliary_loss_factor
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Router network
        self.router = nn.Linear(d_model, num_experts, bias=router_bias)
        
        # Initialize router weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize router parameters."""
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)
    
    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute router probabilities for each token."""
        # Add jitter noise during training for regularization
        if self.training and self.router_jitter_noise > 0:
            noise = torch.randn_like(hidden_states) * self.router_jitter_noise
            hidden_states = hidden_states + noise
        
        # Compute routing logits
        router_logits = self.router(hidden_states)
        
        # Convert to probabilities
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        return router_probs.to(hidden_states.dtype)
    
    def _compute_auxiliary_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.
        
        Encourages experts to be used equally often.
        """
        if attention_mask is not None:
            # Mask out padding tokens
            router_probs = router_probs * attention_mask.unsqueeze(-1)
            num_tokens = attention_mask.sum()
        else:
            num_tokens = router_probs.numel() // router_probs.shape[-1]
        
        # Fraction of tokens routed to each expert
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=router_probs.dtype, device=router_probs.device
        )
        tokens_per_expert.scatter_add_(
            0, expert_indices.view(-1), torch.ones_like(expert_indices.view(-1), dtype=router_probs.dtype)
        )
        tokens_per_expert = tokens_per_expert / num_tokens
        
        # Average router probability for each expert
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        
        # Auxiliary loss: encourage balance
        auxiliary_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * self.num_experts
        return auxiliary_loss
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Tuple of (routing_weights, expert_indices, auxiliary_losses)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states = hidden_states.view(-1, d_model)  # [batch*seq_len, d_model]
        
        # Compute router probabilities
        router_probs = self._compute_router_probabilities(hidden_states)
        
        # Select top-k experts
        routing_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize routing weights
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        
        # Compute auxiliary loss
        auxiliary_losses = {}
        if self.use_auxiliary_loss and self.training:
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1)
            auxiliary_losses["load_balancing_loss"] = (
                self._compute_auxiliary_loss(router_probs.view(batch_size, seq_len, -1), expert_indices, attention_mask)
                * self.auxiliary_loss_factor
            )
        
        return routing_weights, expert_indices, auxiliary_losses


class MixtureOfExperts(nn.Module):
    """
    Complete Mixture of Experts layer.
    
    Combines expert routing with parallel expert computation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_capacity: int = None,
        top_k: int = 2,
        d_ff: int = None,
        expert_configs: Optional[List[Dict]] = None,
        router_config: Optional[Dict] = None,
        use_expert_parallelism: bool = True,
        expert_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity or (d_model * 4) // num_experts
        self.use_expert_parallelism = use_expert_parallelism
        
        if d_ff is None:
            d_ff = d_model * 4
        
        # Create experts
        if expert_configs is not None:
            # Custom expert configurations
            assert len(expert_configs) == num_experts
            self.experts = nn.ModuleList([
                Expert(d_model=d_model, d_ff=d_ff, **config)
                for config in expert_configs
            ])
        else:
            # Default FFN experts
            self.experts = nn.ModuleList([
                Expert(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=expert_dropout,
                    expert_type="ffn"
                )
                for _ in range(num_experts)
            ])
        
        # Router
        router_config = router_config or {}
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            **router_config
        )
        
        # Expert metadata
        self.expert_metadata = [expert.metadata for expert in self.experts]
    
    def _dispatch_tokens(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Dispatch tokens to their assigned experts.
        
        Args:
            hidden_states: Input tokens [batch*seq_len, d_model]
            routing_weights: Routing weights [batch*seq_len, top_k]
            expert_indices: Expert indices [batch*seq_len, top_k]
            
        Returns:
            List of expert inputs
        """
        expert_inputs = [[] for _ in range(self.num_experts)]
        expert_weights = [[] for _ in range(self.num_experts)]
        token_indices = [[] for _ in range(self.num_experts)]
        
        # Dispatch tokens to experts
        for token_idx in range(hidden_states.shape[0]):
            for k in range(self.top_k):
                expert_idx = expert_indices[token_idx, k].item()
                weight = routing_weights[token_idx, k]
                
                expert_inputs[expert_idx].append(hidden_states[token_idx])
                expert_weights[expert_idx].append(weight)
                token_indices[expert_idx].append((token_idx, k))
        
        # Convert to tensors
        expert_batches = []
        for expert_idx in range(self.num_experts):
            if expert_inputs[expert_idx]:
                expert_batch = torch.stack(expert_inputs[expert_idx])
                expert_batch_weights = torch.stack(expert_weights[expert_idx])
                expert_batches.append((expert_batch, expert_batch_weights, token_indices[expert_idx]))
            else:
                expert_batches.append(None)
        
        return expert_batches
    
    def _combine_expert_outputs(
        self,
        expert_outputs: List[torch.Tensor],
        expert_batches: List[Tuple],
        hidden_states_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Combine outputs from all experts.
        
        Args:
            expert_outputs: List of expert outputs
            expert_batches: List of expert batch information
            hidden_states_shape: Original hidden states shape
            
        Returns:
            Combined output tensor
        """
        combined_output = torch.zeros(
            hidden_states_shape, dtype=expert_outputs[0].dtype, device=expert_outputs[0].device
        )
        
        for expert_idx, expert_output in enumerate(expert_outputs):
            if expert_output is not None and expert_batches[expert_idx] is not None:
                _, expert_weights, token_indices = expert_batches[expert_idx]
                
                for i, (token_idx, k) in enumerate(token_indices):
                    weight = expert_weights[i]
                    combined_output[token_idx] += weight * expert_output[i]
        
        return combined_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_auxiliary_loss: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            return_auxiliary_loss: Whether to return auxiliary losses
            
        Returns:
            Tuple of (output, auxiliary_losses)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        original_shape = hidden_states.shape
        
        # Flatten for expert processing
        hidden_states_flat = hidden_states.view(-1, d_model)
        
        # Route tokens to experts
        routing_weights, expert_indices, auxiliary_losses = self.router(
            hidden_states, attention_mask
        )
        
        # Dispatch tokens to experts
        expert_batches = self._dispatch_tokens(
            hidden_states_flat, routing_weights, expert_indices
        )
        
        # Process through experts
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            if expert_batches[expert_idx] is not None:
                expert_input, _, _ = expert_batches[expert_idx]
                expert_output = expert(expert_input)
                expert_outputs.append(expert_output)
            else:
                expert_outputs.append(None)
        
        # Combine expert outputs
        combined_output = self._combine_expert_outputs(
            expert_outputs, expert_batches, hidden_states_flat.shape
        )
        
        # Reshape back to original shape
        combined_output = combined_output.view(original_shape)
        
        if return_auxiliary_loss:
            return combined_output, auxiliary_losses
        else:
            return combined_output


class MoELayer(nn.Module):
    """
    Complete MoE layer with residual connection and layer norm.
    
    Drop-in replacement for standard feed-forward layers.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: int = None,
        expert_configs: Optional[List[Dict]] = None,
        router_config: Optional[Dict] = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_pre_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.use_pre_norm = use_pre_norm
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        if not use_pre_norm:
            self.post_layernorm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        
        # MoE
        self.moe = MixtureOfExperts(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            d_ff=d_ff,
            expert_configs=expert_configs,
            router_config=router_config,
            **kwargs
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_auxiliary_loss: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with residual connection.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask [batch, seq_len]
            return_auxiliary_loss: Whether to return auxiliary losses
            
        Returns:
            Tuple of (output, auxiliary_losses)
        """
        residual = hidden_states
        
        if self.use_pre_norm:
            hidden_states = self.input_layernorm(hidden_states)
        
        # MoE processing
        moe_output, auxiliary_losses = self.moe(
            hidden_states, attention_mask, return_auxiliary_loss
        )
        
        # Dropout and residual
        moe_output = self.dropout(moe_output)
        hidden_states = residual + moe_output
        
        if not self.use_pre_norm:
            hidden_states = self.post_layernorm(hidden_states)
        
        return hidden_states, auxiliary_losses


def create_language_specialized_experts(
    d_model: int,
    languages: List[str],
    d_ff: int = None,
    expert_type: str = "ffn"
) -> List[Dict]:
    """
    Create expert configurations specialized for different languages.
    
    Args:
        d_model: Model dimension
        languages: List of language codes (e.g., ["en", "es", "fr", "de"])
        d_ff: Feed-forward dimension
        expert_type: Type of expert ("ffn", "conv", "attention")
        
    Returns:
        List of expert configurations
    """
    if d_ff is None:
        d_ff = d_model * 4
    
    expert_configs = []
    for language in languages:
        config = {
            "d_ff": d_ff,
            "expert_type": expert_type,
            "specialization": f"language_{language}",
            "dropout": 0.1,
            "activation": "gelu"
        }
        expert_configs.append(config)
    
    return expert_configs


def create_moe_layer(
    d_model: int,
    num_experts: int = 8,
    top_k: int = 2,
    specialization_type: Optional[str] = None,
    languages: Optional[List[str]] = None,
    **kwargs
) -> MoELayer:
    """
    Factory function to create MoE layers.
    
    Args:
        d_model: Model dimension
        num_experts: Number of experts
        top_k: Number of experts to route to
        specialization_type: Type of specialization ("language", "voice", "style")
        languages: List of languages for language specialization
        **kwargs: Additional layer arguments
        
    Returns:
        MoE layer ready for use
    """
    expert_configs = None
    
    if specialization_type == "language" and languages is not None:
        # Ensure we have enough experts for languages
        if len(languages) > num_experts:
            raise ValueError(f"Number of languages ({len(languages)}) exceeds number of experts ({num_experts})")
        
        # Create language-specialized experts
        expert_configs = create_language_specialized_experts(
            d_model=d_model,
            languages=languages,
            expert_type="ffn"
        )
        
        # Add general experts if needed
        while len(expert_configs) < num_experts:
            expert_configs.append({
                "d_ff": d_model * 4,
                "expert_type": "ffn",
                "specialization": "general",
                "dropout": 0.1,
                "activation": "gelu"
            })
    
    return MoELayer(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        expert_configs=expert_configs,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test MoE implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    seq_len = 512
    d_model = 512
    num_experts = 8
    top_k = 2
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"🔀 Testing Mixture of Experts (MoE)")
    print(f"Input shape: {x.shape}")
    print(f"Number of experts: {num_experts}")
    print(f"Top-K routing: {top_k}")
    
    # Test basic MoE layer
    moe_layer = create_moe_layer(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k
    ).to(device)
    
    with torch.no_grad():
        output, aux_losses = moe_layer(x)
        print(f"MoE output shape: {output.shape}")
        if aux_losses:
            print(f"Auxiliary losses: {aux_losses}")
    
    # Test language-specialized MoE
    languages = ["en", "es", "fr", "de"]
    lang_moe = create_moe_layer(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        specialization_type="language",
        languages=languages
    ).to(device)
    
    with torch.no_grad():
        output, aux_losses = lang_moe(x)
        print(f"Language-specialized MoE output shape: {output.shape}")
        print(f"Expert specializations: {[config['specialization'] for config in lang_moe.moe.expert_metadata]}")
    
    print(f"✅ Mixture of Experts test completed successfully!")
