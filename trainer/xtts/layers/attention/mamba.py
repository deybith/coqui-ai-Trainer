"""
Enhanced XTTS Mamba/State Space Models Implementation

This module implements Mamba state space models for XTTS, providing
linear complexity attention alternatives with superior long-range modeling.

Key features:
- Linear O(n) complexity vs O(n²) for transformers
- Selective state space mechanism for audio processing
- Hardware-efficient implementation
- Integration with existing XTTS architecture

Based on:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Selective State Space Models for Long-Range Dependencies
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from einops import rearrange, repeat
import numpy as np

try:
    # Try to import mamba_ssm if available
    from mamba_ssm import Mamba
    from mamba_ssm.modules.mamba_simple import MambaInnerFn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️  Mamba SSM not available. Using custom implementation.")


class SelectiveStateSpace(nn.Module):
    """
    Core selective state space computation.
    
    Implements the selective mechanism that allows the model to
    focus on relevant parts of the input sequence.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        # Activation
        self.activation = "silu"
        self.act = nn.SiLU()
        
        # SSM parameters
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # Initialize dt projection bias
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
            
        # Initialize dt bias to be between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
    
    def forward(self, hidden_states: torch.Tensor, inference_params=None):
        """
        Forward pass of the selective state space model.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            inference_params: Optional inference parameters for optimization
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seqlen, dim = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # (batch, seqlen, d_inner)
        
        # Convolution for local dependencies
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seqlen]
        x = rearrange(x, "b d l -> b l d")
        
        # Activation
        x = self.act(x)
        
        # SSM computation
        y = self.ssm(x, inference_params)
        
        # Gate
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor, inference_params=None):
        """
        Selective state space computation.
        """
        batch, seqlen, d_inner = x.shape
        
        # Compute dt, B, C from input
        x_dbl = self.x_proj(x)  # (batch, seqlen, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project dt
        dt = self.dt_proj(dt)  # (batch, seqlen, d_inner)
        
        # Get A and D
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        D = self.D.float()
        
        # Discretize continuous parameters (A, B)
        # Use Euler method for discretization
        # deltaA = exp(dt * A)
        # deltaB = dt * B
        dt = dt.transpose(-1, -2)  # (batch, d_inner, seqlen)
        A = A.unsqueeze(0).expand(batch, -1, -1)  # (batch, d_inner, d_state)
        
        # Discretize A
        dtA = torch.einsum("bdn,bds->bdns", dt, A)  # (batch, d_inner, seqlen, d_state)
        deltaA = torch.exp(dtA)
        
        # Discretize B
        B = B.transpose(-1, -2)  # (batch, d_state, seqlen)
        deltaB = torch.einsum("bdn,bsn->bdsn", dt, B)  # (batch, d_inner, d_state, seqlen)
        
        # SSM recurrence
        x = x.transpose(-1, -2)  # (batch, d_inner, seqlen)
        
        # Initialize state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for i in range(seqlen):
            # Update state: h = deltaA[:, :, i] * h + deltaB[:, :, :, i] * x[:, :, i:i+1]
            h = deltaA[:, :, i] * h + deltaB[:, :, :, i] * x[:, :, i:i+1].unsqueeze(-1)
            
            # Compute output: y = C @ h + D * x
            C_i = C[:, i:i+1, :].transpose(-1, -2)  # (batch, d_state, 1)
            y_i = torch.einsum("bds,bds->bd", h.squeeze(-1), C_i.squeeze(-1))
            y_i = y_i + D * x[:, :, i]
            outputs.append(y_i.unsqueeze(-1))
        
        y = torch.cat(outputs, dim=-1)  # (batch, d_inner, seqlen)
        y = y.transpose(-1, -2)  # (batch, seqlen, d_inner)
        
        return y


class MambaBlock(nn.Module):
    """
    Complete Mamba block with residual connection and layer norm.
    
    This is the main building block that can replace transformer layers.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_range: float = 0.02,
        residual_in_fp32: bool = False,
        fused_add_norm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Layer norm
        if rms_norm:
            self.norm = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
        else:
            self.norm = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)
        
        # Mamba layer
        self.mixer = SelectiveStateSpace(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            **factory_kwargs,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
    ):
        """
        Forward pass with residual connection.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            residual: Optional residual tensor
            inference_params: Optional inference parameters
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + hidden_states
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # TODO: Implement fused add norm for better performance
            raise NotImplementedError("Fused add norm not implemented yet")
        
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
        return hidden_states, residual


class HybridMambaAttention(nn.Module):
    """
    Hybrid Mamba-Attention module that combines the benefits of both.
    
    Uses Mamba for long-range dependencies and attention for local interactions.
    """
    
    def __init__(
        self,
        d_model: int,
        mamba_config: dict = None,
        attention_config: dict = None,
        mix_ratio: float = 0.5,
        use_gating: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.mix_ratio = mix_ratio
        self.use_gating = use_gating
        
        # Mamba block
        mamba_config = mamba_config or {}
        self.mamba = MambaBlock(d_model=d_model, **mamba_config)
        
        # Attention block (simplified)
        attention_config = attention_config or {}
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_config.get('num_heads', 8),
            dropout=attention_config.get('dropout', 0.1),
            batch_first=True
        )
        
        # Gating mechanism
        if use_gating:
            self.gate = nn.Linear(d_model, 2)
            self.gate_activation = nn.Sigmoid()
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass combining Mamba and attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        residual = x
        
        # Mamba path
        mamba_out, _ = self.mamba(x)
        
        # Attention path
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        if self.use_gating:
            # Learnable gating mechanism
            gate_weights = self.gate_activation(self.gate(x))  # [batch, seq_len, 2]
            mamba_gate = gate_weights[..., 0:1]  # [batch, seq_len, 1]
            attn_gate = gate_weights[..., 1:2]   # [batch, seq_len, 1]
            
            combined = mamba_gate * mamba_out + attn_gate * attn_out
        else:
            # Fixed mixing ratio
            combined = self.mix_ratio * mamba_out + (1 - self.mix_ratio) * attn_out
        
        # Output projection and residual
        output = self.output_proj(combined)
        output = self.norm(output + residual)
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Often used in Mamba models for better performance.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MambaEncoder(nn.Module):
    """
    Multi-layer Mamba encoder for XTTS integration.
    
    Can be used as a drop-in replacement for transformer encoders.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_config: dict = None,
        use_hybrid: bool = False,
        hybrid_config: dict = None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_hybrid = use_hybrid
        
        layer_config = layer_config or {}
        
        if use_hybrid:
            hybrid_config = hybrid_config or {}
            self.layers = nn.ModuleList([
                HybridMambaAttention(
                    d_model=d_model,
                    mamba_config={
                        'd_state': d_state,
                        'd_conv': d_conv,
                        'expand': expand,
                        **layer_config
                    },
                    **hybrid_config
                )
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    layer_idx=i,
                    **layer_config
                )
                for i in range(num_layers)
            ])
        
        self.norm_f = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference_params=None
    ) -> torch.Tensor:
        """
        Forward pass through multiple Mamba layers.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask (used only in hybrid mode)
            inference_params: Optional inference parameters
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        hidden_states = x
        residual = None
        
        for layer in self.layers:
            if self.use_hybrid:
                hidden_states = layer(hidden_states, mask=mask)
            else:
                hidden_states, residual = layer(
                    hidden_states, 
                    residual=residual,
                    inference_params=inference_params
                )
        
        if residual is not None:
            hidden_states = residual + hidden_states
        
        hidden_states = self.norm_f(hidden_states)
        
        return hidden_states


def create_mamba_layer(
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    layer_type: str = "mamba",  # "mamba", "hybrid"
    **kwargs
) -> nn.Module:
    """
    Factory function to create Mamba layers.
    
    Args:
        d_model: Model dimension
        d_state: State dimension for SSM
        d_conv: Convolution kernel size
        expand: Expansion factor
        layer_type: Type of layer ("mamba" or "hybrid")
        **kwargs: Additional layer-specific arguments
        
    Returns:
        Mamba layer ready for use
    """
    if layer_type == "mamba":
        return MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )
    elif layer_type == "hybrid":
        return HybridMambaAttention(
            d_model=d_model,
            mamba_config={
                'd_state': d_state,
                'd_conv': d_conv,
                'expand': expand
            },
            **kwargs
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def create_mamba_encoder(
    num_layers: int,
    d_model: int,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    use_hybrid: bool = False,
    **kwargs
) -> MambaEncoder:
    """
    Factory function to create Mamba encoder.
    
    Args:
        num_layers: Number of layers
        d_model: Model dimension
        d_state: State dimension for SSM
        d_conv: Convolution kernel size
        expand: Expansion factor
        use_hybrid: Whether to use hybrid Mamba-Attention layers
        **kwargs: Additional encoder arguments
        
    Returns:
        Mamba encoder ready for use
    """
    return MambaEncoder(
        num_layers=num_layers,
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        use_hybrid=use_hybrid,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test Mamba implementation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 2
    seq_len = 1024
    d_model = 512
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"🐍 Testing Mamba State Space Models")
    print(f"Input shape: {x.shape}")
    print(f"Mamba SSM available: {MAMBA_AVAILABLE}")
    
    # Test Mamba block
    mamba_block = create_mamba_layer(d_model=d_model).to(device)
    
    with torch.no_grad():
        output, _ = mamba_block(x)
        print(f"Mamba block output shape: {output.shape}")
    
    # Test Hybrid Mamba-Attention
    hybrid_layer = create_mamba_layer(
        d_model=d_model,
        layer_type="hybrid"
    ).to(device)
    
    with torch.no_grad():
        output = hybrid_layer(x)
        print(f"Hybrid layer output shape: {output.shape}")
    
    # Test Mamba encoder
    mamba_encoder = create_mamba_encoder(
        num_layers=4,
        d_model=d_model,
        use_hybrid=False
    ).to(device)
    
    with torch.no_grad():
        output = mamba_encoder(x)
        print(f"Mamba encoder output shape: {output.shape}")
        print(f"✅ Mamba State Space Models test completed successfully!")
