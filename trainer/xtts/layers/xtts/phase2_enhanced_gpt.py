"""
Phase 2 Enhanced XTTS GPT Model

This module integrates all Phase 2 enhancements (Mamba, RoPE, MoE, Flash Attention) 
with the XTTS GPT architecture to create the most advanced TTS model.

Phase 2 Components:
- Mamba/State Space Models for linear complexity
- Rotary Position Embedding (RoPE) for better positional encoding
- Mixture of Experts (MoE) for scalable specialization  
- Flash Attention 2.0 for memory-efficient attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import random
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

# Import Phase 2 components
from trainer.xtts.layers.attention.phase2_integration import (
    Phase2Config, 
    Phase2EnhancedLayer,
    Phase2EnhancedEncoder,
    create_phase2_encoder
)


def null_position_embeddings(range, dim):
    """Null position embeddings for compatibility"""
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

# Import original XTTS components for compatibility
from TTS.tts.layers.tortoise.autoregressive import (
    ConditioningEncoder,
    LearnedPositionEmbeddings as _OriginalLearnedPositionEmbeddings,
    _prepare_attention_mask_for_generation,
)
from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
from TTS.tts.layers.xtts.perceiver_encoder import PerceiverResampler
from transformers import GPT2Config


class LearnedPositionEmbeddings(_OriginalLearnedPositionEmbeddings):
    """Fixed version of LearnedPositionEmbeddings that handles batch dimensions correctly"""
    
    def forward(self, x):
        """
        Fixed forward method that returns proper batch dimension.
        
        Args:
            x: Input tensor [batch, seq_len, ...] 
            
        Returns:
            Position embeddings [batch, seq_len, model_dim]
        """
        batch_size = x.shape[0]
        sl = x.shape[1]
        
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            pos_emb = self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            pos_emb = self.emb(torch.arange(0, sl, device=x.device))
        
        # Expand for batch dimension: [seq_len, model_dim] -> [batch, seq_len, model_dim]
        return pos_emb.unsqueeze(0).expand(batch_size, -1, -1)


@dataclass
class Phase2GPTConfig:
    """Configuration for Phase 2 Enhanced GPT"""
    
    # Basic GPT Configuration
    start_text_token: int = 261
    stop_text_token: int = 0
    layers: int = 8
    d_model: int = 512  # Changed from model_dim to align with Phase2Config
    heads: int = 8
    max_text_tokens: int = 120
    max_mel_tokens: int = 250
    max_prompt_tokens: int = 70
    max_conditioning_inputs: int = 1
    code_stride_len: int = 1024
    number_text_tokens: int = 256
    num_audio_tokens: int = 8194
    start_audio_token: int = 8192
    stop_audio_token: int = 8193
    
    # Training Configuration
    train_solo_embeddings: bool = False
    checkpointing: bool = False
    average_conditioning_embeddings: bool = False
    label_smoothing: float = 0.0
    use_perceiver_resampler: bool = False
    perceiver_cond_length_compression: int = 256
    
    # Phase 2 Configuration
    use_phase2_enhancements: bool = True
    phase2_config: Optional[Phase2Config] = None
    
    def __post_init__(self):
        if self.use_phase2_enhancements and self.phase2_config is None:
            self.phase2_config = Phase2Config(
                d_model=self.d_model,  # Fixed parameter name
                num_heads=self.heads,
                # Enable all Phase 2 techniques
                use_mamba=True,
                use_flash_attention=True,
                use_rope=True,
                use_moe=True,  # Fixed parameter name
                # Mamba configuration
                mamba_d_state=16,
                mamba_d_conv=4,
                mamba_expand=2,
                # MoE configuration
                moe_num_experts=8,
                moe_top_k=2,
                # RoPE configuration
                rope_max_position_embeddings=8192,
                rope_type="linear"
            )


class Phase2EnhancedTransformerBlock(nn.Module):
    """
    A single transformer block enhanced with Phase 2 techniques
    """
    
    def __init__(self, config: Phase2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Create Phase 2 enhanced layer
        self.phase2_layer = Phase2EnhancedLayer(config, layer_idx)
        
        # Layer norm for pre-norm architecture
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model, eps=1e-5)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, ...]]]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply Phase 2 enhancements
        phase2_output = self.phase2_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Extract hidden states from Phase 2 output
        if isinstance(phase2_output, tuple):
            hidden_states = phase2_output[0]
            attention_weights = phase2_output[1] if len(phase2_output) > 1 else None
        else:
            hidden_states = phase2_output
            attention_weights = None
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Post attention layer norm (if using post-norm)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs


class Phase2EnhancedGPTModel(nn.Module):
    """
    Phase 2 Enhanced GPT Model that replaces the HuggingFace GPT2Model
    with state-of-the-art techniques while maintaining compatibility.
    """
    
    def __init__(self, config: Phase2GPTConfig):
        super().__init__()
        self.config = config
        
        # Create transformer blocks with Phase 2 enhancements
        self.h = nn.ModuleList([
            Phase2EnhancedTransformerBlock(config.phase2_config, i) 
            for i in range(config.layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5)
        
        # Dropout
        self.drop = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, Dict[str, Any]]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        if inputs_embeds is None:
            raise ValueError("inputs_embeds cannot be None for Phase2EnhancedGPTModel")
        
        hidden_states = inputs_embeds
        
        # Apply dropout
        hidden_states = self.drop(hidden_states)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = outputs[0]
            
            if use_cache:
                presents = presents + (outputs[-1],)
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1 if use_cache else -1],)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if return_dict:
            return {
                'last_hidden_state': hidden_states,
                'past_key_values': presents,
                'hidden_states': all_hidden_states,
                'attentions': all_self_attentions,
            }
        else:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)


def build_phase2_enhanced_gpt_transformer(
    layers: int,
    model_dim: int,
    heads: int,
    max_mel_seq_len: int,
    max_text_seq_len: int,
    checkpointing: bool,
    max_prompt_len: int = 0,
    use_phase2_enhancements: bool = False,
    phase2_config: Optional[Phase2Config] = None,
    **phase2_kwargs
):
    """
    Enhanced version of build_hf_gpt_transformer with optional Phase 2 enhancements.
    
    This is a drop-in replacement for the original function that can optionally
    use Phase 2 enhanced models while maintaining full backward compatibility.
    
    Args:
        layers: Number of transformer layers
        model_dim: Model dimension
        heads: Number of attention heads  
        max_mel_seq_len: Maximum mel sequence length
        max_text_seq_len: Maximum text sequence length
        checkpointing: Whether to use gradient checkpointing
        max_prompt_len: Maximum prompt length
        use_phase2_enhancements: Whether to use Phase 2 enhancements
        phase2_config: Optional Phase2Config for detailed control
        **phase2_kwargs: Additional Phase 2 configuration parameters
        
    Returns:
        Tuple of (gpt_model, mel_pos_emb, text_pos_emb, conditioning_encoder, perceiver)
        Same interface as original build_hf_gpt_transformer function
    """
    if use_phase2_enhancements:
        # Use Phase 2 Enhanced model
        gpt_config = Phase2GPTConfig(
            layers=layers,
            d_model=model_dim,
            heads=heads,
            max_text_tokens=max_text_seq_len,
            max_mel_tokens=max_mel_seq_len,
            max_prompt_tokens=max_prompt_len,
            use_phase2_enhancements=True,
            checkpointing=checkpointing,
            phase2_config=phase2_config,
            **phase2_kwargs
        )
        
        # Create Phase 2 enhanced model
        gpt = Phase2EnhancedGPTModel(gpt_config)
        
        # Use our fixed position embeddings
        mel_pos_emb = (
            LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
            if max_mel_seq_len != -1
            else functools.partial(null_position_embeddings, dim=model_dim)
        )
        text_pos_emb = (
            LearnedPositionEmbeddings(max_text_seq_len, model_dim)
            if max_mel_seq_len != -1
            else functools.partial(null_position_embeddings, dim=model_dim)
        )
        
        return gpt, mel_pos_emb, text_pos_emb, None, None
    
    else:
        # Fall back to original implementation
        from trainer.xtts.layers.tortoise.autoregressive import build_hf_gpt_transformer as original_build_hf_gpt_transformer
        return original_build_hf_gpt_transformer(
            layers=layers,
            model_dim=model_dim,
            heads=heads,
            max_mel_seq_len=max_mel_seq_len,
            max_text_seq_len=max_text_seq_len,
            checkpointing=checkpointing,
            max_prompt_len=max_prompt_len
        )


class Phase2EnhancedGPT(nn.Module):
    """
    Phase 2 Enhanced XTTS GPT Model
    
    This is a drop-in replacement for the original XTTS GPT class that incorporates
    all Phase 2 enhancements while maintaining full backward compatibility.
    """
    
    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        layers=8,
        d_model=512,  # Changed from model_dim to align with Phase2Config
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=256,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        train_solo_embeddings=False,
        checkpointing=False,
        average_conditioning_embeddings=False,
        label_smoothing=0.0,
        use_perceiver_resampler=False,
        perceiver_cond_length_compression=256,
        # Phase 2 specific parameters
        use_phase2_enhancements=True,
        phase2_config=None,
    ):
        super().__init__()
        
        # Store configuration
        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token
        self.layers = layers
        self.heads = heads
        self.d_model = d_model  # Changed from model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_gen_mel_tokens = max_mel_tokens - self.max_conditioning_inputs - 2
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.code_stride_len = code_stride_len
        self.average_conditioning_embeddings = average_conditioning_embeddings
        self.use_perceiver_resampler = use_perceiver_resampler
        self.perceiver_cond_length_compression = perceiver_cond_length_compression
        self.use_phase2_enhancements = use_phase2_enhancements
        
        # Conditioning encoder
        self.conditioning_encoder = ConditioningEncoder(80, d_model, num_attn_heads=heads)
        self.conditioning_dropout = nn.Dropout1d(0.1)
        
        # Text and mel embeddings
        self.text_embedding = nn.Embedding(self.number_text_tokens, d_model)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, d_model)
        
        # Build the transformer (Phase 2 enhanced or original)
        if use_phase2_enhancements:
            (
                self.gpt,
                self.mel_pos_embedding,
                self.text_pos_embedding,
                self.mel_layer_pos_embedding,
                self.text_layer_pos_embedding,
            ) = build_phase2_enhanced_gpt_transformer(
                layers=layers,
                model_dim=d_model,  # Fixed parameter name
                heads=heads,
                max_mel_seq_len=self.max_mel_tokens,
                max_text_seq_len=self.max_text_tokens,
                max_prompt_len=self.max_prompt_tokens,
                checkpointing=checkpointing,
                use_phase2_enhancements=True,
                phase2_config=phase2_config,
            )
        else:
            # Fallback to original implementation
            from TTS.tts.layers.tortoise.autoregressive import build_hf_gpt_transformer
            (
                self.gpt,
                self.mel_pos_embedding,
                self.text_pos_embedding,
                self.mel_layer_pos_embedding,
                self.text_layer_pos_embedding,
            ) = build_hf_gpt_transformer(
                layers=layers,
                model_dim=d_model,  # Fixed parameter name
                heads=heads,
                max_mel_seq_len=self.max_mel_tokens,
                max_text_seq_len=self.max_text_tokens,
                max_prompt_len=self.max_prompt_tokens,
                checkpointing=checkpointing,
            )
        
        # Solo embeddings for training
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0
        
        # Perceiver resampler for XTTS v2
        if self.use_perceiver_resampler:
            self.conditioning_perceiver = PerceiverResampler(
                dim=d_model,  # Fixed parameter name
                depth=2,
                dim_context=d_model,  # Fixed parameter name
                num_latents=32,
                dim_head=64,
                heads=8,
                ff_mult=4,
                use_flash_attn=False,  # Will be handled by our Phase 2 components
            )
        else:
            # XTTS v1 components
            self.prompt_embedding = nn.Embedding(self.num_audio_tokens, d_model)  # Fixed parameter name
            self.prompt_pos_embedding = LearnedPositionEmbeddings(24 * 9, d_model)  # Fixed parameter name
        
        # Output layers
        self.final_norm = nn.LayerNorm(d_model)  # Fixed parameter name
        self.text_head = nn.Linear(d_model, self.number_text_tokens)  # Fixed parameter name
        self.mel_head = nn.Linear(d_model, self.num_audio_tokens)  # Fixed parameter name
    
    def get_grad_norm_parameter_groups(self):
        """Get parameter groups for gradient norm monitoring"""
        groups = {
            "conditioning_encoder": list(self.conditioning_encoder.parameters()),
            "gpt": list(self.gpt.parameters()),
            "heads": list(self.text_head.parameters()) + list(self.mel_head.parameters()),
        }
        
        if self.use_perceiver_resampler:
            groups["conditioning_perceiver"] = list(self.conditioning_perceiver.parameters())
        
        return groups
    
    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False):
        """Initialize GPT for inference mode"""
        seq_length = self.max_prompt_tokens + self.max_mel_tokens + self.max_text_tokens + 1
        
        # Create GPT2 config for compatibility with inference model
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.d_model,  # Fixed parameter name
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        
        self.gpt_inference = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        
        # Set word token embedding (required by HF interface)
        if not hasattr(self.gpt, 'wte'):
            self.gpt.wte = self.mel_embedding
        
        if use_deepspeed:
            import deepspeed
            self.ds_engine = deepspeed.init_inference(
                model=self.gpt_inference.half(),
                mp_size=1,
                dtype=torch.float32,
                replace_method="auto",
                replace_with_kernel_inject=True,
            )
            self.gpt_inference = self.ds_engine.module.eval()
    
    def set_inputs_and_targets(self, input, start_token, stop_token):
        """Set input and target tensors with start/stop tokens"""
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar
    
    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """Set padding areas within MEL tokens"""
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_audio_token
        return mel_input_tokens
    
    def get_logits(
        self,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        prompt=None,
        get_attns=False,
        return_latent=False,
        attn_mask_cond=None,
        attn_mask_text=None,
        attn_mask_mel=None,
    ):
        """Get logits from the model"""
        if prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)
        else:
            offset = 0
            if second_inputs is not None:
                emb = torch.cat([first_inputs, second_inputs], dim=1)
            else:
                emb = first_inputs
        
        # Create attention mask
        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_cond = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
                attn_mask = torch.cat([attn_mask_cond, attn_mask], dim=1)
        
        # Forward through the model
        gpt_out = self.gpt(
            inputs_embeds=emb,
            attention_mask=attn_mask,
            output_attentions=get_attns,
            return_dict=True,
        )
        
        if get_attns:
            return gpt_out.get('attentions', gpt_out.get('attention_scores'))
        
        enc = gpt_out['last_hidden_state'][:, offset:] if isinstance(gpt_out, dict) else gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)
        
        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]
        
        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits
    
    def get_prompts(self, prompt_codes):
        """Create a prompt from the mel codes"""
        prompt = prompt_codes
        if self.training:
            lengths = []
            # Compute the real prompt length based on the first encounter with padding token
            for i in range(prompt_codes.shape[0]):
                length = 0
                for j in range(prompt_codes.shape[1]):
                    if prompt_codes[i, j] == 83:  # padding token
                        break
                    else:
                        length += 1
                lengths.append(length)
            
            prompt_len = 3 * 24  # 3 seconds in frames
            if prompt_codes.shape[-1] >= prompt_len:
                for i in range(prompt_codes.shape[0]):
                    if lengths[i] < prompt_len:
                        start = 0
                    else:
                        start = torch.randint(0, lengths[i] - prompt_len + 1, (1,)).item()
                    prompt = prompt_codes[:, start:start + prompt_len]
        
        # Add start and stop tokens
        prompt = F.pad(prompt, (1, 0), value=self.start_prompt_token)
        prompt = F.pad(prompt, (0, 1), value=self.stop_prompt_token)
        return prompt
    
    def get_style_emb(self, cond_input, return_latent=False):
        """Get style embeddings from conditioning input"""
        conds = None
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)
            if self.use_perceiver_resampler:
                conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)
        else:
            conds = cond_input.unsqueeze(1)
        return conds
    
    def forward(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        cond_latents=None,
        return_attentions=False,
        return_latent=False,
    ):
        """Forward pass of the Phase 2 Enhanced GPT model"""
        
        # Input validation
        if self.max_conditioning_inputs == 0:
            assert cond_mels is None, "cond_mels is not None, but max_conditioning_inputs == 0"
        
        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3
        
        # Handle conditioning length adjustments
        if cond_lens is not None:
            if self.use_perceiver_resampler:
                cond_lens = cond_lens // self.perceiver_cond_length_compression
            else:
                cond_lens = cond_lens // self.code_stride_len
        
        if cond_idxs is not None:
            for idx in range(cond_idxs.size(0)):
                if self.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[idx] // self.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.code_stride_len
        
        # Prepare inputs
        max_mel_len = code_lengths.max()
        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))
        
        # Validation assertions
        assert max_mel_len <= audio_codes.shape[-1], f"max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert max_text_len <= text_inputs.shape[-1], f"max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"
        
        # Append stop tokens
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_audio_token)
        
        # Set padding
        audio_codes = self.set_mel_padding(audio_codes, code_lengths - 3)
        
        # Build input and target tensors
        text_inputs, text_targets = self.set_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        audio_codes, mel_targets = self.set_inputs_and_targets(audio_codes, self.start_audio_token, self.stop_audio_token)
        
        # Create attention masks
        attn_mask_cond = None
        attn_mask_text = None
        attn_mask_mel = None
        
        if not return_latent:
            attn_mask_cond = torch.ones(cond_mels.shape[0], cond_mels.shape[-1], dtype=torch.bool, device=text_inputs.device)
            attn_mask_text = torch.ones(text_inputs.shape[0], text_inputs.shape[1], dtype=torch.bool, device=text_inputs.device)
            attn_mask_mel = torch.ones(audio_codes.shape[0], audio_codes.shape[1], dtype=torch.bool, device=audio_codes.device)
            
            # Apply masking based on actual lengths
            if cond_idxs is not None:
                for idx, r in enumerate(cond_idxs):
                    l = r[1] - r[0]
                    attn_mask_cond[idx, l:] = 0.0
            elif cond_lens is not None:
                for idx, l in enumerate(cond_lens):
                    attn_mask_cond[idx, l:] = 0.0
            
            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1:] = 0.0
            
            for idx, l in enumerate(code_lengths):
                attn_mask_mel[idx, l + 1:] = 0.0
        
        # Compute embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(audio_codes)
        
        # Get conditioning latents
        if cond_latents is None:
            cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)
        
        # Get logits using Phase 2 enhanced model
        sub = -5 if not self.training else -1
        
        text_logits, mel_logits = self.get_logits(
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            prompt=cond_latents,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_cond=attn_mask_cond,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )
        
        if return_latent:
            return mel_logits[:, :sub]
        
        if return_attentions:
            return mel_logits
        
        # Set paddings to -1 to ignore them in loss
        for idx, l in enumerate(text_lengths):
            text_targets[idx, l + 1:] = -1
        
        for idx, l in enumerate(code_lengths):
            mel_targets[idx, l + 1:] = -1
        
        # Validate stop tokens
        assert (mel_targets == self.stop_audio_token).sum() >= mel_targets.shape[0], \
            f"mel_targets does not contain stop token ({self.stop_audio_token}) in every row."
        
        # Ignore loss for conditioning segment
        if cond_idxs is not None:
            for idx in range(cond_idxs.size(0)):
                cond_start = cond_idxs[idx, 0]
                cond_end = cond_idxs[idx, 1]
                mel_targets[idx, cond_start:cond_end] = -1
        
        # Compute losses
        loss_text = F.cross_entropy(text_logits, text_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing)
        loss_mel = F.cross_entropy(mel_logits, mel_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing)
        
        return loss_text.mean(), loss_mel.mean(), mel_logits
    
    # Inference methods (same as original)
    def inference(self, cond_latents, text_inputs, **hf_generate_kwargs):
        """Inference method"""
        self.compute_embeddings(cond_latents, text_inputs)
        return self.generate(cond_latents, text_inputs, **hf_generate_kwargs)
    
    def compute_embeddings(self, cond_latents, text_inputs):
        """Compute embeddings for inference"""
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (emb.shape[0], emb.shape[1] + 1),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs
    
    def generate(self, cond_latents, text_inputs, **hf_generate_kwargs):
        """Generate method for inference"""
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        stop_token_tensor = torch.tensor(self.stop_audio_token, device=gpt_inputs.device, dtype=torch.long)
        attention_mask = _prepare_attention_mask_for_generation(gpt_inputs, stop_token_tensor, stop_token_tensor)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
            attention_mask=attention_mask,
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1]:], gen
        return gen[:, gpt_inputs.shape[1]:]
    
    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        """Get generator for streaming inference"""
        return self.gpt_inference.generate_stream(
            fake_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + fake_inputs.shape[-1],
            do_stream=True,
            **hf_generate_kwargs,
        )


# Factory function for easy integration
def create_phase2_enhanced_gpt(**kwargs) -> Phase2EnhancedGPT:
    """
    Factory function to create a Phase 2 Enhanced GPT model.
    
    Args:
        **kwargs: Arguments passed to Phase2EnhancedGPT constructor
        
    Returns:
        Phase2EnhancedGPT: The enhanced model instance
    """
    return Phase2EnhancedGPT(**kwargs)


# Validation function
def validate_phase2_integration():
    """
    Validate that Phase 2 integration works correctly
    """
    print("🚀 Validating Phase 2 Enhanced XTTS GPT Integration...")
    
    try:
        # Create a test configuration
        config = Phase2GPTConfig(
            layers=4,  # Smaller for testing
            d_model=256,  # Fixed parameter name
            heads=4,
            use_phase2_enhancements=True
        )
        
        # Create the model
        model = Phase2EnhancedGPT(
            layers=config.layers,
            d_model=config.d_model,  # Fixed parameter name
            heads=config.heads,
            use_phase2_enhancements=True,
            number_text_tokens=512,  # Increase vocabulary size to accommodate start token (261)
            num_audio_tokens=8500,   # Increase vocabulary size to accommodate start/stop tokens
        )
        
        print("✅ Phase 2 Enhanced GPT model created successfully")
        print(f"   - Model dimension: {config.d_model}")  # Fixed parameter name
        print(f"   - Number of layers: {config.layers}")
        print(f"   - Number of heads: {config.heads}")
        print(f"   - Phase 2 enhancements: {model.use_phase2_enhancements}")
        
        # Validate configuration alignment
        assert config.phase2_config is not None, "Phase2Config should be auto-created"
        assert config.phase2_config.d_model == config.d_model, "d_model should match between configs"
        assert config.phase2_config.num_heads == config.heads, "num_heads should match"
        
        print("✅ Configuration parameter alignment verified")
        print(f"   - Phase2GPTConfig.d_model: {config.d_model}")
        print(f"   - Phase2Config.d_model: {config.phase2_config.d_model}")
        print(f"   - Parameters are aligned: {config.d_model == config.phase2_config.d_model}")
        
        # Test parameter groups
        param_groups = model.get_grad_norm_parameter_groups()
        print(f"✅ Parameter groups: {list(param_groups.keys())}")
        
        # Test that we can create embeddings (basic functionality test)
        text_inputs = torch.randint(0, 200, (2, 10))
        text_emb = model.text_embedding(text_inputs)
        print(f"✅ Basic embedding test passed - shape: {text_emb.shape}")
        
        print("🎉 Phase 2 Enhanced XTTS GPT configuration validation completed successfully!")
        print("📝 Note: Full forward pass testing requires Mamba dimension adjustments for smaller test models")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    validate_phase2_integration()
