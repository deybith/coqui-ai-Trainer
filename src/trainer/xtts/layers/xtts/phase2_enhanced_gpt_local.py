"""
Phase 2 Enhanced XTTS GPT Model - Local Imports Version

This is a version of the Phase 2 Enhanced GPT that uses local imports
instead of external TTS dependencies to avoid import hanging issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from transformers import GPT2Config

# Import Phase 2 components
from .xtts.layers.attention.phase2_integration import (
    Phase2Config, 
    Phase2EnhancedLayer,
    Phase2EnhancedEncoder,
    create_phase2_encoder
)

# Use local imports instead of TTS imports
from .local_imports import (
    ConditioningEncoder,
    LearnedPositionEmbeddings,
    _prepare_attention_mask_for_generation,
    GPT2InferenceModel,
    PerceiverResampler,
)


def null_position_embeddings(range, dim):
    """Null position embeddings for compatibility"""
    return torch.zeros((range, dim), dtype=torch.float)


@dataclass
class Phase2GPTConfig:
    """Configuration for Phase 2 Enhanced GPT"""
    layers: int = 8
    d_model: int = 512  
    heads: int = 8
    max_text_tokens: int = 120
    max_mel_tokens: int = 250
    max_prompt_tokens: int = 70
    max_conditioning_inputs: int = 1
    code_stride_len: int = 1024
    number_text_tokens: int = 512
    num_audio_tokens: int = 8194
    start_audio_token: int = 8192
    stop_audio_token: int = 8193
    start_text_token: int = 261
    stop_text_token: int = 0
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
                d_model=self.d_model,
                num_layers=self.layers,
                num_heads=self.heads,
                use_mamba=True,
                use_flash_attention=True,
                use_rope=True,
                use_moe=True,
            )


class Phase2EnhancedTransformerBlock(nn.Module):
    """A single transformer block enhanced with Phase 2 techniques"""
    
    def __init__(self, config: Phase2Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Use Phase 2 enhanced layer
        self.enhanced_layer = Phase2EnhancedLayer(
            config=config,
            layer_idx=layer_idx,
            is_decoder=True,
        )
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs
    ):
        return self.enhanced_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )


class Phase2EnhancedGPTModel(nn.Module):
    """Phase 2 Enhanced GPT Model Core"""
    
    def __init__(self, config: Phase2GPTConfig):
        super().__init__()
        self.config = config
        
        if config.use_phase2_enhancements:
            # Use Phase 2 enhanced layers
            self.h = nn.ModuleList([
                Phase2EnhancedTransformerBlock(config.phase2_config, i)
                for i in range(config.layers)
            ])
        else:
            # Fall back to standard transformer blocks
            from transformers import GPT2Block, GPT2Config
            gpt_config = GPT2Config(
                n_embd=config.d_model,
                n_layer=config.layers,
                n_head=config.heads,
            )
            self.h = nn.ModuleList([
                GPT2Block(gpt_config, layer_idx=i)
                for i in range(config.layers)
            ])
            
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5)
        self.drop = nn.Dropout(0.1)
    
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
    """Build Phase 2 enhanced GPT transformer"""
    
    if use_phase2_enhancements:
        # Create Phase 2 Enhanced GPT config
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
        
        # Use local position embeddings
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
        # Fall back to basic transformer
        from transformers import GPT2Model, GPT2Config
        gpt_config = GPT2Config(
            vocab_size=8194,
            n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len,
            n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
            n_embd=model_dim,
            n_layer=layers,
            n_head=heads,
            gradient_checkpointing=checkpointing,
            use_cache=True,
        )
        gpt = GPT2Model(gpt_config)
        
        mel_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        
        return gpt, mel_pos_emb, text_pos_emb, None, None


class Phase2EnhancedGPT(nn.Module):
    """
    Phase 2 Enhanced XTTS GPT Model - Local Imports Version
    
    This is a version that uses local imports to avoid TTS dependency issues.
    """
    
    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        layers=8,
        d_model=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=512,
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
        self.layers = layers
        self.heads = heads
        self.d_model = d_model
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
                model_dim=d_model,
                heads=heads,
                max_mel_seq_len=self.max_mel_tokens,
                max_text_seq_len=self.max_text_tokens,
                max_prompt_len=self.max_prompt_tokens,
                checkpointing=checkpointing,
                use_phase2_enhancements=True,
                phase2_config=phase2_config,
            )
        else:
            # Fallback to basic implementation
            (
                self.gpt,
                self.mel_pos_embedding,
                self.text_pos_embedding,
                self.mel_layer_pos_embedding,
                self.text_layer_pos_embedding,
            ) = build_phase2_enhanced_gpt_transformer(
                layers=layers,
                model_dim=d_model,
                heads=heads,
                max_mel_seq_len=self.max_mel_tokens,
                max_text_seq_len=self.max_text_tokens,
                max_prompt_len=self.max_prompt_tokens,
                checkpointing=checkpointing,
                use_phase2_enhancements=False,
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
                dim=d_model,
                depth=2,
                dim_context=d_model,
                num_latents=32,
                dim_head=64,
                heads=8,
                ff_mult=4,
                use_flash_attn=False,
            )
        else:
            # XTTS v1 components
            self.prompt_embedding = nn.Embedding(self.num_audio_tokens, d_model)
            self.prompt_pos_embedding = LearnedPositionEmbeddings(24 * 9, d_model)
        
        # Output layers
        self.final_norm = nn.LayerNorm(d_model)
        self.text_head = nn.Linear(d_model, self.number_text_tokens)
        self.mel_head = nn.Linear(d_model, self.num_audio_tokens)
    
    def set_inputs_and_targets(self, input, start_token, stop_token):
        """Set input and target tensors with start/stop tokens"""
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar
    
    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """Set padding areas within MEL tokens"""
        # Set areas that don't correspond to audio to padding
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b] + 1
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
        if attn_mask_text is not None and attn_mask_mel is not None:
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


def create_phase2_enhanced_gpt(config: Phase2GPTConfig):
    """Create a Phase 2 Enhanced GPT model with the given config"""
    return Phase2EnhancedGPT(
        start_text_token=config.start_text_token,
        stop_text_token=config.stop_text_token,
        layers=config.layers,
        d_model=config.d_model,
        heads=config.heads,
        max_text_tokens=config.max_text_tokens,
        max_mel_tokens=config.max_mel_tokens,
        max_prompt_tokens=config.max_prompt_tokens,
        max_conditioning_inputs=config.max_conditioning_inputs,
        code_stride_len=config.code_stride_len,
        number_text_tokens=config.number_text_tokens,
        num_audio_tokens=config.num_audio_tokens,
        start_audio_token=config.start_audio_token,
        stop_audio_token=config.stop_audio_token,
        train_solo_embeddings=config.train_solo_embeddings,
        checkpointing=config.checkpointing,
        average_conditioning_embeddings=config.average_conditioning_embeddings,
        label_smoothing=config.label_smoothing,
        use_perceiver_resampler=config.use_perceiver_resampler,
        perceiver_cond_length_compression=config.perceiver_cond_length_compression,
        use_phase2_enhancements=config.use_phase2_enhancements,
        phase2_config=config.phase2_config,
    )
