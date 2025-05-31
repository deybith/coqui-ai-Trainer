"""
Local imports replacement for TTS dependencies

This module provides local implementations of TTS components
to avoid import dependency issues during testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class ConditioningEncoder(nn.Module):
    """Local implementation of ConditioningEncoder"""
    
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, attn_dim=512, stochastic_inputs=True):
        super().__init__()
        attn_dim = embedding_dim if attn_dim == 0 else attn_dim
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.ModuleList()
        for a in range(attn_blocks):
            self.attn.append(nn.MultiheadAttention(embedding_dim, num_attn_heads, batch_first=True))
        self.dim = embedding_dim
        self.do_checkpointing = False
        self.mean = stochastic_inputs

    def forward(self, x):
        h = self.init(x).permute(0, 2, 1)
        for attn in self.attn:
            h_res = h
            h, _ = attn(h, h, h)
            h = h + h_res
        if self.mean:
            return h.mean(dim=1, keepdim=True)
        else:
            return h[:, 0].unsqueeze(1)


class LearnedPositionEmbeddings(nn.Module):
    """Local implementation of LearnedPositionEmbeddings"""
    
    def __init__(self, max_len, model_dim):
        super().__init__()
        self.max_len = max_len
        self.model_dim = model_dim
        self.embeddings = nn.Embedding(max_len, model_dim)
        
    def forward(self, x):
        # x should be a tensor of token indices
        if x.dim() == 2:
            seq_len = x.size(1)
        else:
            seq_len = x.size(0)
            
        if seq_len > self.max_len:
            seq_len = self.max_len
            
        pos_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0).expand(x.size(0), -1)
        
        return self.embeddings(pos_ids)
    
    def get_fixed_embedding(self, seq_len, device):
        """Get fixed positional embeddings for a given sequence length"""
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        return self.embeddings(pos_ids)


def _prepare_attention_mask_for_generation(inputs, stop_token_tensor, stop_token_tensor2):
    """Local implementation of attention mask preparation"""
    # Create a simple attention mask
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)
    return attention_mask


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Local implementation of GPT2InferenceModel"""
    
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=True):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache
        self.cached_mel_emb = None

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb
    
    def store_prefix_emb(self, prefix_emb):
        self.cached_mel_emb = prefix_emb

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # Simple forward pass for testing
        if inputs_embeds is not None:
            emb = inputs_embeds
        else:
            emb = self.embeddings(input_ids)
            
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=None,
        )


class PerceiverResampler(nn.Module):
    """Local implementation of PerceiverResampler"""
    
    def __init__(self, dim, depth=2, dim_context=None, num_latents=32, dim_head=64, heads=8, ff_mult=4, use_flash_attn=False):
        super().__init__()
        dim_context = dim_context or dim
        
        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    nn.MultiheadAttention(dim, heads, batch_first=True),
                    nn.Sequential(
                        nn.Linear(dim, dim * ff_mult),
                        nn.GELU(),
                        nn.Linear(dim * ff_mult, dim),
                    )
                ])
            )
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]
        x = self.proj_context(x)
        
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        
        for attn, ff in self.layers:
            attn_out, _ = attn(latents, x, x, key_padding_mask=mask)
            latents = attn_out + latents
            latents = ff(latents) + latents
        
        return self.norm(latents)
