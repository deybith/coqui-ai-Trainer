"""
Simplified local imports without transformers inheritance
to isolate the hanging issue.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditioningEncoder(nn.Module):
    """Simplified ConditioningEncoder"""
    
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=4, attn_dim=512, stochastic_inputs=True):
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.ModuleList()
        for a in range(attn_blocks):
            self.attn.append(nn.MultiheadAttention(embedding_dim, num_attn_heads, batch_first=True))
        self.dim = embedding_dim
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
    """Simplified LearnedPositionEmbeddings"""
    
    def __init__(self, max_len, model_dim):
        super().__init__()
        self.max_len = max_len
        self.model_dim = model_dim
        self.embeddings = nn.Embedding(max_len, model_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            seq_len = x.size(1)
        else:
            seq_len = x.size(0)
            
        if seq_len > self.max_len:
            seq_len = self.max_len
            
        pos_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0).expand(x.size(0), -1)
        
        return self.embeddings(pos_ids)


class SimpleGPT2InferenceModel(nn.Module):
    """Simplified GPT2InferenceModel without transformers inheritance"""
    
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=True):
        super().__init__()
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

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            emb = inputs_embeds
        else:
            emb = self.embeddings(input_ids)
            
        hidden_states = self.transformer(emb)
        lm_logits = self.lm_head(hidden_states)
        
        return type('Output', (), {'logits': lm_logits})()


class PerceiverResampler(nn.Module):
    """Simplified PerceiverResampler"""
    
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


def _prepare_attention_mask_for_generation(inputs, stop_token_tensor, stop_token_tensor2):
    """Simple attention mask preparation"""
    attention_mask = torch.ones_like(inputs, dtype=torch.bool)
    return attention_mask
