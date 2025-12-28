import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import LladaBlockAttention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class LladaDecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        
        # --- Attention Components ---
        tp_size = dist.get_world_size()
        self.total_num_heads = config.n_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.n_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.d_model // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.d_model,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, 'attention_bias', False),
        )

        self.attn_out = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.d_model,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", config.max_sequence_length),
            base=getattr(config, "rope_theta", 500000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # Use LladaBlockAttention for llada model
        self.attn = LladaBlockAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
        # --- MLP Components ---
        self.gate_up_proj = MergedColumnParallelLinear(
            config.d_model,
            [config.mlp_hidden_size] * 2,
            bias=False,
        )
        self.ff_out = RowParallelLinear(
            config.mlp_hidden_size,
            config.d_model,
            bias=False,
        )
        assert config.activation_type == "silu"
        self.act_fn = SiluAndMul()

        # --- Layer-level LayerNorms ---
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ff_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.attn_norm(hidden_states)
        else:
            hidden_states, residual = self.attn_norm(hidden_states, residual)
            
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply rotary embeddings on flattened format
        # reshape -> rotary -> reshape back to flattened
        q_reshaped = q.view(-1, self.num_heads, self.head_dim)
        k_reshaped = k.view(-1, self.num_kv_heads, self.head_dim)
        q_reshaped, k_reshaped = self.rotary_emb(positions, q_reshaped, k_reshaped)
        q = q_reshaped.reshape(q.shape)
        k = k_reshaped.reshape(k.shape)
        
        # Pass flattened q, k, v to attention (LladaBlockAttention will reshape internally)
        o = self.attn(q, k, v)
        hidden_states = self.attn_out(o)
        
        hidden_states, residual = self.ff_norm(hidden_states, residual)
        
        gate_up = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states = self.ff_out(hidden_states)
        
        return hidden_states, residual


class LladaModel(nn.Module):
    """
    Llada model (transformer block structure).
    """
    def __init__(self, config) -> None:
        super().__init__()
        # Embedding layer
        self.wte = VocabParallelEmbedding(config.vocab_size, config.d_model)
        # Decoder layers
        self.blocks = nn.ModuleList([LladaDecoderLayer(config) for _ in range(config.n_layers)])
        # Final layer norm
        self.ln_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # LM head
        self.ff_out = ParallelLMHead(config.vocab_size, config.d_model)
            
        if config.weight_tying:
            self.ff_out.weight.data = self.wte.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        residual = None
        for layer in self.blocks:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states


class LladaForCausalLM(nn.Module):
    """
    Top-level Llada model wrapper for causal language modeling.
    Uses nested structure to match weight file paths: model.transformer.xxx
    """
    
    # Mapping for fused weight loading
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "ff_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        # Create nested structure to match weight file paths: model.transformer.xxx
        self.model = nn.Module()
        self.model.transformer = LladaModel(config)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.transformer(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.model.transformer.ff_out(hidden_states)
        return logits
