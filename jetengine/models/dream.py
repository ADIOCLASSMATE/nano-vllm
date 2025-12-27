# Dream model implementation for JetEngine inference engine
# Dream predicts next-token logits (shifted by 1) unlike LLaDA which predicts current token

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from jetengine.layers.activation import SiluAndMul
from jetengine.layers.attention import LladaBlockAttention  # Reuse bidirectional attention
from jetengine.layers.layernorm import RMSNorm
from jetengine.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear
)
from jetengine.layers.rotary_embedding import get_rope
from jetengine.layers.embed_head import VocabParallelEmbedding
from jetengine.engine.sequence import RunType
from jetengine.utils.context import get_context


class DreamParallelLMHead(VocabParallelEmbedding):
    """
    Dream-specific LM Head that handles next-token prediction.
    
    Key difference from standard ParallelLMHead:
    - Dream model predicts the NEXT token at each position
    - hidden_state[i] predicts token[i+1]
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        process_group: dist.ProcessGroup,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim, process_group)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        Compute logits from hidden states.
        
        For Dream: hidden_state[i] predicts token[i+1]
        During prefill, we take the last hidden state of each sequence.
        During denoise, we return logits for all block positions.
        """
        context = get_context()
        
        if context.run_type == RunType.PREFILL:
            # Take last hidden state of each sequence
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0, group=self.process_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits


class DreamAttention(nn.Module):
    """
    Dream attention module with bidirectional attention.
    Module path: model.layers.N.self_attn.*
    """

    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        
        tp_size = dist.get_world_size(group=process_group)
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Dream uses bias in q_proj, k_proj, and v_proj
        # Weight names: model.layers.N.self_attn.{q,k,v}_proj.{weight,bias}
        # Fused to: model.layers.N.self_attn.qkv_proj.{weight,bias}
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            process_group,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,  # Dream has attention bias
        )

        # Weight name: model.layers.N.self_attn.o_proj.weight
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            process_group,
            bias=False,
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # Use bidirectional attention (same as LLaDA)
        self.attn = LladaBlockAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class DreamMLP(nn.Module):
    """
    Dream MLP module.
    Module path: model.layers.N.mlp.*
    """

    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        
        # Weight names: model.layers.N.mlp.{gate,up}_proj.weight
        # Fused to: model.layers.N.mlp.gate_up_proj.weight
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            process_group,
            bias=False,
        )
        
        # Weight name: model.layers.N.mlp.down_proj.weight
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            process_group,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DreamDecoderLayer(nn.Module):
    """
    Dream decoder layer with bidirectional attention.
    Module path: model.layers.N.*
    
    Weight structure:
    - model.layers.N.input_layernorm.weight
    - model.layers.N.post_attention_layernorm.weight
    - model.layers.N.self_attn.{q,k,v,o}_proj.*
    - model.layers.N.mlp.{gate,up,down}_proj.*
    """

    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        
        # Attention sub-module (matches self_attn.* in weight names)
        self.self_attn = DreamAttention(config, process_group)
        
        # MLP sub-module (matches mlp.* in weight names)
        self.mlp = DreamMLP(config, process_group)

        # LayerNorms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        # Self Attention
        hidden_states = self.self_attn(positions, hidden_states)
        
        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class DreamModel(nn.Module):
    """
    Dream base model (transformer backbone).
    Module path: model.*
    
    Weight structure:
    - model.embed_tokens.weight
    - model.layers.N.*
    - model.norm.weight
    """
    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        # model.embed_tokens.weight
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, process_group)
        # model.layers.N...
        self.layers = nn.ModuleList([DreamDecoderLayer(
            config, process_group) for _ in range(config.num_hidden_layers)])
        # model.norm.weight
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DreamForCausalLM(nn.Module):
    """
    Dream model for causal language modeling.
    
    Key difference from LLaDA:
    - Dream predicts NEXT token logits (shifted by 1 position)
    - LLaDA predicts CURRENT token logits (no shift)
    - Both use bidirectional attention for diffusion
    
    Weight mapping matches HuggingFace Dream checkpoint:
    - model.embed_tokens.weight
    - model.layers.N.self_attn.{q,k,v}_proj.{weight,bias} -> self_attn.qkv_proj
    - model.layers.N.self_attn.o_proj.weight -> self_attn.o_proj
    - model.layers.N.mlp.{gate,up}_proj.weight -> mlp.gate_up_proj
    - model.layers.N.mlp.down_proj.weight -> mlp.down_proj
    - model.layers.N.{input_layernorm,post_attention_layernorm}.weight
    - model.norm.weight
    - lm_head.weight
    """
    
    # This mapping tells the loader how to handle the fused weights.
    # For Dream model:
    # - model.layers.N.self_attn.q_proj.* -> model.layers.N.self_attn.qkv_proj.* (shard "q")
    # - model.layers.N.self_attn.k_proj.* -> model.layers.N.self_attn.qkv_proj.* (shard "k")
    # - model.layers.N.self_attn.v_proj.* -> model.layers.N.self_attn.qkv_proj.* (shard "v")
    # - model.layers.N.mlp.gate_proj.* -> model.layers.N.mlp.gate_up_proj.* (shard 0)
    # - model.layers.N.mlp.up_proj.* -> model.layers.N.mlp.gate_up_proj.* (shard 1)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        process_group
    ) -> None:
        super().__init__()
        self.config = config
        # The 'model' attribute matches 'model.' prefix in weight_map
        self.model = DreamModel(config, process_group)
        # lm_head.weight - separate from model (no 'model.' prefix)
        self.lm_head = DreamParallelLMHead(
            config.vocab_size, config.hidden_size, process_group)
        
        # Handle weight tying if configured
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.model.embed_tokens.weight
                
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returns hidden states.
        """
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits from hidden states.
        
        For Dream model, the logits at position i predict token i+1.
        This is the standard next-token prediction paradigm.
        The shifting/alignment logic should be handled in the sampling step.
        """
        logits = self.lm_head(hidden_states)
        return logits
