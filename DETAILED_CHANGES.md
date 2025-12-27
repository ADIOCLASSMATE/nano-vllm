# SDAR模型重构 - 详细变更对比

## 文件1: nanovllm/models/sdar.py

### 变更 1: 导入调整
```python
# 之前
from nanovllm.layers.attention import BlockAttention

# 之后
from nanovllm.layers.attention import Attention
```

### 变更 2: SDARAttention.__init__ 签名
```python
# 之前 (jetengine风格)
def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_position: int = 4096 * 32,
    head_dim: int | None = None,
    rms_norm_eps: float = 1e-06,
    qkv_bias: bool = False,
    rope_theta: float = 10000,
    rope_scaling: tuple | None = None,
    process_group: dist.ProcessGroup = None,  # ❌ 移除此参数
) -> None:

# 之后 (nanovllm风格)
def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_position: int = 4096 * 32,
    head_dim: int | None = None,
    rms_norm_eps: float = 1e-06,
    qkv_bias: bool = False,
    rope_theta: float = 10000,
    rope_scaling: tuple | None = None,
) -> None:
```

### 变更 3: SDARAttention 初始化代码
```python
# 之前 (jetengine风格)
tp_size = dist.get_world_size(group=process_group)  # ❌ 显式group参数
self.total_num_heads = num_heads
...
self.qkv_bias = qkv_bias  # ❌ 未保存这个标志

self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    process_group,              # ❌ 多余参数
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=qkv_bias,
)
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    process_group,              # ❌ 多余参数
    bias=False,
)
...
self.attn = BlockAttention(...)  # ❌ 使用BlockAttention

self.q_norm = RMSNorm(...)       # ❌ 无条件创建
self.k_norm = RMSNorm(...)       # ❌ 无条件创建

# 之后 (nanovllm风格)
tp_size = dist.get_world_size()  # ✅ 全局调用
self.total_num_heads = num_heads
...
self.qkv_bias = qkv_bias         # ✅ 保存标志

self.qkv_proj = QKVParallelLinear(
    hidden_size,
    self.head_dim,
    self.total_num_heads,        # ✅ 直接传递
    self.total_num_kv_heads,
    bias=qkv_bias,
)
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    bias=False,                  # ✅ 无需process_group
)
...
self.attn = Attention(...)       # ✅ 使用Attention

if not self.qkv_bias:            # ✅ 条件创建
    self.q_norm = RMSNorm(...)
    self.k_norm = RMSNorm(...)
```

### 变更 4: SDARAttention.forward 逻辑
```python
# 之前 (复杂的view和reshape)
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q_by_head = q.view(-1, self.num_heads, self.head_dim)
q_by_head = self.q_norm(q_by_head)      # ❌ 无条件规范化
q = q_by_head.view(q.shape)              # ❌ 无必要的reshape
k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
k_by_head = self.k_norm(k_by_head)      # ❌ 无条件规范化
k = k_by_head.view(k.shape)              # ❌ 无必要的reshape
q, k = self.rotary_emb(positions, q, k)  # ❌ 传递错误形状的q/k
o = self.attn(q, k, v)
output = self.o_proj(o)                  # ❌ o应该被flatten

# 之后 (清晰的view和条件规范化)
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)           # ✅ 直接view
k = k.view(-1, self.num_kv_heads, self.head_dim)        # ✅ 直接view
v = v.view(-1, self.num_kv_heads, self.head_dim)        # ✅ 直接view
if not self.qkv_bias:                   # ✅ 条件规范化
    q = self.q_norm(q)
    k = self.k_norm(k)
q, k = self.rotary_emb(positions, q, k) # ✅ 传递head维度的q/k
o = self.attn(q, k, v)
output = self.o_proj(o.flatten(1, -1))  # ✅ 在传递前flatten
```

### 变更 5: SDARMLP.__init__ 签名
```python
# 之前
def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    process_group: dist.ProcessGroup  # ❌ 移除此参数
) -> None:
    ...
    self.gate_up_proj = MergedColumnParallelLinear(
        hidden_size,
        [intermediate_size] * 2,
        process_group,          # ❌ 移除
        bias=False,
    )
    self.down_proj = RowParallelLinear(
        intermediate_size,
        hidden_size,
        process_group,          # ❌ 移除
        bias=False,
    )

# 之后
def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
) -> None:
    ...
    self.gate_up_proj = MergedColumnParallelLinear(
        hidden_size,
        [intermediate_size] * 2,
        bias=False,             # ✅ 无需process_group
    )
    self.down_proj = RowParallelLinear(
        intermediate_size,
        hidden_size,
        bias=False,             # ✅ 无需process_group
    )
```

### 变更 6: SDARDecoderLayer.__init__ 签名
```python
# 之前
def __init__(
    self,
    config,
    process_group   # ❌ 移除
) -> None:
    ...
    self.self_attn = SDARAttention(
        ...,
        qkv_bias=getattr(config, 'attention_bias', False),  # ❌ 默认False
        ...,
        process_group=process_group  # ❌ 传递process_group
    )
    self.mlp = SDARMLP(
        ...,
        process_group=process_group  # ❌ 传递process_group
    )

# 之后
def __init__(
    self,
    config,
) -> None:
    ...
    self.self_attn = SDARAttention(
        ...,
        qkv_bias=getattr(config, 'attention_bias', True),   # ✅ 默认True
        ...,
    )
    self.mlp = SDARMLP(
        ...,
    )
```

### 变更 7: SDARDecoderLayer.forward 的residual处理
```python
# 之前
if residual is None:
    residual = hidden_states           # ❌ 先保存residual
    hidden_states = self.input_layernorm(hidden_states)  # ❌ 分离调用

# 之后
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states  # ✅ 同时处理
```

### 变更 8: SDARModel.__init__ 签名
```python
# 之前
def __init__(
    self,
    config,
    process_group   # ❌ 移除
) -> None:
    ...
    self.embed_tokens = VocabParallelEmbedding(
        config.vocab_size, config.hidden_size, process_group)  # ❌ 传递process_group
    self.layers = nn.ModuleList([SDARDecoderLayer(
        config, process_group) for _ in range(config.num_hidden_layers)])  # ❌ 传递process_group

# 之后
def __init__(
    self,
    config,
) -> None:
    ...
    self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)  # ✅ 无需process_group
    self.layers = nn.ModuleList([SDARDecoderLayer(config) for _ in range(config.num_hidden_layers)])  # ✅ 无需process_group
```

### 变更 9: SDARForCausalLM.__init__ 签名
```python
# 之前
def __init__(
    self,
    config,
    process_group   # ❌ 移除
) -> None:
    ...
    self.model = SDARModel(config, process_group)  # ❌ 传递process_group
    self.lm_head = ParallelLMHead(
        config.vocab_size, config.hidden_size, process_group)  # ❌ 传递process_group

# 之后
def __init__(
    self,
    config,
) -> None:
    ...
    self.model = SDARModel(config)  # ✅ 无需process_group
    self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)  # ✅ 无需process_group
```

### 变更 10: SDARForCausalLM.forward
```python
# 之前
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hidden_states = self.model(input_ids, positions)
    return hidden_states  # ❌ 额外变量

# 之后
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    return self.model(input_ids, positions)  # ✅ 直接返回
```

### 变更 11: SDARForCausalLM.compute_logits
```python
# 之前
def compute_logits(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    logits = self.lm_head(hidden_states)
    return logits  # ❌ 额外变量

# 之后
def compute_logits(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    return self.lm_head(hidden_states)  # ✅ 直接返回
```

---

## 文件2: nanovllm/engine/model_runner.py

### 变更 1: 模型初始化
```python
# 之前 (line 46)
if 'sdar' in model_type:
    self.model = SDARForCausalLM(hf_config, dist.get_process_group())  # ❌ 传递process_group
else:
    self.model = Qwen3ForCausalLM(hf_config)

# 之后 (line 46)
if 'sdar' in model_type:
    self.model = SDARForCausalLM(hf_config)  # ✅ 无需process_group
else:
    self.model = Qwen3ForCausalLM(hf_config)
```

---

## 总体影响分析

### ✅ 优点
1. **API一致性**：SDAR与Qwen3使用相同的API签名
2. **代码简化**：减少参数传递，代码更清晰
3. **维护性**：与nanovllm核心设计对齐，便于后续维护
4. **扩展性**：遵循nanovllm的分布式设计模式
5. **TP支持**：完整支持Tensor Parallel，通过全局dist状态管理

### ⚠️ 注意事项
1. **权重加载**：packed_modules_mapping保持不变，兼容现有权重
2. **配置依赖**：config必须包含所有必要字段
3. **采样参数**：使用nanovllm中的SamplingParams定义

### 📊 代码量对比
- 总行数：无显著变化（都是~230行）
- 参数移除：5个process_group参数
- 条件语句增加：1个（qkv_bias条件规范化）
- 复杂度降低：简化了forward逻辑中的视图操作

---

## 验证清单

- [x] 所有process_group参数已移除
- [x] dist.get_world_size()调用已全局化
- [x] BlockAttention已替换为Attention
- [x] QKV规范化逻辑已条件化
- [x] 张量操作已简化
- [x] 与Qwen3的API已对齐
- [x] ModelRunner初始化已更新
- [x] 句法错误检查已通过
