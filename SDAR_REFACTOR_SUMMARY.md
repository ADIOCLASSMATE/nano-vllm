# SDAR模型重构总结

## 概述
将nanovllm中的SDAR模型实现从完全照抄jetengine逻辑改为**完全对齐nanovllm的设计模式**，特别是在TP（Tensor Parallel）分布式训练策略上。

## 核心修改点

### 1. **移除process_group参数** ✓
**原因：** nanovllm使用全局的`dist.get_world_size()`和`dist.get_rank()`，不依赖显式的process_group参数。这与jetengine的差异体现在：
- jetengine：显式传递process_group给每一个分布式层
- nanovllm：隐式使用全局分布式状态

**修改文件：**
- `nanovllm/models/sdar.py`：所有类去除process_group参数
- `nanovllm/engine/model_runner.py`：模型初始化不再传递process_group

**具体改动：**

#### SDARAttention
```python
# 之前（jetengine风格）
def __init__(self, hidden_size, num_heads, num_kv_heads, ..., process_group=None):
    tp_size = dist.get_world_size(group=process_group)
    self.qkv_proj = QKVParallelLinear(..., process_group, ...)
    self.o_proj = RowParallelLinear(..., process_group, ...)

# 之后（nanovllm风格）
def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
    tp_size = dist.get_world_size()
    self.qkv_proj = QKVParallelLinear(...)
    self.o_proj = RowParallelLinear(...)
```

#### SDARMLP
```python
# 之前
def __init__(self, hidden_size, intermediate_size, hidden_act, process_group):
    self.gate_up_proj = MergedColumnParallelLinear(..., process_group, ...)
    self.down_proj = RowParallelLinear(..., process_group, ...)

# 之后
def __init__(self, hidden_size, intermediate_size, hidden_act):
    self.gate_up_proj = MergedColumnParallelLinear(...)
    self.down_proj = RowParallelLinear(...)
```

#### SDARDecoderLayer、SDARModel、SDARForCausalLM
所有类都移除了process_group参数的传递链。

### 2. **统一Attention实现** ✓
**原因：** 虽然jetengine有BlockAttention用于稀疏注意，但nanovllm的Attention已支持所有必要功能。

**修改：**
```python
# 之前
from nanovllm.layers.attention import BlockAttention
self.attn = BlockAttention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

# 之后
from nanovllm.layers.attention import Attention
self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
```

### 3. **对齐Q/K规范化逻辑** ✓
**原因：** Qwen3和SDAR都使用optional的Q/K规范化（基于qkv_bias），但实现方式需要一致。

**修改：**
```python
# 之前（过度规范化）
if not self.qkv_bias:
    self.q_norm = RMSNorm(...)
    self.k_norm = RMSNorm(...)

def forward(self, positions, hidden_states):
    q = q.view(-1, self.num_heads, self.head_dim)
    q = self.q_norm(q)  # 总是规范化
    q = q.view(q.shape)
    # 重复同样操作用于k

# 之后（按条件规范化）
if not self.qkv_bias:
    self.q_norm = RMSNorm(...)
    self.k_norm = RMSNorm(...)

def forward(self, positions, hidden_states):
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    if not self.qkv_bias:  # 仅当qkv_bias为False时规范化
        q = self.q_norm(q)
        k = self.k_norm(k)
```

### 4. **优化张量形状处理** ✓
**原因：** 减少不必要的reshape操作，直接使用flatten获得最终形状。

**修改：**
```python
# 之前
q_by_head = q.view(-1, self.num_heads, self.head_dim)
q_by_head = self.q_norm(q_by_head)
q = q_by_head.view(q.shape)  # 改回原始形状
# ...
o = self.attn(q, k, v)  # o仍是分离的头维度
output = self.o_proj(o)  # 假设o已flatten

# 之后
q = q.view(-1, self.num_heads, self.head_dim)  # 直接view为头维度
if not self.qkv_bias:
    q = self.q_norm(q)
# ...
o = self.attn(q, k, v)  # 返回已reshape的张量
output = self.o_proj(o.flatten(1, -1))  # 在o_proj之前flatten
```

### 5. **对齐Embedding和LMHead初始化** ✓
**原因：** nanovllm的VocabParallelEmbedding和ParallelLMHead不需要process_group参数。

**修改：**
```python
# 之前
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size, process_group)
self.lm_head = ParallelLMHead(vocab_size, hidden_size, process_group)

# 之后
self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
self.lm_head = ParallelLMHead(vocab_size, hidden_size)
```

### 6. **更新ModelRunner初始化** ✓
**文件：** `nanovllm/engine/model_runner.py`

```python
# 之前
self.model = SDARForCausalLM(hf_config, dist.get_process_group())

# 之后
self.model = SDARForCausalLM(hf_config)
```

### 7. **统一forward方法返回值** ✓
**原因：** Qwen3ForCausalLM直接返回model的输出，SDAR现在也这样做。

```python
# 之前
def forward(self, input_ids, positions):
    hidden_states = self.model(input_ids, positions)
    return hidden_states

# 之后
def forward(self, input_ids, positions):
    return self.model(input_ids, positions)
```

## 关键变化总结表

| 方面 | jetengine风格 | nanovllm风格（新） |
|------|---------------|------------------|
| 分布式 | 显式process_group | 隐式全局dist状态 |
| 注意力 | BlockAttention | Attention |
| Q/K规范化 | 无条件 | 有条件（基于qkv_bias） |
| Embedding | 需要process_group | 不需要process_group |
| LMHead | 需要process_group | 不需要process_group |

## TP（Tensor Parallel）处理一致性

SDAR现在完全采用nanovllm的TP处理方式：

1. **分布式状态管理：** 通过全局`dist.get_world_size()`和`dist.get_rank()`
2. **层级隔离分片：** 每个分布式层（ColumnParallel、RowParallel等）内部处理分片逻辑
3. **通信隐藏：** RowParallel中的all_reduce隐含在层的forward中
4. **一致的激活函数：** 使用SiluAndMul实现SiLU+乘法融合

## 测试要点

- [x] 所有导入无误
- [x] 句法正确性
- [x] 参数签名一致性
- [x] TP逻辑与Qwen3对齐
- [x] 没有遗留的process_group引用
- [ ] 实际推理验证（待端到端测试）
- [ ] TP多卡推理验证（待多GPU环境测试）

## 兼容性说明

### 权重加载
packed_modules_mapping保持不变，确保权重加载兼容性。

### 配置要求
SDAR模型config应包含：
- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `intermediate_size`
- `max_position_embeddings`
- `rms_norm_eps`
- `attention_bias`（可选，默认True）
- `head_dim`（可选）
- `rope_theta`（可选）
- `rope_scaling`（可选）

## 后续优化方向

1. 可选：实现SDAR特定的稀疏注意优化（目前使用通用Attention）
2. 可选：实现Block-aware的KV缓存管理
3. 持续：与jetengine的性能对比验证
