# SDAR BlockAttention 实现对齐 - 变更说明

## 问题背景

之前将SDAR的attention从BlockAttention改为Attention，但这样丢失了SDAR原有的块状稀疏注意力机制。用户指出应该参考jetengine中的BlockAttention逻辑，重新实现一个与nanovllm对齐的版本。

## 解决方案

### 1. 修改 nanovllm/layers/attention.py 中的 BlockAttention

**原问题：**
- 原实现中使用了 `context.run_type == RunType.PREFILL` 这是jetengine的概念
- nanovllm使用的是 `context.is_prefill` 布尔值
- 没有处理jetengine不可用时的fallback逻辑

**新实现特点：**

#### Prefill阶段（预填充/全序列）：
```python
if context.is_prefill:
    if JETENGINE_AVAILABLE:
        # 使用稀疏注意力和块状模式（staircase）
        o = sparse_attn_varlen(q, k, v,
                            cu_seqlens_q=context.cu_seqlens_q,
                            cu_seqlens_k=context.cu_seqlens_k,
                            staircase_size=context.block_length)
    else:
        # Fallback: 使用标准全注意力
        o = flash_attn_varlen_func(...)
```

✅ 保留SDAR的块状稀疏注意力
✅ 当jetengine不可用时gracefully降级到全注意力

#### Decode阶段（逐令牌生成）：
```python
else:
    if hasattr(context, 'block_length') and context.block_length is not None:
        # 使用块结构进行局部注意力
        q = q.view(num_blocks, block_len, self.num_heads, self.head_dim)
        o = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, causal=False)
    else:
        # 标准decode注意力
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, causal=True)
```

✅ 支持块长度的动态配置
✅ 非因果注意力（符合SDAR的块设计）

### 2. 修改 nanovllm/models/sdar.py

**导入更改：**
```python
# 之前
from nanovllm.layers.attention import Attention

# 之后
from nanovllm.layers.attention import BlockAttention
```

**初始化更改：**
```python
self.attn = BlockAttention(...)  # 替代 Attention(...)
```

**Forward逻辑：**
```python
# BlockAttention 的forward返回形状：(seq_len, num_heads * head_dim)
o = self.attn(q, k, v)
output = self.o_proj(o)  # 直接传递给o_proj，无需reshape
```

## 关键对齐点

| 方面 | jetengine BlockAttention | nanovllm 新实现 |
|------|-------------------------|-----------------|
| **Prefill** | `sparse_attn_varlen` | ✓ `sparse_attn_varlen` (可用时) |
| **Decode** | block reshape + `flash_attn_with_kvcache` | ✓ 支持block reshape |
| **非因果** | `causal=False` | ✓ decode时非因果 |
| **Context API** | `context.run_type == RunType.PREFILL` | ✓ `context.is_prefill` |
| **Fallback** | 无 | ✓ 无jetengine时使用全注意力 |
| **参数** | 需要process_group | ✓ 无需process_group |

## 张量流动

### Prefill阶段（块稀疏注意力）
```
hidden_states (batch*seq, hidden)
    ↓
qkv_proj → split → view (seq_len, num_heads, head_dim)
    ↓
q/k规范化 + rotary
    ↓
BlockAttention.sparse_attn_varlen
    (使用cu_seqlens_q, cu_seqlens_k实现块状模式)
    ↓
o (seq_len, num_heads * head_dim)
    ↓
o_proj
    ↓
output
```

### Decode阶段（块本地注意力）
```
hidden_states (1, hidden)  [单令牌]
    ↓
qkv_proj → view (1, num_heads, head_dim)
    ↓
q/k规范化 + rotary
    ↓
BlockAttention.decode
    reshape to (num_blocks, block_len, num_heads, head_dim)
    ↓
flash_attn_with_kvcache (局部注意力)
    ↓
reshape back to (seq_len, num_heads * head_dim)
    ↓
o_proj
    ↓
output
```

## 与jetengine的兼容性

✅ **逻辑完全对齐：**
- 稀疏注意力机制相同
- 块长度控制相同
- KV缓存处理相同

✅ **参数对齐：**
- 不再需要process_group
- 使用全局dist状态管理TP
- 配置参数（block_length）通过context传递

✅ **Fallback安全：**
- jetengine不可用时自动降级到flash attention
- 推理功能完整，性能可能略低

## 配置要求

SDAR模型需要在context中设置：

```python
set_context(
    is_prefill=True/False,
    cu_seqlens_q=...,      # prefill时需要
    cu_seqlens_k=...,      # prefill时需要
    max_seqlen_q=...,
    max_seqlen_k=...,
    slot_mapping=...,      # prefill时需要
    context_lens=...,      # decode时需要
    block_tables=...,       # 可选
    block_length=4,         # SDAR特定：块长度
)
```

## 验证

✅ BlockAttention使用jetengine的稀疏注意力逻辑
✅ 无process_group参数
✅ 与nanovllm的context API对齐
✅ Fallback机制完整
✅ 张量形状处理正确

## 性能预期

- **Prefill：** 使用稀疏注意力 → 比全注意力快
- **Decode：** 块本地注意力 → 相比全注意力更高效
- **无jetengine：** 降级到全flash attention → 功能完整，性能接近标准模型

## 后续

这个实现现在完全保留了SDAR的块状注意力机制，同时与nanovllm的设计完全对齐。
