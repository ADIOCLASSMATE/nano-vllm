# SDAR 张量形状修复 - Bug修复日志

## 错误现象

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (524288x128 and 4096x4096)
```

发生在`o_proj` forward()调用时，说明BlockAttention返回的张量形状不正确。

## 根本原因

对比jetengine的实现发现，问题在于：

1. **BlockAttention的输入/输出形状约定**：
   - jetengine: 输入是**平展的** `(seq_len, num_heads*head_dim)`，内部reshape，输出也是**平展的**
   - 之前的nanovllm实现: 误认为输入是**已view的** `(seq_len, num_heads, head_dim)`

2. **SDARAttention规范化逻辑**：
   - jetengine: view for norm，然后**恢复原形状**后传给BlockAttention
   - 之前的nanovllm: 直接保持view后的形状传给BlockAttention

3. **q_norm/k_norm创建策略**：
   - jetengine: **总是创建**（不管qkv_bias的值）
   - 之前的nanovllm: 条件创建（`if not self.qkv_bias`）

## 修复方案

### 1. nanovllm/layers/attention.py - BlockAttention

**关键改动：**

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # 输入: (seq_len, num_heads*head_dim) - 平展形式
    
    # 内部reshape用于处理
    q = q.view(-1, self.num_heads, self.head_dim)
    k = k.view(-1, self.num_kv_heads, self.head_dim)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    
    # ... attention计算 ...
    
    # 最后必须恢复平展形式
    o = o.view(-1, self.num_heads * self.head_dim)
    return o
```

**重点：** 所有返回路径都必须执行最后的reshape，确保返回 `(seq_len, num_heads*head_dim)` 的形式。

### 2. nanovllm/models/sdar.py - SDARAttention

**关键改动：**

```python
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)  # 返回平展形状
    q, k, v = qkv.split([...])
    
    # 1. 规范化（需要view for norm）
    q_by_head = q.view(-1, self.num_heads, self.head_dim)
    q_by_head = self.q_norm(q_by_head)
    q = q_by_head.view(q.shape)  # 恢复平展！
    
    k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
    k_by_head = self.k_norm(k_by_head)
    k = k_by_head.view(k.shape)  # 恢复平展！
    
    # 2. Rotary embeddings（需要view for rotary_emb）
    q_for_rope = q.view(-1, self.num_heads, self.head_dim)
    k_for_rope = k.view(-1, self.num_kv_heads, self.head_dim)
    q_for_rope, k_for_rope = self.rotary_emb(positions, q_for_rope, k_for_rope)
    
    q = q_for_rope.view(q.shape)  # 恢复平展！
    k = k_for_rope.view(k.shape)  # 恢复平展！
    
    # 3. 传给BlockAttention（现在q, k, v都是平展的）
    o = self.attn(q, k, v)  # 返回 (seq_len, num_heads*head_dim)
    
    # 4. 直接传给o_proj
    output = self.o_proj(o)
    return output
```

**q_norm/k_norm策略修正：**
```python
# 总是创建（不是条件创建）
self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
```

## 张量流动对比

### jetengine（正确）
```
qkv (seq, num_heads*head_dim + num_kv_heads*2*head_dim)
  ↓ split
q, k, v (seq, num_heads*head_dim), (seq, num_kv_heads*head_dim), ...
  ↓ view for norm
q_by_head (seq, num_heads, head_dim)
  ↓ norm
  ↓ view(q.shape) - 恢复平展
q (seq, num_heads*head_dim)
  ↓ view for rotary
q_for_rope (seq, num_heads, head_dim)
  ↓ rotary
  ↓ view(q.shape) - 恢复平展
q (seq, num_heads*head_dim)
  ↓ BlockAttention(q, k, v)
o (seq, num_heads*head_dim)
  ↓ o_proj
output (seq, hidden)
```

### nanovllm新实现（修复后）
```
与jetengine完全相同 ✓
```

## 关键学习

| 方面 | jetengine | nanovllm新 |
|------|-----------|-----------|
| 输入形状 | 平展 `(seq, feat)` | 平展 `(seq, feat)` |
| 规范化 | view→norm→view(原形) | view→norm→view(原形) |
| Rotary | view→rotary→view(原形) | view→rotary→view(原形) |
| BlockAttention | 接收平展，返回平展 | 接收平展，返回平展 |
| q_norm/k_norm | 总是创建 | 总是创建 |

## 验证

✅ BlockAttention最后的view(-1, num_heads*head_dim)保证输出形状正确
✅ SDARAttention在每次操作后恢复平展形状
✅ q_norm/k_norm无条件创建
✅ 语法检查通过

## 后续测试

应该能正常运行推理了：
```bash
python example_sdar.py
```
