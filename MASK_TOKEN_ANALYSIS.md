# Nanovllm 中 mask_token_id 和 Denoising 处理分析

## 执行总结

**关键发现**：nanovllm 目前**未实现** mask_token_id 和 block denoising 功能。这些功能仅在 jetengine 中得到完整实现。nanovllm 是一个简化版本，专注于标准因果语言模型推理。

---

## 详细分析

### 1. Config 中 mask_token_id 的使用

#### ✅ nanovllm/config.py
```python
@dataclass
class Config:
    ...
    # SDAR-specific parameters
    mask_token_id: int = -1
    block_length: int = 4
```

**状态**：mask_token_id 在 Config 中被**定义但未使用**
- 默认值为 -1（无效值）
- 没有在任何调度、模型前向或采样逻辑中被引用
- block_length 仅是配置参数，但没有与 denoising 相关的逻辑

---

### 2. Sequence 初始化中的 mask_token 处理

#### ❌ nanovllm/engine/sequence.py
```python
class Sequence:
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

**状态**：**完全缺少 denoising 相关字段**
- 无 `mask_token_id` 参数
- 无 `intermediate_block_tokens` 字段（去噪中间块）
- 无 `denoising_steps` 跟踪
- 无 `current_denoising_step` 计数器
- 无 `block_trajectory` 或 `block_entropies` 跟踪

#### ✅ jetengine/engine/sequence.py (对比参考)
```python
class Sequence:
    def __init__(self, prompt_token_ids: list[int], mask_token_id: int, 
                 sampling_params=SamplingParams(), block_size: int | None = None):
        ...
        self.prompt_token_ids = prompt_token_ids
        self.num_prefill_tokens = (prompt_len // self.block_length) * self.block_length
        first_denoise_part = self.prompt_token_ids[self.num_prefill_tokens:]
        self.generation_start_index = len(first_denoise_part)
        
        # ✅ 中间块初始化（去噪的核心）
        self.intermediate_block_tokens = first_denoise_part + \
            [mask_token_id] * (self.block_length - len(first_denoise_part))
        self.intermediate_block_tokens_entropy = [0.0] * self.block_length
        
        # ✅ 去噪步骤跟踪
        self.current_denoising_step = 0
        self.global_denoising_step = 0
        self.denoising_steps = sampling_params.denoising_steps
        
        # ✅ 状态管理
        self.status = SequenceStatus.DENOISING if self.num_prefill_tokens == 0 \
                     else SequenceStatus.WAITING
```

---

### 3. Scheduler 中的 Denoising 逻辑

#### ❌ nanovllm/engine/scheduler.py
```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(...)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill phase
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            # ... 标准 prefill 逻辑
        
        # decode phase
        while self.running and num_seqs < self.max_num_seqs:
            # ... 标准 decode 逻辑
        
        return scheduled_seqs, is_prefill_flag
```

**状态**：**完全缺失 denoising 支持**
- 无 `mask_token_id` 属性
- 无 `SequenceStatus.DENOISING` 处理
- 无 `RunType` 枚举（PREFILL vs DENOISE）
- 无重掩码策略（sequential, low_confidence, entropy_bounded）
- 无中间块完成检查
- 返回简单的布尔值（is_prefill），而不是 ScheduleResult

#### ✅ jetengine/engine/scheduler.py (对比参考)

```python
class Scheduler:
    def __init__(self, config: Config):
        self.mask_token_id = config.mask_token_id  # ✅ 存储 mask token
        ...
        self.running: list[Sequence] = []
        self.waiting_prefill: deque[Sequence] = deque()
        self.prefill_ready: deque[Sequence] = deque()

    def schedule(self) -> ScheduleResult:
        # ✅ 返回 {prefill: [...], denoise: [...]}
        return ScheduleResult(prefill=prefill_seqs, denoise=denoise_seqs)
    
    def postprocess_unify(self, seqs: list[Sequence], logits: torch.Tensor, 
                         run_type: RunType) -> list[Sequence]:
        # ✅ 区分 PREFILL 和 DENOISE 运行类型
        if run_type == RunType.PREFILL:
            # Prefill 逻辑
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        
        elif run_type == RunType.DENOISE:
            # ✅ Denoising 逻辑示例：
            device = logits.device
            batch_size = len(seqs)
            block_len = seqs[0].block_length
            
            # 计算概率和取样
            probs = self.sample_pipe(logits, temperature=seqs[0].temperature)\
                .view(batch_size, block_len, -1)
            
            # ✅ 获取当前中间块
            batch_current_tokens = torch.tensor(
                [seq.intermediate_block_tokens for seq in seqs], 
                device=device, dtype=torch.int64
            )
            
            # ✅ 识别掩码位置
            for seq_idx, seq in enumerate(seqs):
                current_block_tensor = torch.tensor(
                    seq.intermediate_block_tokens, device=logits.device
                )
                mask_index = (current_block_tensor == self.mask_token_id)
                
                # ✅ 根据重掩码策略选择要取消掩码的位置
                if seq.remasking_strategy == 'sequential':
                    # 从第一个掩码开始取消
                    first_mask_pos = mask_index.nonzero()[0].min().item()
                    num_to_transfer = seq.num_transfer_tokens_per_step[
                        seq.current_denoising_step
                    ]
                    transfer_index[first_mask_pos:first_mask_pos+num_to_transfer] = True
                
                elif seq.remasking_strategy == 'low_confidence_static':
                    # 根据置信度选择最低置信度的掩码
                    confidence = torch.where(
                        mask_index, seq_x0_p, -np.inf
                    )
                    _, top_indices = torch.topk(confidence, num_to_transfer)
                    transfer_index[top_indices] = True
                
                elif seq.remasking_strategy == 'entropy_bounded':
                    # 选择总熵不超过阈值的掩码
                    P = block_probs[mask_index]
                    entropies = -(P * P.log()).sum(dim=-1)
                    ent_sorted, order = torch.sort(entropies, descending=False)
                    cumsum = torch.cumsum(ent_sorted, dim=0)
                    k = torch.searchsorted(cumsum, seq.eb_threshold).item()
                    selected_indices = order[:k]
                    transfer_index[selected_indices] = True
                
                # ✅ 更新中间块
                seq.intermediate_block_tokens[transfer_index] = \
                    seq_x0[transfer_index]
                seq.current_denoising_step += 1
                
                # ✅ 检查块是否完全去噪
                is_fully_denoised = \
                    (self.mask_token_id not in seq.intermediate_block_tokens) or \
                    (seq.current_denoising_step >= seq.denoising_steps)
                
                if is_fully_denoised:
                    seq.status = SequenceStatus.SAVING
```

---

### 4. Model Runner 中的 Denoise Forward 逻辑

#### ❌ nanovllm/engine/model_runner.py
```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = \
        self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures, top_ps = \
        self.prepare_sample(seqs) if self.rank == 0 else (None, None)
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures, top_ps).tolist() \
        if self.rank == 0 else None
    reset_context()
    return token_ids

def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, 
              is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Standard decode with cuda graph
        ...
```

**状态**：**仅支持 PREFILL 和 DECODE，无 DENOISE 支持**
- 参数为简单布尔值 `is_prefill`
- 无 `run_type: RunType` 参数
- 无 mask token 掩码生成
- 无 intermediate block 处理
- 无去噪步骤跟踪

#### ✅ jetengine/engine/model_runner.py (对比参考)

```python
def prepare_denoise(self, seqs: list[Sequence]):
    # ✅ 从 intermediate_block_tokens 构建输入，而非 token_ids
    input_ids = []
    positions = []
    slot_mapping = []
    
    for seq in seqs:
        # ✅ 使用中间块而非标准 token_ids
        input_ids.extend(seq.intermediate_block_tokens)
        positions.extend(range(seq.block_length))
        
        # ✅ 设置 slot mapping
        for pos in range(seq.block_length):
            # 计算物理块和偏移
            ...
    
    set_context(
        run_type=RunType.DENOISE,  # ✅ 明确运行类型
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=block_length,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_last_denoise_step=is_last_step,  # ✅ 跟踪最后去噪步骤
        block_length=self.config.block_length
    )
    return input_ids, positions

def run(self, seqs: list[Sequence], run_type: RunType) -> torch.Tensor:
    # ✅ 根据运行类型分派
    if run_type == RunType.PREFILL:
        input_ids, positions = self.prepare_prefill(seqs)
        logits = self.run_model(input_ids, positions)
    elif run_type == RunType.DENOISE:
        input_ids, positions = self.prepare_denoise(seqs)
        logits = self.run_model(input_ids, positions)  # ✅ 同一模型前向，不同输入
    
    return logits
```

---

### 5. Context 中的 Denoising 信息

#### ❌ nanovllm/utils/context.py
```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

def set_context(is_prefill, cu_seqlens_q=None, ...):
    _CONTEXT = Context(is_prefill, ...)
```

**状态**：**简化设计，仅跟踪 is_prefill 布尔值**
- 无 `run_type: RunType`
- 无 `is_last_denoise_step` 列表
- 无 `block_length`

#### ✅ jetengine/utils/context.py (对比参考)

```python
@dataclass
class Context:
    run_type: RunType | None = None        # ✅ PREFILL 或 DENOISE
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])  # ✅
    block_length: int = 4  # ✅

def set_context(run_type, cu_seqlens_q=None, ..., 
                is_last_denoise_step=[False], block_length=4):  # ✅
    _CONTEXT = Context(run_type, ..., is_last_denoise_step, block_length)
```

---

### 6. SamplingParams 中的 Denoising 参数

#### ✅ nanovllm/sampling_params.py
```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # Block Diffusion Parameters (for SDAR models)
    block_length: int = 4
    denoising_steps: int = 4              # ✅ 定义了参数
    dynamic_threshold: float = 0.9
    eb_threshold: float = 0.35
    topk: int = 0
    topp: float = 1.0
    remasking_strategy: Literal['sequential', 'low_confidence_static', 
                                'low_confidence_dynamic', 'entropy_bounded'] = \
        'low_confidence_static'
    stop_words: list[int] | None = None
```

**状态**：参数已定义但**未被使用**
- nanovllm 的 Sequence 初始化不接受这些参数
- Scheduler 完全忽略重掩码策略
- Model runner 无去噪逻辑

---

## 对比总结表

| 功能点 | nanovllm | jetengine |
|--------|----------|-----------|
| **Config.mask_token_id** | ✅ 定义 | ✅ 定义 + 使用 |
| **Sequence.mask_token_id** | ❌ 无 | ✅ 有 |
| **Sequence.intermediate_block_tokens** | ❌ 无 | ✅ 有 |
| **Sequence.current_denoising_step** | ❌ 无 | ✅ 有 |
| **Sequence.denoising_steps** | ❌ 无 | ✅ 有 |
| **SequenceStatus.DENOISING** | ❌ 无 | ✅ 有 |
| **RunType 枚举** | ❌ 无 | ✅ PREFILL/DENOISE |
| **Scheduler.mask_token_id** | ❌ 无 | ✅ 有 |
| **重掩码策略** | ❌ 无 | ✅ 4 种 |
| **Context.run_type** | ❌ 仅 is_prefill | ✅ RunType 枚举 |
| **Context.is_last_denoise_step** | ❌ 无 | ✅ 有 |
| **ModelRunner.prepare_denoise** | ❌ 无 | ✅ 有 |
| **模型前向（denoise）** | ❌ 无 | ✅ 有 |

---

## 结论

### nanovllm 当前状态
- **一个简化的 LLM 推理引擎**，专注于标准因果语言模型（Qwen3, Llada, SDAR）
- **仅支持标准 Prefill-Decode 范式**
- 配置中包含 mask_token_id 和 block_length 字段，但**纯粹是占位符**
- SamplingParams 包含 denoising_steps 等参数，但在代码路径中**完全被忽略**

### 迁移工作量评估
如果要将 LLADA 的块去噪模式添加到 nanovllm，需要：

1. **Sequence 层**（中等工作量）
   - 添加 `mask_token_id` 参数传递
   - 实现 `intermediate_block_tokens` 初始化逻辑
   - 添加 denoising 步骤跟踪字段

2. **Scheduler 层**（高工作量）
   - 实现 `RunType` 枚举和运行类型区分
   - 实现重掩码策略（sequential, confidence-based, entropy-bounded）
   - 中间块完成性检查和状态转换

3. **ModelRunner 层**（中等工作量）
   - 添加 `prepare_denoise()` 方法
   - 修改 `run()` 方法签名使用 `RunType`
   - 处理 intermediate_block_tokens 作为输入

4. **Context 层**（低工作量）
   - 将 `is_prefill: bool` 替换为 `run_type: RunType`
   - 添加 `is_last_denoise_step` 和 `block_length`

**总估算**：3-5 个工作日，取决于模型本身是否需要适配（attention 层、采样器等）

---

## 代码位置参考

### nanovllm
- Config: [nanovllm/config.py](nanovllm/config.py#L21)
- SamplingParams: [nanovllm/sampling_params.py](nanovllm/sampling_params.py#L14)
- Sequence: [nanovllm/engine/sequence.py](nanovllm/engine/sequence.py#L18-L30)
- Scheduler: [nanovllm/engine/scheduler.py](nanovllm/engine/scheduler.py#L8-L25)
- ModelRunner: [nanovllm/engine/model_runner.py](nanovllm/engine/model_runner.py#L248-L250)
- Context: [nanovllm/utils/context.py](nanovllm/utils/context.py)

### jetengine (参考实现)
- Sequence: [jetengine/engine/sequence.py](jetengine/engine/sequence.py#L24-L78)
- Scheduler: [jetengine/engine/scheduler.py](jetengine/engine/scheduler.py#L30-L250)
- ModelRunner: [jetengine/engine/model_runner.py](jetengine/engine/model_runner.py#L257-L410)
- Context: [jetengine/utils/context.py](jetengine/utils/context.py)
