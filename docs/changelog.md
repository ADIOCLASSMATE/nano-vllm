# CHANGELOG

## [0.0.2] 2025.12.27

```bash
LLMEngine.update_model_param(param_dict)
    │
    ├─► model_runner.update_model_param_local(param_dict)  # Rank 0 本地更新
    │       └─► 直接 copy_ 权重到 GPU
    │
    └─► model_runner.call("sync_model_param", param_names)  # IPC 只传参数名
            │
            ├─► write_shm([sync_model_param, param_names])  # 只有几KB
            │
            └─► 所有 ranks 执行 sync_model_param()
                    └─► dist.broadcast(param.data, src=0)  # NCCL 高效同步
```

解决问题：
1. 共享内存只有 1MB，无法传输模型权重
2. nanovllm 模型已按 tensor parallel 分片，但传入的是完整权重


## [0.0.1] 2025.12.21

1. 学习如何调用模型 [llm_engine](../nanovllm/engine/llm_engine.py)
2. 如何实际跑一个模型 [model_runner](../nanovllm/engine/model_runner.py)
3. 学习如何内存管理 [block_manager](../nanovllm/engine/block_manager.py)
4. 学习过程中辅助看一下是如何调度的 [sequence](../nanovllm/engine/sequence.py) 和 [scheduler](../nanovllm/engine/scheduler.py)

- 开始认真读这个项目的代码
