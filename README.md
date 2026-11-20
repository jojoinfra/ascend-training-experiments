# 昇腾训练性能基线

按 `models/<具体模型>/<部署形态>/` 组织部署入口、运行说明和策略基线。模型实现与固定版本的框架依赖在模型层共用；训练循环复用 Megatron 标准 `pretrain()`。

```text
models/
└── qwen3.8-27b/
    ├── 910b4-1x8/
    │   ├── train.sh
    │   └── README.md
    ├── 910b4-2x8/
    │   ├── train.sh
    │   └── README.md
    ├── pretrain_qwen.py
    ├── dependencies.lock.json
    └── runtime/
```

| 模型 | 部署形态 | 工作负载 | 配置与运行 |
|---|---|---|---|
| Qwen3.8-27B | `910b4-1x8`：单机 8 卡 | BF16 文本训练，4K 合成序列 | [单机运行与基线](models/qwen3.8-27b/910b4-1x8/README.md) |
| Qwen3.8-27B | `910b4-2x8`：双机 16 卡 | BF16 文本训练，4K 合成序列 | [双机运行与策略收益](models/qwen3.8-27b/910b4-2x8/README.md) |

先按[模型环境准备](models/qwen3.8-27b/README.md#准备环境)创建训练容器，再进入对应部署目录运行 `train.sh`。结果保存在该部署目录的 `results/<RUN_NAME>/`。
