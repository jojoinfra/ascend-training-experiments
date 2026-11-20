# Qwen3.8-27B / 910B4 单机 8 卡

默认 TP2 × PP4、GBS=32、MBS=1、全层重算，使用共用的 [模型入口](../pretrain_qwen.py) 和 [环境配置](../runtime/env.sh)。

## 运行

先按[模型环境准备](../README.md#准备环境)创建容器，在容器中执行：

```bash
cd /work/910b4-1x8
RUN_NAME=single-test STEPS=8 bash train.sh
```

结果写入宿主机 `models/qwen3.8-27b/910b4-1x8/results/single-test/`，包括 `node-0.log` 和自动生成的 `summary.json`。每次使用新的 `RUN_NAME`。可在命令前覆盖 `TP`、`PP`、`GBS`、`MBS`、重算等策略参数；额外 Megatron 开关追加到 `train.sh` 后。

## 已完成的性能基线

2026-11-20 实测完整 64 层文本主干与独立 LM head，26,895,998,464 参数、4,096 token 合成序列，BF16 参数、FP32 梯度和 CPU AdamW，CPU 卸载比例 1.0。每 rank 24 个 CPU 线程，按 [runtime/env.sh](../runtime/env.sh) 的 socket/NUMA 配置绑核。

完整运行 8 步，统计第 3–7 步；8 个 rank 全部完成，launcher 退出码为 0，无跳步或 NaN。这是随机初始化、合成数据的短程吞吐基线，不是收敛或长时间稳定性验证。详细[统计口径](../README.md#统计口径)与[双机基线](../910b4-2x8/README.md)可供核对；单双机 GBS 不同，吞吐比不能作为固定全局批量的扩展效率。

| 策略 | tokens/s | 相对收益 | 启动命令 |
|---|---:|---|---|
| `single-full` | 2,745.2 | 基线 | `STEPS=8 bash train.sh` |
