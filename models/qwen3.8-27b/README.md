# Qwen3.8-27B

模型入口 [pretrain_qwen.py](pretrain_qwen.py) 从固定版本的 MindSpeed-MM 构建 64 层文本主干和独立 LM head，随机初始化 26,895,998,464 个可训练参数，在 BF16、4,096 token 合成序列上执行前反向和 CPU AdamW 更新。

## 部署形态

| 部署目录 | 默认配置 | 运行与基线 |
|---|---|---|
| `910b4-1x8/` | 单机 8 卡，TP2 × PP4，GBS=32 | [单机说明](910b4-1x8/README.md) · [train.sh](910b4-1x8/train.sh) |
| `910b4-2x8/` | 双机 16 卡，TP4 × PP4，GBS=64 | [双机说明与策略收益](910b4-2x8/README.md) · [train.sh](910b4-2x8/train.sh) |

每个部署目录包含启动脚本、运行说明与性能基线，运行结果保存在该目录的 `results/<RUN_NAME>/`。部署脚本默认 5 步，复现性能表时显式使用 `STEPS=8`。

## 共用代码与配置

| 文件 | 内容 |
|---|---|
| [pretrain_qwen.py](pretrain_qwen.py) | 模型构建、合成数据与前向回调 |
| [runtime/launch.sh](runtime/launch.sh) | 共用 Megatron 参数与 torchrun 启动命令，由部署脚本调用 |
| [runtime/env.sh](runtime/env.sh) | Ascend 环境变量、CPU 线程数、NUMA 绑定 |
| [model-reference/config.json](model-reference/config.json) | 模型结构参数 |
| [dependencies.lock.json](dependencies.lock.json) | 外部框架的固定 Git commit 和补丁 |
| [measure.py](measure.py) | 从完整训练日志计算吞吐 |

入口采用 Megatron 标准 `pretrain(data_provider, model_provider, ModelType.encoder_or_decoder, forward_step)`：

- Megatron 负责初始化、BF16/DDP 包装、流水线调度、梯度归约、优化器、学习率调度和训练日志。
- 本仓提供 Qwen 模型配置和合成数据回调；交叉熵复用 `LanguageModule.compute_language_model_loss`，loss 归约和异常检查复用上游 `pretrain_gpt.loss_func`。分块 CE 保留为可选模型优化。
- CPU 卸载通过 `--use-distributed-optimizer --use-precision-aware-optimizer --optimizer-cpu-offload` 启用，主权重和 AdamW 状态由上游管理。`OFFLOAD_FRACTION` 控制卸载比例，默认 1.0；学习率固定 1e-4，`--clip-grad 0`。

此样例使用随机初始化和合成数据，不支持从 checkpoint 恢复；不提供验证集。启动脚本读取 `/model-reference/config.json`（默认挂载仓内模型结构），启动参数必须与其一致。

## 准备环境

每台 Linux arm64 宿主机需要 8 张 32 GiB 910B4、Docker、Git、Python 3 和可用的昇腾驱动。实测环境为驱动 25.5.1、CANN 9.0.0、Python 3.11.15、PyTorch 2.10.0+cpu、torch-npu 2.10.0.post2、FLA-NPU 26.6.0、triton-ascend 3.2.1，每节点 192 个 CPU 线程和约 1.5 TiB 主存。双机需要打通 HCCL 和控制网。

将同一版本代码放在各宿主机，进入 `models/qwen3.8-27b/`。取得训练镜像后执行以下命令。默认镜像为 `qwen38-training:910b4-20261114`，可用 `TRAINING_IMAGE` 指定已准备好的镜像。

```bash
docker load -i /path/to/training-image.tar
bash runtime/create-container.sh
docker exec -it qwen38-benchmark bash
cd /work
```

创建脚本把整个模型目录挂载到 `/work`，按锁文件检出框架到外部缓存并只读挂载。随后进入 `/work/910b4-1x8` 或 `/work/910b4-2x8`，按对应部署说明运行 `train.sh`。

MindSpeed 的 `mindspeed-external-data.patch` 仅让 external 合成数据跳过不使用的 C++ 数据集索引编译，避免初始化时写入只读源码；它不修改训练循环或优化器。

`runtime/env.sh` 默认使用原机的 socket 绑核和每 rank 24 个 CPU 线程。换机器时核对卡 0–7 的 PCI NUMA 节点与 CPU 编号，修改 `NPU_NUMA_NODES`；原映射为 `6,0,6,0,4,2,4,2`，绑核算法使用每 NUMA 节点连续 24 个 CPU 编号。可用 `NUMA_BIND=none` 先运行未绑定的配置，并记录这一条件差异。

两种部署共用模型目录下的 `.training.lock`，同一份模型目录不能同时启动两种部署。

## 统计口径

`measure.py` 检查所有 rank 的完成标记、相同的模型/并行配置、连续的训练步、有限 loss 和零跳步，再使用上游 `elapsed time per iteration (ms)` 计算总 token 数 / 累计耗时。必须保留 `--log-interval 1`。默认丢弃前两步和末步：5 步取 3–4，8 步取 3–7；其他窗口用 `--last-timed-step` 指定，单机脚本也可设置 `LAST_TIMED_STEP`。

计时包含前反向、通信、优化器更新以及上游训练循环的其他开销。新统计不再输出旧循环的逐参数梯度覆盖、首尾步抽样更新或峰值显存验证字段。异常检查使用 Megatron 的默认 loss / 梯度检查；需要诊断时使用上游 profiler 参数。

<details>
<summary>自备镜像与离线依赖</summary>

原镜像为 Linux arm64，ID 为 `sha256:1df4dde4578af3816e4b21b6eb295eb6ed791e9e060dd4b0ca26cca93c7a290b`。自备基础镜像需要 CANN 9.0.0、Python 3.11、CPU 版 PyTorch 2.10.0、Ascend 头文件和 C++ 编译工具。FP32 累加融合还用到 NNAL/ATB。创建容器后，在宿主机补充 Python 依赖与 FLA-NPU。

```bash
docker exec qwen38-benchmark bash /work/runtime/setup-env.sh
docker exec qwen38-benchmark bash /work/runtime/build-fla.sh
```

联网机器可在模型目录预取框架，再把缓存传到训练机。

```bash
bundle=$(python3 ../../tools/prepare_dependencies.py dependencies.lock.json) &&
tar -czf /tmp/qwen-deps.tar.gz -C "$(dirname "$bundle")" "$(basename "$bundle")"
```

训练机在模型目录导入缓存后，继续创建容器。

```bash
export ASCEND_BENCH_DEPS_CACHE="$HOME/.cache/ascend-deps"
mkdir -p "$ASCEND_BENCH_DEPS_CACHE"
tar -xzf /tmp/qwen-deps.tar.gz -C "$ASCEND_BENCH_DEPS_CACHE"
GIT_ALLOW_PROTOCOL=none python3 ../../tools/prepare_dependencies.py dependencies.lock.json
```

</details>
