# Qwen3.8-27B / 910B4 双机 16 卡

默认 TP4 × PP4、GBS=64、MBS=1、全层重算，使用共用的 [模型入口](../pretrain_qwen.py) 和 [环境配置](../runtime/env.sh)。所有策略都通过本目录的 [train.sh](train.sh) 启动。

## 运行

先按[模型环境准备](../README.md#准备环境)在两台宿主机创建容器。在两台容器中执行 `cd /work/910b4-2x8`，分别设置本节点网络信息，然后执行同一条训练命令。第一台为 `NODE_RANK=0`，第二台为 `NODE_RANK=1`。两端的 `MASTER_ADDR`、`MASTER_PORT` 和策略参数相同。以下 IP 和网卡按实际环境填写。

节点 0 的容器执行以下设置。

```bash
export NODE_RANK=0 MASTER_ADDR=192.168.0.80 MASTER_PORT=29871
export HCCL_IF_IP=192.168.0.80 GLOO_SOCKET_IFNAME=enp67s0f5
```

节点 1 的容器执行以下设置。

```bash
export NODE_RANK=1 MASTER_ADDR=192.168.0.80 MASTER_PORT=29871
export HCCL_IF_IP=192.168.0.98 GLOO_SOCKET_IFNAME=enp67s0f5
```

两端分别执行下列命令，即 `split` 组合。

```bash
RUN_NAME=split STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 \
MC2=1 QWEN_FUSED_SWIGLU=1 \
PIPELINE_LAYERS=14,18,18,14 \
bash train.sh
```

各宿主机部署目录 `models/qwen3.8-27b/910b4-2x8/` 下的 `results/split/` 保存本机节点日志。训练完成后，在节点 0 宿主机的部署目录收集第二台的结果并核验。路径中的 `user@node1` 和宿主机部署目录按实际填写。

```bash
scp 'user@node1:/path/to/models/qwen3.8-27b/910b4-2x8/results/split/node-1.log' results/split/
python3 ../measure.py results/split --world 16 > results/split/summary.json
```

完整日志核验与计时定义见[统计口径](../README.md#统计口径)。必须保留 `--log-interval 1`；两端使用相同且新的 `RUN_NAME`。

## 标准 pretrain() 性能基线

2026-11-20 在两台 8×32 GiB Ascend 910B4 上复测，使用共用模型入口和锁定依赖。两端训练源码的 SHA-256、依赖 Git commit/补丁树、模型结构和 NUMA 映射均已核对。环境为 Python 3.11.15、PyTorch 2.10.0+cpu、torch-npu 2.10.0.post2、FLA-NPU 26.6.0、triton-ascend 3.2.1，驱动 25.5.1、CANN 9.0.0。

全部配置均使用完整 64 层文本主干与独立 LM head，共 26,895,998,464 参数、4,096 token 序列；不包含视觉塔和 MTP。BF16 参数、FP32 梯度及 CPU AdamW，默认卸载比例 1.0、MBS=1、seed=1234。GBS=64；与 [单机参考](../910b4-1x8/README.md) 的 GBS=32 不同，吞吐比不能当作固定全局批量的扩展效率。

每项设置 8 步，成功项统一统计第 3–7 步，表中为整个作业的 tokens/s。只有所有节点的 launcher 正常退出、全部 rank 完成、8 步 loss/梯度范数有限且无跳步/NaN，才记录吞吐。初始化、前两步预热和末步不计入。成功项均为一次短程测量，不是长时间稳定性或收敛验证；小幅差异不能据此认定为稳定加速。

完整模型初始化时发现，上游 DDP 重建参数存储后保留的空闲 NPU 缓存可能挤占 CANN workspace。本次在数据回调中、DDP/优化器准备完成后增加一次 `torch.npu.empty_cache()`，再执行全部基线。训练循环与优化器仍由上游 `pretrain()` 管理，没有逐步清理缓存。已有容器可通过 `FRAMEWORK_ROOT` 指向独立依赖目录，默认仍为 `/work`。

下面的命令均在本部署目录执行；双机在两端设置好网络变量后执行同一行。每次在参与节点上设置相同且新的 `RUN_NAME`，不要把某一行的策略变量永久 export 到后续实验。未写出的策略采用部署脚本默认值。

旧手写循环的性能表已由本轮结果替换。`HOST_VERIFY`、`PIN_CPU_MASTER`、`PROFILE_STAGES` 和 `OBSERVE_*` 不再适用；脚本会拒绝这些变量。CPU 参数/梯度页锁定由上游卸载实现管理，旧 `pin-master` 没有独立等价策略。

本轮共测试 **26 个双机配置：25 个完成 8 步，1 个 OOM**。建议从上面的 `split` 命令开始：**8,522.5 tokens/s**，相对同为 TP4×PP4 的全重算基线 `5,997.8` 为 **+42.09%**。

### 策略为何有效，哪些没有确认收益

- **减少重算**：`block12`、`block8` 分别只重算每段前 12、8 层，用更多激活显存换取更少的重复前向计算；相对全重算为 +5.80%、+11.69%。`stage` 再将重算深度改为 `8/0/0/0`，相对 `block8` 为 +11.16%；本轮没有叠加旧循环的 CPU 梯度校验迁移。
- **融合算子与 TP 通信**：SwiGLU 融合减少独立算子和中间结果，MC² 融合 TP 通信与矩阵乘。单独开启时相对 `stage` 分别为 +3.05%、+5.56%；同时开启的 `uniform` 为 +8.86%，不能将两个单项百分比直接相加。
- **调整流水线分层**：`split` 将 `16/16/16/16` 改为 `14/18/18/14`，给还负责 embedding/LM head 的首尾段减少 Transformer 层，保持全局层序和参数总数不变；相对 `uniform` 为 +5.14%。继续减少首段重算的 `front6`、`split-front4`、`split-front0` 均未显示出明确收益。
- **CE 与 FP32 累加**：按 2,048 token 分块限制 LM head 单次投影/softmax 的临时张量，但 `uniform-ce`、`split-ce` 相对各自不分块对照为 -0.13%、-0.18%，没有确认吞吐加速。MC² FP32 梯度累加融合相对 `split-ce` 为 +0.19%，也小于该组步间波动；推荐命令不启用这两项。
- **并行布局与微批**：全重算下 TP4×PP2×DP2 为 6,340.8 tokens/s，比 TP4×PP4 为 +5.72%。标准 distributed optimizer 会在 DP 组内分片优化器状态，因此须以新入口实测排序。该布局改为 `block24` 再增加 +5.44%；把 MBS 改为 2 则在 DP2、DP1 两个对照中分别为 -0.72%、-1.24%，本轮均无加速。
- **卸载与通信选项**：`OFFLOAD_FRACTION=0.85` 将部分优化器状态留在 NPU，改变 CPU 计算/传输与显存之间的取舍，相对 `block8` 为 +0.41%。rank 映射、梯度归约重叠分别相对 `dp2-block` 为 -0.42%、-0.27%；按通信组自适应设置 HCCL 缓冲、取消 P2P 批量发收分别相对 `stage` 为 -0.04%、+0.26%。这些小幅变化没有证明稳定收益。

**失败配置 `block4`**：首个迭代期间，首段 rank 0–3 在 HCCL AllReduce 报显存不足，申请 420,478,976 字节（约 401 MiB）失败；两端 launcher 均以 1 退出。没有完成可用计时窗口，因此只记录 OOM，不报告吞吐或加速比。更少重算不一定适合 32 GiB 卡。

### 双机 16 卡：重算与融合

| 策略 | tokens/s | 相对收益 | 启动命令 |
|---|---:|---|---|
| `full` | 5,997.8 | 基线 | `STEPS=8 bash train.sh` |
| `block12` | 6,345.7 | +5.80% vs `full` | `STEPS=8 RECOMPUTE=block RECOMPUTE_LAYERS=12 bash train.sh` |
| `block8` | 6,698.9 | +11.69% vs `full` | `STEPS=8 RECOMPUTE=block RECOMPUTE_LAYERS=8 bash train.sh` |
| `stage` | 7,446.6 | +11.16% vs `block8` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 bash train.sh` |
| `swiglu` | 7,673.7 | +3.05% vs `stage` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 QWEN_FUSED_SWIGLU=1 bash train.sh` |
| `mc2` | 7,860.4 | +5.56% vs `stage` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 MC2=1 bash train.sh` |
| `uniform` | 8,106.0 | +8.86% vs `stage` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 bash train.sh` |
| `split` | 8,522.5 | +5.14% vs `uniform` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 PIPELINE_LAYERS=14,18,18,14 bash train.sh` |
| `uniform-ce` | 8,095.4 | -0.13% vs `uniform` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 CE_CHUNK_TOKENS=2048 bash train.sh` |
| `split-ce` | 8,507.0 | -0.18% vs `split` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 PIPELINE_LAYERS=14,18,18,14 CE_CHUNK_TOKENS=2048 bash train.sh` |
| `fp32-accum` | 8,523.1 | +0.19% vs `split-ce` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 PIPELINE_LAYERS=14,18,18,14 CE_CHUNK_TOKENS=2048 MC2_FP32_ACCUM=1 bash train.sh` |
| `front6` | 8,119.3 | +0.16% vs `uniform` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=6,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 bash train.sh` |
| `split-front4` | 8,502.4 | -0.24% vs `split` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=4,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 PIPELINE_LAYERS=14,18,18,14 bash train.sh` |
| `split-front0` | 8,519.7 | -0.03% vs `split` | `STEPS=8 RECOMPUTE=none STAGE_RECOMPUTE=0,0,0,0 MC2=1 QWEN_FUSED_SWIGLU=1 PIPELINE_LAYERS=14,18,18,14 bash train.sh` |

### 双机 16 卡：并行、卸载与通信

| 策略 | tokens/s | 相对收益 | 启动命令 |
|---|---:|---|---|
| `tp2-pp8` | 5,187.8 | -13.50% vs `full` | `STEPS=8 TP=2 PP=8 bash train.sh` |
| `tp2-pp4-dp2` | 5,728.0 | +10.41% vs `tp2-pp8` | `STEPS=8 TP=2 bash train.sh` |
| `tp4-pp2-dp2` | 6,340.8 | +22.22% vs `tp2-pp8` | `STEPS=8 PP=2 bash train.sh` |
| `dp2-block` | 6,685.5 | +5.44% vs `tp4-pp2-dp2` | `STEPS=8 PP=2 RECOMPUTE=block RECOMPUTE_LAYERS=24 bash train.sh` |
| `dp2-mbs2` | 6,295.2 | -0.72% vs `tp4-pp2-dp2` | `STEPS=8 PP=2 MBS=2 bash train.sh` |
| `mbs2` | 5,923.6 | -1.24% vs `full` | `STEPS=8 MBS=2 bash train.sh` |
| `rank-map` | 6,657.7 | -0.42% vs `dp2-block` | `STEPS=8 PP=2 RECOMPUTE=block RECOMPUTE_LAYERS=24 bash train.sh --use-tp-pp-dp-mapping` |
| `grad-overlap` | 6,667.5 | -0.27% vs `dp2-block` | `STEPS=8 PP=2 RECOMPUTE=block RECOMPUTE_LAYERS=24 bash train.sh --overlap-grad-reduce` |
| `offload85` | 6,726.2 | +0.41% vs `block8` | `STEPS=8 RECOMPUTE=block RECOMPUTE_LAYERS=8 OFFLOAD_FRACTION=0.85 bash train.sh` |
| `block4` | OOM（无吞吐） | — | `STEPS=8 RECOMPUTE=block RECOMPUTE_LAYERS=4 bash train.sh` |
| `adaptive-hccl` | 7,443.3 | -0.04% vs `stage` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 bash train.sh --hccl-group-buffer-adaptive` |
| `unbatch-p2p` | 7,465.8 | +0.26% vs `stage` | `STEPS=8 RECOMPUTE=block STAGE_RECOMPUTE=8,0,0,0 QWEN_UNBATCH_P2P=1 bash train.sh` |
