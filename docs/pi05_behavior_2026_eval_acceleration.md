# PI0.5 的 2026 BEHAVIOR 加速评测

本文档说明 `speedup_eval` 分支上的 2026 Challenge 加速评测方案、冠军 PI0.5 模型的环境侧适配，以及常用启动方式。

推荐入口：

```bash
bash run_pi05_behavior_2026_eval_chunk_balance.sh
```

普通动态队列入口是 `run_pi05_behavior_2026_eval_chunk.sh`。两者使用相同的 simulator、vector evaluator、policy server 和输出校验逻辑；`balance` 版本额外根据各任务的人类演示时长预估计算量，预先将任务均衡分配到 GPU worker，通常更适合多卡长时间运行。

## 评测协议

启动器遵循 2026 Challenge 的公开评测协议：

- 支持 100 个任务，task ID 为 `0-99`。
- 排行榜公开评测使用每个任务的 public instance index `0-9`。
- 每个 instance 运行 1 次 rollout。
- 不设置 `EVAL_MAX_STEPS` 时，使用该任务平均人类演示长度的 `1.5x` 作为 timeout。
- `submission` profile 为每条 rollout 写出 metrics JSON 和 MP4。
- 官方仿真频率保持为 physics/render/action = `120/30/30 Hz`。

默认 100-task PI0.5 配置来自：

```text
PI0.5 repo:       /mnt/data_nas/wangjm/unirobot/behavior-1k-solution
PI0.5 branch:     champion_2026
BEHAVIOR branch:  speedup_eval
policy config:    pi_behavior_b1k_2026
```

启动器会检查 checkpoint、norm stats、100-task 配置和输出完整性，防止误用 50-task checkpoint 或不兼容的 proprioception schema。

## 加速架构

默认每张 GPU 的拓扑如下：

```text
每张 GPU
├── 1 个持久 PI0.5 server
└── 1 个 Isaac Sim 进程
    ├── vector env slot 0
    └── vector env slot 1
```

主要加速点包括：

1. 每张 GPU 只加载一次 checkpoint，server 在该 worker 的任务队列耗尽前保持运行。
2. 一个 Isaac Sim 进程同时承载两个 scene/env，两个 slot 共享 simulator tick。
3. 两个活跃 env 的 observation 合并为一次 `observation_batch`，server 一次执行 batch-2 推理。
4. 如果其中一个 instance 提前结束，剩余 slot 自动退化为 batch-1，直到该组完成。
5. task-relevant partial scene load 减少无关房间和对象的加载开销。
6. policy camera 在创建 render product 前固定为 `224x224`，避免运行中重建相机。
7. 关闭无用 viewer camera，但保留每个 action 后的正式 policy observation、render、reward、termination 和 metrics 更新。
8. evaluator/server 设置独立 CPU affinity，并限制 BLAS、Torch 和 OpenCV 线程。
9. 输出先写入 attempt 目录，通过完整性校验后再提升为正式结果；失败任务可按配置整体重试。

没有降低物理频率，也没有跳过中间 render、`get_obs()` 或 metrics，因此这些加速不会主动改变官方 rollout 的时间和观测语义。

### 两个调度脚本的区别

`run_pi05_behavior_2026_eval_chunk.sh` 使用共享动态任务队列：空闲 worker 领取下一个 task，结构简单。

`run_pi05_behavior_2026_eval_chunk_balance.sh` 使用 LPT 风格的预均衡调度：根据各任务 timeout 和 instance group 数估算计算量，将长任务分散到不同 GPU，再让每个 worker 按自己的队列运行。多卡运行后 50 个任务时推荐使用这个版本，可以减少最后等待少量长任务的尾部时间。

## Batch-2 推理

正常双路运行时的数据流为：

```text
env slot 0 observation ─┐
                       ├─> server observation_batch[2] -> action_chunk[2]
env slot 1 observation ─┘
                                         │
                       同步推进两个 vector env slot
```

默认参数为：

```text
VECTOR_ENVS_PER_PROCESS=2
PI05_DYNAMIC_BATCH_MAX_SIZE=2
PI05_DYNAMIC_BATCH_GRANULARITY=1
```

`granularity=1` 是为了允许某一路提前结束后继续发送 batch-1。日志中正常会看到：

```text
policy_batch=2
request_batch_max=2
```

## Reset 和随机种子

环境随机种子固定为 `0`。两个 vector slot 都使用同一个 seed：

```text
slot 0 seed = 0
slot 1 seed = 0
```

每处理完一组并行 instance、开始下一组 reset/load 时，两个 slot 都会重新播种为 `0`。日志会记录：

```text
Resetting vector group RNGs: instances=[...] slot_seeds=[0, 0]
```

启动脚本同时设置 `PYTHONHASHSEED=0`，evaluator 会固定 Python、NumPy、Torch CPU/CUDA 和 Warp RNG。

server 端的 JAX policy 默认从 `jax.random.key(0)` 启动，之后随每次 inference request 持续 split/推进；环境 group reset 不会重置 server RNG。因此完整复现还要求 server 生命周期和请求顺序一致。

## Base velocity 和 action frame

`--base-velocity-frame` 只控制输入 policy 的 `observation.state` 中 base qvel 的坐标系：

- `absolute`：使用旧训练数据的 raw/canonical virtual-joint base qvel；这是当前启动器默认值。
- `relative`：使用新版 robot-local base qvel。

policy 输出的 base action 始终按 R1Pro controller 的 robot-local `[vx, vy, wz]` 约定直接执行，不再由 `base_velocity_frame` 触发 yaw 旋转。

对于当前旧版 2026 checkpoint，推荐显式使用：

```bash
--base-velocity-frame absolute
```

## 冠军方案的环境侧适配

这套 evaluator 不只是通用 vector acceleration，还适配了 `champion_2026` PI0.5 方案。原先和冠军 server/policy 耦合的部分 rollout 后处理已经迁移到 env/evaluator 侧，位于：

```text
OmniGibson/omnigibson/eval/utils/pi05_action_chunk.py
```

其中包括：

- subtask stage voting；
- 冠军模型的 task/stage-specific correction rules；
- gripper variation 检查；
- action chunk 压缩和底盘累计量补偿；
- 保留 chunk 尾部 action，作为下一次 inference 的 inpainting prefix。

correction rules 实际从 PI0.5 仓库加载：

```text
${PI05_REPO}/src/b1k/shared/correction_rules.py
```

### 接入其他模型时必须注意

correction rules 是冠军 PI0.5 checkpoint 的专用启发式规则，不属于 BEHAVIOR 官方评测协议，也不是通用模型后处理。接入其他模型时不能默认套用这些规则，否则可能按照冠军模型的 task ID、stage 和 action 分布错误修改其他模型的输出。

复用当前 evaluator 接入其他模型时，至少需要关闭冠军 correction rule 和 gripper-variation gate：

```bash
PI05_APPLY_EVAL_TRICKS=false \
bash run_pi05_behavior_2026_eval_chunk_balance.sh
```

该开关只会关闭 correction rules 和与其绑定的 gripper variation 判断。stage voting、chunk 压缩和 inpainting carry 仍属于当前 PI0.5 action-chunk pipeline。如果新模型不采用相同的 30-step chunk、stage logits 或 inpainting 协议，还需要替换或绕过 `B1KActionChunkPostprocessor`，不能把当前 server/evaluator 当作完全通用的即插即用接口。

换句话说，可以复用的通用加速部分是：

- persistent server；
- 双 vector env；
- batch-2 observation/inference；
- scene/reset/callback 隔离；
- GPU/CPU 调度；
- 输出校验与失败重试。

需要按模型重新确认的部分是：

- observation/action schema；
- normalization；
- stage logits；
- action chunk 后处理；
- correction rules；
- inpainting prefix。

## 启动方式

### 推荐：后 50 个任务，每个任务前 6 个 instance

下面的命令运行 task `50-99`，每个 task 使用 public instance index `0-5`，共 `50 x 6 = 300` 条 rollout，并生成 submission 所需视频：

```bash
TASK_IDS="$(seq 50 99)" \
TASK_LIMIT=50 \
EVAL_INSTANCE_INDICES='0 1 2 3 4 5' \
EVAL_PROFILE=submission \
EVAL_FAIL_FAST=false \
bash run_pi05_behavior_2026_eval_chunk_balance.sh \
  --base-velocity-frame absolute
```

### 完整 100-task submission

```bash
EVAL_PROFILE=submission \
bash run_pi05_behavior_2026_eval_chunk_balance.sh \
  --base-velocity-frame absolute
```

默认运行 100 个 task、每个 task 的 public index `0-9`，合计 1,000 条 rollout。

### 不生成视频的吞吐测试

```bash
EVAL_PROFILE=throughput \
bash run_pi05_behavior_2026_eval_chunk_balance.sh \
  --base-velocity-frame absolute
```

`throughput` 和 `submission` 的 simulator、observation、policy action 与 metrics 路径相同；前者不编码 MP4，不能直接作为完整 submission。

### 单卡 smoke test

```bash
GPU_IDS=0 \
NUM_GPUS=1 \
TASK_IDS=50 \
TASK_LIMIT=1 \
EVAL_INSTANCE_INDICES='0 1' \
EVAL_MAX_STEPS=100 \
EVAL_PROFILE=throughput \
bash run_pi05_behavior_2026_eval_chunk_balance.sh \
  --base-velocity-frame absolute
```

### 只检查配置和任务队列

```bash
bash run_pi05_behavior_2026_eval_chunk_balance.sh \
  --base-velocity-frame absolute \
  --dry-run
```

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `TASK_IDS` | `0-99` | 空格或逗号分隔的 task ID |
| `TASK_LIMIT` | `100` | 从 `TASK_IDS` 中取前多少个 task |
| `EVAL_INSTANCE_INDICES` | `0 ... 9` | 每个 task 的 public instance index |
| `EVAL_PROFILE` | `throughput` | `throughput` 或 `submission` |
| `EVAL_FAIL_FAST` | `true` | 一个 task 失败后是否停止整个调度器 |
| `EVAL_MAX_TASK_ATTEMPTS` | `2` | task 级最大尝试次数 |
| `NUM_GPUS` | `8` | 启动的 GPU worker 数量 |
| `GPU_IDS` | `0 ... 7` | 使用的 GPU ID |
| `PI05_BASE_VELOCITY_FRAME` | `absolute` | policy observation 的 base qvel frame |
| `PI05_APPLY_EVAL_TRICKS` | `true` | 是否启用冠军 correction rules 和 gripper gate |
| `EVAL_SEED` | `0` | 兼容参数；当前两个 env slot 固定使用 seed 0 |

## 输出与完整性检查

主要输出目录：

```text
rollout JSON/MP4: logs/pi05_behavior_2026_outputs/<timestamp>/
scheduler 日志:   logs/pi05_behavior_2026_<timestamp>/
```

启动器会：

- 为每个 task 检查 `evaluation_complete.json`；
- 检查每个选中 instance 是否有且仅有一个 JSON；
- 在 `submission` 模式检查对应 MP4；
- 扫描 simulator/server fatal、segfault、Vulkan 和 contact-view 错误；
- 使用 immutable run manifest 校验预期 task 和 instance 集合；
- 只有完整 attempt 才会提升为正式输出；
- 全部完成后写出 `run_complete.json`。

预期输出数为：

```text
选中的 task 数 x 每个 task 的 instance 数
```

Isaac Sim 的 RTX camera 还要求容器提供 Vulkan/graphics capability。NVIDIA 容器中的 `NVIDIA_DRIVER_CAPABILITIES` 通常需要包含 `graphics` 或 `all`。
