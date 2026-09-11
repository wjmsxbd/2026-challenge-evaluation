# RFT 成功轨迹采集

采集入口为 `python -m omnigibson.eval.collect_rft`。它复用 2026 向量评测器和
raw-action-chunk websocket 策略服务，按动作步记录原始观测，只保存完整成功的轨迹。
这一步输出 Comet 风格的 NPZ 和三路 MP4，不做 LeRobot 转换或训练。

## 启动

task-0001（`picking_up_trash`）的四实例并行采集测试可以直接从仓库根目录启动：

```bash
bash run_rft_2026_task0001_4instances_1p2x.sh
```

该脚本通过与 `run_eval_2026_persistent_last50_4instances_1p2x.sh` 相同的 pretrain
启动器取得策略代码和运行环境，在目标机器的 GPU 0 和 GPU 1 上同时启动采集。
该测试脚本默认使用
`/mnt/data/ckpt/[b1k]/pi_behavior_b1k_2026/b1k_2026_dlc_2node_bs2048_wo_fast_pi05base_downsample6/60000_inference`，
可通过 `PI05_POLICY_DIR` 覆盖。
每张卡各运行一个策略服务、一个 Isaac 进程及两个环境，合计四个并行环境。
GPU 0 收集训练实例 0–1，GPU 1 收集实例 2–3，保持每个策略服务的 batch size 为 2。
默认使用训练实例 0–3，各尝试一次，时限为平均人工轨迹长度的 1.2 倍（6321 步）。
沿用参考配置的动作修正规则，开启逐步渲染和初始位姿扰动。
日志及原始轨迹保存在 `logs/rft_task0001_test/<时间>/`；每张卡分别写入
`workers/gpu-<编号>/data/task-0001_picking_up_trash/`，避免输出目录锁冲突。
两组完成后，根目录的 `collection_summary.json` 汇总四个实例的统计。

```bash
# 在目标机器检查路径、训练实例、端口及启动参数，不启动策略或仿真
bash run_rft_2026_task0001_4instances_1p2x.sh --dry-run
# 更多尝试；达到每个实例一个成功样本时停止
RFT_NUM_ROLLOUTS=5 bash run_rft_2026_task0001_4instances_1p2x.sh
# 在原来的目录继续，可同时提高 RFT_NUM_ROLLOUTS
bash run_rft_2026_task0001_4instances_1p2x.sh --output-dir /path/to/run --resume
```

`RFT_GPU_IDS='0 1'` 可选择两张卡；`RFT_PORT` 默认 18101，第二个服务使用下一个端口。
每个 Isaac 进程自动分配最多 16 个 CPU 核（可用 `RFT_CPU_THREADS` 覆盖），两张卡使用互不重叠的 CPU 集合。
`RFT_MAX_STEPS` 可设置短测试的绝对步数上限。只有完整成功的样本才保留训练数据，
短测试触发超时时仅保留失败统计。`--dry-run` 可在 GPU 不足的机器上打印计划并报告缺少的卡；
实际运行会在启动任何策略或仿真前检查两张卡是否可见。

使用已经安装好 2026 OmniGibson 的 conda 环境。仓库现有 2026 启动器使用
`/mnt/data_nas/wangjm/miniconda3/envs/behavior_2026`。策略服务继续使用它自己的环境。
先按现有评测方式启动 `serve_pi05_behavior_2026_vector.py`，确保服务端
`--dynamic-batch-max-size` 与采集端 `--num-envs` 一致。

从仓库根目录运行，例如：

```bash
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$PWD/datasets}"
export PYTHONPATH="$PWD/OmniGibson:$PWD/bddl3:$PWD/joylo${PYTHONPATH:+:$PYTHONPATH}"

/mnt/data_nas/wangjm/miniconda3/condabin/conda run --no-capture-output \
  -p /mnt/data_nas/wangjm/miniconda3/envs/behavior_2026 \
  python -m omnigibson.eval.collect_rft \
  --task-name turning_on_radio \
  --mode train --instance-indices 0 1 \
  --host 127.0.0.1 --port 8000 --num-envs 2 \
  --policy-checkpoint /path/to/the/served/checkpoint \
  --num-rollouts 20 --successes-per-instance 5 \
  --output-dir outputs/rft
```

`train` 下的 `--instance-indices` 是实际训练实例 ID，需在本地存在。
`public_test` 下仍按现有评测器的公开实例索引解析，采集记录会保留 split。
用于训练的数据应从训练实例收集，独立保留评测实例。

`--policy-checkpoint` 是用户提供的来源标识，用于记录及续采配置检查；采集端不会据此加载模型，
也不能从该参数验证远端实际权重。使用 task-to-checkpoint 映射服务时，应填写当前任务实际使用的 checkpoint。
同时会保存服务端返回的 metadata。环境种子与位姿扰动可按样本复现，策略服务的随机状态仍由服务端管理。

## 采样与动作设置

- `--num-rollouts`：每个实例最多尝试的次数，默认 10。
- `--successes-per-instance`：每个实例的成功目标数，默认 0 表示用完尝试次数。
  达到目标后跳过该实例，未达到目标也会在尝试预算耗尽时结束。
- `--sample-start`：起始采样编号，默认 0；编号范围为
  `[sample_start, sample_start + num_rollouts)`。成功配额按这个范围统计。
- `--seed`：用于派生每个任务、实例、样本的环境种子，不依赖它被分到哪个向量槽位。
- `--perturb-pose` / `--no-perturb-pose`：默认开启初始位姿扰动。
  `--perturb-translation` 默认 0.15 米，`--perturb-yaw-degrees` 默认 15 度。
  每次从原始实例重新加载并扰动，扰动写入 reset 使用的元数据，不跨样本累加。
- 默认沿用 26 个预测动作压缩为 20 个执行步、保留 4 个动作的设置。
  `--actions-to-execute`、`--execute-in-n-steps`、`--no-compression`、
  `--no-action-chunk-maintenance` 等参数与向量评测器相同，应与策略的部署设置一致。
- 默认关闭额外动作修正规则。显式开启 `--apply-eval-tricks` 时，仍需按现有评测方式配置
  `PI05_CORRECTION_RULES_PATH`。
- 采集始终对每个动作步进行渲染，不使用 `PI05_SKIP_ACTION_CHUNK_RENDERING` 加速。
  相机分辨率沿用传入的 robot config，写入实际分辨率，不另行缩放。

## 输出与对齐约定

```text
outputs/rft/task-0000_<task>/
  collection.json
  collection_summary.json
  metrics/
  rollouts/instance-0000/sample-000000/
    episode.json
    state_action.npz
    head.mp4
    left_wrist.mp4
    right_wrist.mp4
```

实际 task 编号由 2026 元数据决定。失败样本的目录只保留 `episode.json`。
`--write-video` 可额外写入现有拼接评测视频，它不控制三路训练视频的记录。

NPZ 保持 `np.load(path, allow_pickle=True)["arr_0"].item()` 的 Comet 读取方式：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `state` | `[T, 61]`，float32 | 每次执行动作前的 2026 原生 proprio，底盘速度为机器人局部坐标系 |
| `action` | `[T, 23]`，float32 | 修正、压缩后，实际传入环境的控制指令 |
| `timestamp` | `[T]` | 相对采样时间，按实际动作频率生成 |
| `base_qpos`、`base_qvel` | 各 `[T, 3]` | 与 state 同步的基座虚拟关节位置 `[x,y,yaw]` 和 canonical 速度，供之后核对坐标系 |
| `reward`、`terminated`、`truncated` | 各 `[T]` | 执行该动作后返回的结果 |

三路视频各 T 帧，第 t 帧对应 `state[t]`，监督目标是 `action[t]`。
保留 reset 后的初始观测，不固定删除前 20 帧；最后一次动作之后的终止观测不额外写入视频。
记录用的原始状态与提供给策略的状态分别维护，策略端选择 legacy 状态布局或 absolute
底盘速度不会改变存档的原生状态语义。

`episode.json` 记录成功标志、帧数、实际相机尺寸、任务/实例/样本 ID、坐标系、扰动参数和来源信息。
只有环境的完整成功标签为真时才保存 NPZ 和视频，Q-score 的部分进展不会被当作成功。

## 续采和异常

在相同命令后加 `--resume` 可跳过已完成的成功和失败样本。增大 `--num-rollouts` 可扩展采样范围。
checkpoint 标识、机器人配置、split、实例列表、动作设置及扰动配置需要与该任务的 `collection.json` 一致；
改变这些内容时使用新的输出目录。
续采会检查样本 ID、成功样本的 NPZ 形状及视频帧数/帧率；发现输出不完整时会报错。

视频先写入不可见的临时样本目录，编码器全部关闭且 NPZ 写入完成后才发布最终目录。
异常退出会清理当前进程未发布的样本；强制终止遗留的隐藏临时目录不会被视为成功样本。
同一任务目录有进程锁，多个进程采集同一任务时应使用不同输出目录。

当前入口每个进程处理一个任务，可使用多个向量槽位。现有多节点常驻评测调度器不在此入口中启动。
`collection_summary.json` 包括实际成功数、尝试数及成功目标是否达到。
