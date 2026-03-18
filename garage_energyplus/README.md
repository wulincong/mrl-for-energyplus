# garage_energyplus: RL2TRPO for EnergyPlus 5-Zone Building Control

## 设计背景

标准 garage 框架假设 **1 worker = 1 env**，但 EnergyPlus 的约束是：

- 1 个 EnergyPlus 进程 = 5 个 zone 同时运行
- 5 个 zone 共享同一个管道通信，无法拆分成独立进程

因此本模块绕过 `LocalSampler`，实现了自定义的 `EplusSharedSampler`，让 1 个 `EnergyPlusMultiAgentEnv` 对外表现为 5 个 RL2 "worker"。

## 文件结构

```
garage_energyplus/
├── env.py          # EplusMonthEnv：单 zone 的 gym.Env 视图，task=(month, zone_id)
├── sampler.py      # EplusSharedSampler：核心，1 env 驱动 5 zone，输出 5 份 EpisodeBatch
├── metrics.py      # 训练指标打印工具
├── evaluation.py   # 全年验证：12 个月 × 5 个 zone
└── run_rl2trpo.py  # 训练入口
```

## 核心数据流

```
EplusRL2TRPO.train()
  → task_sampler.sample(5)          # 采样 1 个月份，生成 5 个 (month, zone_i) 任务
  → EplusSharedSampler.obtain_samples()
      → _ensure_shared_env(month)   # 按需重建 EnergyPlus 进程（月份变化时）
      → _rollout_all_zones()        # 5 个 zone 同步驱动，policy 各自独立推理
      → 返回 5 份 EpisodeBatch，各带 batch_idx=0..4
  → RL2._process_samples()          # 按 batch_idx 分组，拼接同 zone 的多 episode
  → TRPO 策略优化
  → print_epoch_metrics()           # 打印奖励/舒适度/能耗
```

## 空间定义

| 空间 | 维度 | 说明 |
|------|------|------|
| 原始 zone obs | 4 | `[T_outdoor, T_zone, CoolRate, HeatRate]` |
| RL2 增强 obs | 8 | `[obs(4), prev_action(2), prev_reward(1), prev_done(1)]` |
| action（归一化） | 2 | `[htg_setpoint, clg_setpoint]` ∈ [-1, 1] |
| action（实际） | 2 | 加热设定点 [10, 35]°C，制冷设定点 [15, 40]°C |

## 任务定义

- **任务** = `{"month": int, "zone_id": str}`，共 12 × 5 = 60 种任务
- 每次 meta-batch 采样 1 个月份，5 个 zone 共享该月份的 EnergyPlus 仿真
- 月份变化时自动重建 EnergyPlus 进程（修改 IDF RunPeriod 字段）

## 奖励函数

每个 zone 的奖励由 `EnergyPlusMultiAgentEnv._compute_rewards()` 计算：

```
reward = gaussian(T_zone, center=23.5°C) + trapezoid_penalty + hvac_power_penalty + smoothness_penalty
```

- **gaussian**：温度越接近 23.5°C 奖励越高
- **trapezoid_penalty**：温度偏离 [23.0, 24.0]°C 时线性惩罚
- **hvac_power_penalty**：HVAC 功率惩罚（节能目标）
- **smoothness_penalty**：动作变化量惩罚（避免频繁调节）

## 日志文件

训练过程中自动写入 `log_dir/` 下：

| 文件 | 内容 |
|------|------|
| `episode_metrics.csv` | 每 episode 的 month、zone_id、return、comfort_ratio、hvac_power_sum |
| `yearly_validation.csv` | 每 epoch 验证后的 12 个月 return、comfort_ratio、hvac_power |
| `month-MM-*/` | 各月份 EnergyPlus 仿真输出（IDF、EPW、eplusout.csv.gz 等） |

## 使用方法

```bash
source /home/wlc/miniconda3/etc/profile.d/conda.sh
conda activate Eplus

python garage_energyplus/run_rl2trpo.py \
  --model EnergyPlus/5Zone/5ZoneAirCooled.idf \
  --weather EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw \
  --max_episode_length 288 \
  --n_epochs 30 \
  --episode_per_task 2 \
  --validate_full_year \
  --log_dir eplog/garage-rl2-trpo
```

python garage_energyplus/run_rl2trpo.py \
    --model EnergyPlus/5Zone/5ZoneAirCooled.idf \
    --weather EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw \
    --full_year_sim \
    --n_epochs 30 \
    --episode_per_task 1 \
    --log_dir eplog/garage-rl2-full-year


### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_episode_length` | 96 | 每 episode 的时间步数（IDF Timestep=4，即15分钟/步，96步=1天；全年模式自动设为35040） |
| `--n_epochs` | 30 | 训练 epoch 数 |
| `--episode_per_task` | 1 | 每个 zone 每次 epoch 的 episode 数 |
| `--hidden_dim` | 64 | GRU 隐状态维度 |
| `--validate_full_year` | False | 每 epoch 后运行全年验证 |
| `--print_metrics_last_n` | 50 | 打印最近 N 个 episode 的平均指标 |

## 训练输出示例

```
============================================================
[epoch  5] Collecting samples from EnergyPlus...
[EplusSharedSampler] Built env for month=07, log=eplog/garage-rl2-trpo/month-07-...
[train][epoch   5] n_eps=  50  return= -0.1423  comfort=0.612  hvac_power=8432.1
  month= 7  return= -0.1423  comfort=0.612  hvac=8432.1

[epoch  5] Running full-year validation (12 months x 5 zones)...
  [valid] month= 1  return=  -0.089  comfort=0.821  hvac=3241.0
  [valid] month= 2  return=  -0.095  comfort=0.798  hvac=3512.3
  ...
  [valid] month=12  return=  -0.091  comfort=0.815  hvac=3380.2
[valid][epoch   5] year_return=  -1.234  comfort=0.714  hvac=5821.3
```

## 注意事项

- `meta_batch_size` 固定为 5（对应 5 个 zone），不可修改
- 每次 meta-batch 只运行 1 个月份的仿真，月份在 epoch 间随机变化
- EnergyPlus 进程在月份切换时会重启，切换有一定开销
- 验证时每个月份会单独启动 EnergyPlus 进程（共 12 次），耗时较长
