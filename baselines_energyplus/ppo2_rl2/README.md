# EnergyPlus 的 PPO2 + RL^2（LSTM）说明

本目录是在现有 EnergyPlus 多智能体环境之上，添加 RL^2 风格的元强化学习训练流程。
核心思想是：
- 每个 meta-episode 内包含多个普通 episode；
- policy 的 LSTM 状态在 meta-episode 内不清零，用于“适应任务”；
- 仅在 meta-episode 边界切换任务并重置策略状态。

修改点与新增文件

1) 元任务封装（RL^2 风格）
- 文件：baselines_energyplus/ppo2_rl2/rl2_env.py
- EnergyPlusTaskSampler：每个 meta-episode 开始时采样一个任务（当前为天气文件）。
- RL2MetaEnv：
  - 包装单个 EnergyPlus 环境，连续运行 K 个 inner episode；
  - 不在 inner episode 之间重置 policy 状态；
  - 将上一时刻的 action、reward、done 拼接到 observation；
  - 只在 meta-episode 边界切换任务并重置环境。

2) LSTM 版 PPO2
- 文件：
  - baselines_energyplus/ppo2_rl2/ppo2_lstm.py
  - baselines_energyplus/ppo2_rl2/runner_lstm.py
- ppo2_lstm.py：
  - 使用 tf.keras.layers.LSTMCell 实现策略/价值网络；
  - 实现支持序列的 PPO2 更新（recurrent PPO2）；
  - 使用 masks 在 episode 结束时重置隐藏状态。
- runner_lstm.py：
  - 采样时携带 LSTM 状态；
  - 计算 GAE 作为 returns。

3) 训练入口
- 文件：baselines_energyplus/ppo2_rl2/run_energyplus.py
- 将 RL2MetaEnv + LSTM PPO2 连接起来。
- 底层环境仍使用 EnergyPlusMASingleEnv（5-zone 多智能体拼接成单智能体）。
- 任务采样支持：--task-weathers 或 --task-weather-file。

与原 baselines PPO2 的区别
- 原版 baselines PPO2 不支持 recurrent（会直接报错）。
- 这里新增了一套 LSTM 版 PPO2 路径，使 RL^2 真正依赖 RNN 记忆。

示例命令

python baselines_energyplus/ppo2_rl2/run_energyplus.py \
  --model EnergyPlus/5Zone/5ZoneAirCooled.idf \
  --weather EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw \
  --task-weathers "EnergyPlus/Model-9-5-0/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw,EnergyPlus/Model-9-5-0/WeatherData/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw" \
  --meta-episodes 3 \
  --total-timesteps 200000

备注
- EnergyPlusMASingleEnv 的 action 是 [-1, 1] 归一化后再缩放到设定点。
- meta-episode 长度通过 --meta-episodes 控制。
- LSTM hidden size 可在 ppo2_lstm.learn 中调整 hidden_size。
