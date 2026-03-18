训练方式的共同模式 

所有算法都遵循相同的训练模式： 

def train(self, trainer):
# 初始化
for _ in trainer.step_epochs():# 外层：epoch 循环
    for _ in range(self._n_samples):# 内层：样本循环
        episodes = trainer.obtain_episodes(trainer.step_itr)  # 执行数据搜集
        self._train_once(trainer.step_itr, episodes)
        trainer.step_itr += 1

关键点：
- trainer.step_epochs() 提供 epoch 级别的服务（快照、采样控制）
- trainer.obtain_episodes() 负责数据收集 
- _train_once() 处理单次迭代的优化逻辑 

### meta_rl_algorithm.py - 元强化学习算法基类
你当前打开的这个文件，继承自 RLAlgorithm，专门用于元学习场景：
核心概念：
- get_exploration_policy() - 获取探索策略，用于在适应特定任务前进行探索
- adapt_policy(exploration_policy, exploration_episodes) - 根据探索数据适应到特定任务

设计思想： 元强化学习的两阶段过程
1. 探索阶段：使用探索策略收集数据
2. 适应阶段：基于探索数据快速适应到新任务

  EpisodeBatch 的定义    
 
  EpisodeBatch 是一个批量 episode 的数据容器，继承自 TimeStepBatch（第455行）。它用于存储多个完整 episode 的数据。       
 
  核心数据结构   
 
  @dataclass(frozen=True)
  class EpisodeBatch(TimeStepBatch):     
 
  关键属性：     
 
  1. env_spec - 环境规格 
  2. observations - 形状 (N•[T], O*)     
    - N 是 episode 数量  
    - [T] 是每个 episode 的可变长度      
    - N•[T] 表示所有 episode 的时间步被展平成一维
  3. last_observations - 形状 (N, O*) - 每个 episode 的最后一个观察      
  4. actions - 形状 (N•[T], A*)  
  5. rewards - 形状 (N•[T],)     
  6. step_types - 形状 (N•[T],) - 每步的类型（FIRST/MID/TERMINAL/TIMEOUT）       
  7. lengths - 形状 (N,) - 关键！ 每个 episode 的长度    
  8. env_infos - 环境信息字典    
  9. agent_infos - 智能体信息字典（如 RNN 隐状态）       
  10. episode_infos - episode 级别信息（如目标状态）     
 
  数据存储方式示例       
 
  假设有 3 个 episode，长度分别为 [3, 2, 4]：    
 
  # 原始数据     
  Episode 1: obs=[o1, o2, o3], rewards=[r1, r2, r3]      
  Episode 2: obs=[o4, o5], rewards=[r4, r5]      
  Episode 3: obs=[o6, o7, o8, o9], rewards=[r6, r7, r8, r9]      
 
  # EpisodeBatch 中的存储（展平）
  observations = [o1, o2, o3, o4, o5, o6, o7, o8, o9]  # 形状 (9, O*)    
  rewards = [r1, r2, r3, r4, r5, r6, r7, r8, r9]       # 形状 (9,)       
  lengths = [3, 2, 4]   # 形状 (3,)      
 
  split() 方法详解       
 
  位置： 第648-674行     
 
  def split(self):       
        """Split an EpisodeBatch into a list of EpisodeBatches.    
    
        The opposite of concatenate.       
    
        Returns:   
  list[EpisodeBatch]: A list of EpisodeBatches, with one 
        episode per batch. 
        """
        episodes = []      
        for i, (start, stop) in enumerate(self._episode_ranges()): 
  eps = EpisodeBatch(    
        env_spec=self.env_spec,    
        episode_infos=slice_nested_dict(self.episode_infos_by_episode, i, i + 1),  
        observations=self.observations[start:stop],
        last_observations=np.asarray([self.last_observations[i]]), 
        actions=self.actions[start:stop],  
        rewards=self.rewards[start:stop],  
        env_infos=slice_nested_dict(self.env_infos, start, stop),  
        agent_infos=slice_nested_dict(self.agent_infos, start, stop),      
        step_types=self.step_types[start:stop],    
        lengths=np.asarray([self.lengths[i]]))     
  episodes.append(eps)   
 
      return episodes    
 
  工作原理       
 
  步骤1： 使用 _episode_ranges() 计算每个 episode 的索引范围（第634-646行）：    
 
  def _episode_ranges(self):     
      """Iterate through start and stop indices for each episode."""     
      start = 0  
      for length in self.lengths:
  stop = start + length  
  yield (start, stop)    
  start = stop   
 
  示例： 
  lengths = [3, 2, 4]    
  # _episode_ranges() 生成：     
  # (0, 3)   - Episode 1 
  # (3, 5)   - Episode 2 
  # (5, 9)   - Episode 3 
 
  步骤2： 对每个范围，切片出单个 episode 的数据，创建新的 EpisodeBatch   
 
  在 RL2 中的使用
 
  回到 RL2 代码的第427行：       
 
  for episode in episodes.split():       
      if hasattr(episode, 'batch_idx'):  
  paths_by_task[episode.batch_idx[0]].append(episode)    
      elif 'batch_idx' in episode.agent_infos:   
  paths_by_task[episode.agent_infos['batch_idx'][0]].append(episode)     
 
  这里发生了什么：       
 
  1. episodes 是一个包含多个任务、多个 episode 的大批次  
  2. episodes.split() 将其拆分成单个 episode 的列表      
  3. 每个 episode 通过 batch_idx 标识属于哪个任务
  4. 按任务 ID 分组到 paths_by_task 字典中       
 
  完整数据流示例 
 
  # 假设采样了 2 个任务，每个任务 2 个 episode   
  # Task 0: Episode A (长度3), Episode B (长度2) 
  # Task 1: Episode C (长度4), Episode D (长度3) 
 
  # 原始 EpisodeBatch    
  episodes = EpisodeBatch(       
      observations=[...12个obs...],  # 3+2+4+3=12
      rewards=[...12个reward...],
      lengths=[3, 2, 4, 3],      
      agent_infos={'batch_idx': [0,0,0, 0,0, 1,1,1,1, 1,1,1]}    
  )      
 
  # episodes.split() 返回
  [      
      EpisodeBatch(obs=[...3个...], lengths=[3], agent_infos={'batch_idx': [0,0,0]}),  # Episode A       
      EpisodeBatch(obs=[...2个...], lengths=[2], agent_infos={'batch_idx': [0,0]}),    # Episode B       
      EpisodeBatch(obs=[...4个...], lengths=[4], agent_infos={'batch_idx': [1,1,1,1]}),# Episode C       
      EpisodeBatch(obs=[...3个...], lengths=[3], agent_infos={'batch_idx': [1,1,1]}),  # Episode D       
  ]      
 
  # 按任务分组后 
  paths_by_task = {      
      0: [Episode A, Episode B], 
      1: [Episode C, Episode D]  
  }      
 
  关键设计优势   
 
  1. 内存效率：展平存储避免了嵌套列表    
  2. 批处理友好：可以直接用 numpy 操作整个批次   
  3. 灵活性：通过 lengths 数组可以重建原始 episode 结构  
  4. 可逆操作：concatenate() 和 split() 互为逆操作       
 
  这种设计在 RL2 中特别重要，因为需要频繁地在"批量处理"和"按任务分组"之间切换。  


 EpisodeBatch 的完整生成流程    
 
  1. 顶层调用链  
 
  # RL2 算法中 (rl2.py:342-344)  
  trainer.step_episode = trainer.obtain_episodes(
      trainer.step_itr,          
      env_update=self._task_sampler.sample(self._meta_batch_size)
  )              
 
  2. Trainer 层 (trainer.py:179-229)             
 
  def obtain_episodes(self, itr, batch_size, agent_update, env_update):          
      # 1. 获取策略参数          
      policy = self._algo.policy 
      agent_update = policy.get_param_values()   
 
      # 2. 调用 Sampler 采样     
      episodes = self._sampler.obtain_samples(   
          itr, batch_size,       
          agent_update=agent_update,             
          env_update=env_update  
      )          
 
      # 3. 统计环境步数          
      self._stats.total_env_steps += sum(episodes.lengths)       
      return episodes  # 返回 EpisodeBatch       
 
  3. Sampler 层 (local_sampler.py:134-166)       

  def obtain_samples(self, itr, num_samples, agent_update, env_update):          
      # 1. 更新所有 worker 的策略和环境          
      self._update_workers(agent_update, env_update)             
 
      # 2. 循环采样直到达到所需样本数            
      batches = []               
      completed_samples = 0      
      while True:
          for worker in self._workers:           
              batch = worker.rollout()  # 每个 worker 采样一个 episode           
              completed_samples += len(batch.actions)            
              batches.append(batch)              
              if completed_samples >= num_samples:               
        # 3. 拼接所有 episode 成一个大 batch           
        samples = EpisodeBatch.concatenate(*batches)   
        return samples 

  4. Worker 层 - 核心采样逻辑 (default_worker.py)
 
  rollout() - 采样一个完整 episode (第176-186行) 
 
  def rollout(self):             
      """Sample a single episode of the agent in the environment."""             
      self.start_episode()  # 开始新 episode     
      while not self.step_episode():  # 循环执行步骤             
          pass   
      return self.collect_episode()  # 收集并返回 EpisodeBatch   
 
  start_episode() - 初始化 episode (第91-98行)   
 
  def start_episode(self):       
      self._eps_length = 0       
      self._prev_obs, episode_info = self.env.reset()  # 重置环境
 
      # 保存 episode 级别信息（如目标状态）      
      for k, v in episode_info.items():          
          self._episode_infos[k].append(v)       
 
      self.agent.reset()  # 重置策略（如 RNN 隐状态）            
 
  step_episode() - 执行单步交互 (第100-122行)    
 
  def step_episode(self):        
      """Take a single time-step in the current episode."""      
      if self._eps_length < self._max_episode_length:            
          # 1. 策略选择动作      
          a, agent_info = self.agent.get_action(self._prev_obs)  
 
          # 2. 环境执行动作      
          es = self.env.step(a)  # EnvStep       
 
          # 3. 保存数据到缓冲区  
          self._observations.append(self._prev_obs)              
          self._env_steps.append(es)  # 包含 action, reward, next_obs, env_info  
          for k, v in agent_info.items():        
              self._agent_infos[k].append(v)          


