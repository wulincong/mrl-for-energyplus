import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register

class GridWorldEnv(gym.Env):
    """
    状态空间：[0, size-1]
    动作空间：0=向左，1=向右
    奖励：到达终点+10， 每步-1，撞墙-2
    终止：到达终点 or 超过最步数
    """
    metadata = {"render.modes": ["human", "ansi"]}   # 旧版 gym 用点号格式

    def __init__(self, size=5, max_steps=30):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.max_steps = max_steps

        self.observation_space = spaces.Discrete(size)

        self.action_space = spaces.Discrete(2)  # 取0 1
        self._agent_pos = 0
        self._step_count=0

    def reset(self): 
        self._agent_pos = 0  # 当前位置
        self._step_count = 0 # 计数器

        return self._agent_pos
    
    def step(self, action):
        self._step_count += 1
        hit_wall = False

        if action == 0:
            if self._agent_pos > 0:
                self._agent_pos -= 1
            else:
                hit_wall = True
        elif action == 1:
            if self._agent_pos < self.size - 1:
                self._agent_pos += 1
            else:
                hit_wall = True
        reach_goal = (self._agent_pos == self.size - 1)
        timeout    = (self._step_count >= self.max_steps)
        done       = reach_goal or timeout

        if reach_goal: reward = 10.0
        elif hit_wall: reward = -2.0
        else:          reward = -1.0

        obs = self._agent_pos
        info = {"step": self._step_count, "hit_wall":hit_wall}

        return obs, reward, done, info
    
    # ⑤ render：旧版接受 mode 参数
    def render(self, mode="human"):
        grid = ["_"] * self.size
        grid[self._agent_pos] = "A"
        grid[-1] = "G" if self._agent_pos != self.size - 1 else "★"
        output = "[" + "|".join(grid) + f"]  step={self._step_count}"
        if mode == "human":
            print(output)
        return output

    # ⑥ close
    def close(self):
        pass


register(id="GridWorld-v310",
         entry_point=GridWorldEnv, 
         kwargs={"size":5, "max_steps":20},
         max_episode_steps=50,
         reward_threshold=9.0,
         )


def run_random_policy(episodes=3):
    env = gym.make("GridWorld-v310")

    for ep in range(episodes):
        obs = env.reset()              # 旧版：只返回 obs
        print(f"\n=== Episode {ep + 1} ===")
        total_reward = 0.0

        while True:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)   # 旧版：四元组
            total_reward += reward

            if done:
                env.render()
                reached = (obs == env.unwrapped.size - 1)
                status = "✅ 到达终点" if reached else "⏰ 超时"
                print(f"{status} | 总奖励: {total_reward:.1f}")
                break

    env.close()


# ─────────────────────────────────────────────
# 5. Q-Learning 训练
# ─────────────────────────────────────────────
def train_q_learning(episodes=500, alpha=0.1, gamma=0.99, epsilon=0.3):
    env = gym.make("GridWorld-v310")
    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rewards_log = []

    for ep in range(episodes):
        obs = env.reset()              # 旧版
        total_reward = 0.0

        while True:
            # ε-greedy 策略
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))

            next_obs, reward, done, _ = env.step(action)   # 旧版

            # Q-Learning 更新
            best_next = 0.0 if done else np.max(Q[next_obs])
            Q[obs, action] += alpha * (reward + gamma * best_next - Q[obs, action])

            total_reward += reward
            obs = next_obs

            if done:
                break

        rewards_log.append(total_reward)

    env.close()

    print("\n=== Q-Learning 训练完成 ===")
    print(f"最终 Q 表:\n{np.round(Q, 2)}")
    avg_last = np.mean(rewards_log[-50:])
    print(f"最后 50 局平均奖励: {avg_last:.2f}")

    policy_names = {0: "←", 1: "→"}
    policy = [policy_names[int(np.argmax(Q[s]))] for s in range(n_states)]
    print(f"学到的策略: {' '.join(policy)}")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_random_policy()
    train_q_learning()