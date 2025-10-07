# Gridworld Agent 设计

# Q 学习：
# 行 = 智能体可能遇到的不同情况（状态）
# 列 = 智能体可以采取的不同动作
# 值 = 在那种情况下该动作有多好（预期的未来奖励）

# 对于 Gridworld Agent：
# 状态 = 智能体和目标的位置
# 动作 = 上、下、左、右
# 奖励 = 到达目标 +1，其他 0

# 学习过程：
# 1. 尝试一个动作，看看会发生什么 (奖励 + 新状态)
# 2. 更新 Q 表格以反映新信息
# 3. 逐渐改进通过尝试动作和更新估计
# 4. 平衡探索与利用： 尝试新事物与利用已知有效的方法
import logging
import numpy as np
import gymnasium_env
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

class GridWorldAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay:float,
            final_epsilon:float,
            discount_factor: float = 0.95,
        ):
        self.env = env

        # Q-table: maps (state, action) to expected reward
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Epsilon-greedy parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning process
        self.training_error = []

    def get_action(self, obs: tuple[int, int, int, int]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (right), 1 (up), 2 (left), 3 (down)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: action with highest Q-value for current state
            obs_tuple = self.convert_obs_to_tuple(obs)
            return int(np.argmax(self.q_values[obs_tuple]))

    def update(
        self,
        obs: tuple[int, int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, int, int],
    ):
        """
        Update Q-value based on observed transition.
        This is the heart of Q-learning: learn from (state, action, reward, next_state).
        """
        next_obs_tuple = self.convert_obs_to_tuple(next_obs)

        future_q_values = (not terminated) * np.max(self.q_values[next_obs_tuple])
        target = reward + self.discount_factor * future_q_values

        obs_tuple = self.convert_obs_to_tuple(obs)
        temporal_difference = target - self.q_values[obs_tuple][action]

        # Update Q-value for the (state, action) pair
        self.q_values[obs_tuple][action] = (
            self.q_values[obs_tuple][action] + self.lr * temporal_difference
        )

        # Track learning progress
        self.training_error.append(abs(temporal_difference))

    def decay_epsilon(self):
        """
        Decay epsilon after each episode to reduce exploration over time.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
    def convert_obs_to_tuple(self, obs: dict) -> tuple[int, int, int, int]:
        """
        Convert observation dict to a tuple for Q-table indexing.
        """
        agent_x, agent_y = obs["agent"]
        target_x, target_y = obs["target"]
        return (agent_x, agent_y, target_x, target_y)

def test_agent(agent, env, num_episodes=10):
    """
    Test agent performence 
    """
    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during testing

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            obs = next_obs

        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon  # Restore original epsilon

    average_reward = np.mean(total_rewards)
    win_rate = np.mean(np.array(total_rewards) > 0)
    logging.info(f"Tested over {num_episodes} episodes: Average Reward = {average_reward:.3f}, Win Rate = {win_rate:.1%}, Std Dev = {np.std(total_rewards):.3f}")
    print(f"Tested over {num_episodes} episodes: \n Average Reward = {average_reward:.3f} \n Win Rate = {win_rate:.1%} \n Std Dev = {np.std(total_rewards):.3f}")

if __name__ == "__main__":
    n_episodes = 200000
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    training_perids = 10000
    env_name = "gymnasium_env/GridWorld-v0"

    # Set up logging for episode statistics
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename='./logs/gridworld_agent_v4.log', filemode='w')

    # Reward list for tracking performance
    reward_list = []

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=f"./videos/{env_name}",
        episode_trigger=lambda x: x % training_perids == 0
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    print(f"Training on {env_name} for {n_episodes} episodes...")
    print(f"Video recordings every {training_perids} episodes.")
    print(f"Video folder: ./videos/{env_name}")

    agent = GridWorldAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=0.95,
    )

    for episode in tqdm(range(n_episodes)):
        # Reset environment to initial state
        obs, info = env.reset()
        done = False

        # Run one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        reward_list.append(reward)

        agent.decay_epsilon()
        if "episode_data" in info:
            episode_data = info["episode_data"]
            logging.info(
                f"Episode {episode + 1}: Reward = {episode_data['reward']}, Steps = {episode_data['steps']}, Time = {episode_data['time']}"
            )

    # Test the trained agent
    test_agent(agent, env, num_episodes=10000)

    # Plot training error over time
    episodes = range(len(reward_list))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, reward_list, alpha=0.3, label='Episode Reward')
    
    window = 100
    if len(reward_list) > window:
        moving_avg = [sum(reward_list[i:i+window])/window
                    for i in range(len(reward_list)-window+1)]
        plt.plot(range(window-1, len(reward_list)), moving_avg,
                label=f'{window}-Episode Moving Average', linewidth=2)
        
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Reward over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
        
    env.close()