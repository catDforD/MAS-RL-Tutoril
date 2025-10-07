# # Run `pip install "gymnasium[classic-control]"` for this example.
# import gymnasium as gym

# # Create our training environment - a cart with a pole that needs balancing
# env = gym.make("CartPole-v1", render_mode="human")

# # Reset environment to start a new episode
# observation, info = env.reset()
# # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# # info: extra debugging information (usually not needed for basic learning)

# print(f"Starting observation: {observation}")
# # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

# episode_over = False
# total_reward = 0

# while not episode_over:
#     # Choose an action: 0 = push cart left, 1 = push cart right
#     action = env.action_space.sample()  # Random action for now - real agents will be smarter!

#     # Take the action and see what happens
#     observation, reward, terminated, truncated, info = env.step(action)

#     # reward: +1 for each step the pole stays upright
#     # terminated: True if pole falls too far (agent failed)
#     # truncated: True if we hit the time limit (500 steps)

#     total_reward += reward
#     episode_over = terminated or truncated

# print(f"Episode finished! Total reward: {total_reward}")
# print(f"Final status: {episode_over}")
# env.close()

# # 2.space for more examples...
# import gymnasium as gym

# # Discrete action space (button presses)
# env = gym.make("CartPole-v1")
# print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
# print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# # Box observation space (continuous values)
# print(f"Observation space: {env.observation_space}")  # Box with 4 values
# # Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
# print(f"Sample observation: {env.observation_space.sample()}")  # Random valid observation

# # 3. wrappers
# import gymnasium as gym
# from gymnasium.wrappers import TimeLimit, RecordVideo

# # Create base environment
# env = gym.make("CartPole-v1", render_mode="rgb_array")
# # Limit episodes to 200 steps
# env = TimeLimit(env, max_episode_steps=200)
# # Record videos of episodes
# env = RecordVideo(env=env, video_folder="./videos", episode_trigger=lambda x: x % 5 == 0)

# # Reset environment to start a new episode
# observation, info = env.reset()
# # observation: what the agent can "see" - cart position, velocity, pole angle,  
# # info: extra debugging information (usually not needed for basic learning)

# episode_over = False
# total_reward = 0

# while not episode_over:
#     # Choose an action: 0 = push cart left, 1 = push cart right
#     action = env.action_space.sample()  # Random action for now - real agents will be smarter!

#     # Take the action and see what happens
#     observation, reward, terminated, truncated, info = env.step(action)

#     # reward: +1 for each step the pole stays upright
#     # terminated: True if pole falls too far (agent failed)
#     # truncated: True if we hit the time limit (500 steps)

#     total_reward += reward
#     episode_over = terminated or truncated

# print(f"Episode finished! Total reward: {total_reward}")
# print(f"Final status: {episode_over}")
# env.close()

# 