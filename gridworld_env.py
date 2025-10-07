from typing import Optional
import gymnasium as gym
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, size: int = 5):
        # The size of square grid
        self.size = size

        # Define the default position of the agent and the goal
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._goal_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured , human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32), # Agent's position box
                "goal": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32), # Goal's position box
            } 
        )

        # Define what actions are avaliable (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map actions to movements in the grid
        # This makes the code more readable than using raw numbers

        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([-1, 0]),  # Move up (positive y)
            2: np.array([0, -1]),  # Move left (negative x)
            3: np.array([0, 1]),   # Move down (negative y)
        }

    # Construct the observation
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "agent": self._agent_location,
            "target": self._goal_location
        }
    
    # Construct the info dict
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._goal_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere in the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place the goal anywhere in the grid and ensure it's not the same as the agent
        self._goal_location = self._agent_location
        while np.array_equal(self._goal_location, self._agent_location):
            self._goal_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    # Step the environment by one timestep
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action to a movement directin
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._goal_location)
        truncated = False

        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

# # Register the environment with Gym
# gym.register(
#     id="gym_gridworld/GridWorld-v0",
#     entry_point=GridWorldEnv,
#     max_episode_steps=300,
# )  

if __name__ == "__main__":
    import gymnasium
    import gymnasium_env
    from gymnasium.utils.env_checker import check_env
    env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human")

    # # This will check your custom environment and output additional warnings if needed
    # try:
    #     check_env(env, warn=True)
    #     print("Environment check passed!")
    # except Exception as e:
    #     print(f"Environment check failed: {e}")

    # Take specific actions sequence to verify behavior
    obs, info = env.reset()
    # Test each action type
    actions = [0, 1, 2, 3]  # Right, Up, Left, Down
    for action in actions:
        old_pos = obs["agent"].copy()
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = obs["agent"]
        print(f"Action {action}: {old_pos} -> {new_pos}, reward: {reward}")