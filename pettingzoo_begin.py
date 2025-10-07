from pettingzoo.butterfly import cooperative_pong_v5

if __name__ == "__main__":
    env = cooperative_pong_v5.env()
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()

        env.step(action)
        # print(f"Agent: {agent}, Obs: {observation}, Reward: {reward}, Termination: {termination}, Truncation: {truncation}, Info: {info}")

    env.close()
