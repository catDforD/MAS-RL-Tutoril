import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack, VecMonitor

def make_env_eval(render_mode="human"):
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        return env
    return _thunk

def build_eval_vec():
    env = DummyVecEnv([make_env_eval("human")])
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4, channels_order='first')
    return env

if __name__ == "__main__":
    # 改成你的模型路径：优先加载评估回调保存的 best_model.zip
    model_path = "./logs/models/models/best_model.zip"
    assert os.path.exists(model_path), f"模型路径不存在: {model_path}"

    env = build_eval_vec()
    model = SAC.load(model_path, env=env, device="auto")

    obs = env.reset()
    done = False
    episode_return = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_return += float(reward)
        if done.any() if isinstance(done, np.ndarray) else done:
            print(f"Episode return: {episode_return:.2f}")
            episode_return = 0.0
            obs = env.reset()
        # 降低渲染抢占（可按需调整）
        time.sleep(1/60)

    env.close()
