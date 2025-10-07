import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# 1. Env 构造与包装
def make_env(seed: int = 0, render_mode=None):
    """
    返回一个已包装好的单环境创建函数。
    - Resize 到 84x84（更快）
    - FrameSkip=2（减少冗余帧）
    - RecordEpisodeStatistics（记录 ep_rew_mean等）
    - ClipAction（动作范围对齐）
    """
    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # 色彩信息很重要， 保留 RGB： 仅仅缩放以降算力
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        # env = gym.wrappers.FrameSkip(env, skip=2)
        # if np.any(np.isinf(env.action_space.low)) or np.any(np.isinf(env.action_space.high)):
        #     env.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        #                                     high=np.array([1.0, 1.0, 2.0], dtype=np.float32),
        #                                     dtype=np.float32)
        # env = gym.wrappers.ClipAction(env)
        env.reset(seed=seed)
        # print(env.action_space)
        # print("low =", env.action_space.low)
        # print("high=", env.action_space.high)

        return env
    return _thunk

def build_vec_env(n_envs: int, seed_base: int = 0, render_mode=None):
    """
    多线程并行环境
    """
    env_fns = [make_env(seed=seed_base + i, render_mode=render_mode) for i in range(n_envs)]
    vec = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec)  # 记录每回合统计
    vec = VecTransposeImage(vec) # 调整图像维度从 (H, W, C) 到 (C, H, W)
    # 通过多帧堆叠，通常用于增强时序信息（常见于时序数据处理、视频帧分析等场景），为后续模型输入提供更丰富的时序上下文
    vec = VecFrameStack(vec, n_stack=4, channels_order='first') # 堆叠 4 帧
    return vec

# 2. 训练主流程
def main():
    run_name = "SAC_CarRacing_v1"
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    models_dir = f"./logs/models/models"
    os.makedirs(models_dir, exist_ok=True)

    # 并行 8 个环境
    train_env = build_vec_env(n_envs=4, seed_base=42)

    # 评估环境
    eval_env = build_vec_env(n_envs=1, seed_base=1000)

    # Logger (可用 tensorboard --logdir runs)
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # SAC 超参： 对 CarRacing 较稳的设置
    policy_kwargs = dict(    # 使用 NatureCNN（SB3 内置）即可
        # 可按需自定义 features extractor 的通道数
        # features_extractor_class=NatureCNN, features_extractor_kwargs=dict(features_dim=512)
    )
    model = SAC(
        policy=CnnPolicy,
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100_000,        # 图像任务需要更大的经验回放
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(4, "step"),  # 每回合训练一次（也可用 (4, "step")）
        gradient_steps=4,            # 每次更新步数
        ent_coef="auto",             # 最大熵自动调节
        target_update_interval=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto",
        seed=42,
    )
    model.set_logger(new_logger)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=models_dir,
        eval_freq=10_000,            # 每 1e4 env-steps 评估一次
        n_eval_episodes=5,
        deterministic=False,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=100_000,           # 每 1e5 步存一次
        save_path=models_dir,
        name_prefix="ckpt"
    )
    total_timesteps = 1_500_000          # 总训练步数
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, ckpt_callback],
    )

    # 保存最终模型
    file_path = os.path.join(models_dir, f"{run_name}_final")
    model.save(file_path)
    print(f"✅ Training done. Model saved to: {file_path}")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()