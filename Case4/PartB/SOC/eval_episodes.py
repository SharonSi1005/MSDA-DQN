from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
import numpy as np

# 给定不同的error_probability和error_percentage进行evaluate

np.random.seed(42) 

error_probability = 0.1
error_percentage = 0.15
print(f"error_probability:{error_probability},error_percentage: {error_percentage}")

env = DynChgEnv(error_probability=error_probability, error_percentage=error_percentage)
env = FlattenObservation(env)
log_eval = "Result/10-15"
env = Monitor(env, log_eval)

log_dir = "..."   # 训练好的模型所在位置
model = DQN.load(f"{log_dir}/best_model.zip", env=env)
print('model loaded')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
