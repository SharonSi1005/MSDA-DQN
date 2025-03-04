from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor

env = DynChgEnv()
env = FlattenObservation(env)
log_eval = ".../Case4/PartA/S1/result/eval"
env = Monitor(env, log_eval)

log_dir = ".../Case4/PartA/S1/result/saved_model"
model = DQN.load(f"{log_dir}/rl_model_600000_steps.zip", env=env)
print('model loaded')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
