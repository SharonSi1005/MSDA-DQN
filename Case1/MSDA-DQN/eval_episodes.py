from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor


env = DynChgEnv()
env = FlattenObservation(env)

log_eval = "30/eval"
env = Monitor(env, log_eval)

log_dir = "30"
model = DQN.load(f"{log_dir}/best_model.zip", env=env)
print('model loaded')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
