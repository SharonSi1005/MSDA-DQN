from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
import utils

sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix = \
    utils.paramInit()
env = DynChgEnv(sorted_nodes, WCL, isTesting=True)
env = FlattenObservation(env)
log_eval = "eval"
env = Monitor(env, log_eval)

log_dir = "logs"
model = PPO.load(f"{log_dir}/best_model.zip", env=env)
print('model loaded')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
