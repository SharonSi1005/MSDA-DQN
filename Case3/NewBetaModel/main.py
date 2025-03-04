import os
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from CustomCb import RiskPreferenceCallback
from stable_baselines3.common.evaluation import evaluate_policy
# from gymnasium.wrappers import TimeLimit


# Create log dir
log_dir = "TrainingResult"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = DynChgEnv()
env = FlattenObservation(env)
vec_env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)

# Create Callback
callback = RiskPreferenceCallback(log_file="risk_log_energy.csv", verbose=1)

model = DQN('MlpPolicy', vec_env, learning_rate=0.0003, batch_size=64, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128, 128, 128]))

# train the model
model.learn(total_timesteps=800000, callback=callback, progress_bar=True)  # 显示进度条

save_path = os.path.join(log_dir, "final_model")
model.save(save_path)

# test the model
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
