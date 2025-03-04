import os
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from CustomCb import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.evaluation import evaluate_policy


# Create log dir
log_dir = "risk-prone/FINAL25"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = DynChgEnv()
env = FlattenObservation(env)
# env = TimeLimit(env, max_episode_steps=20)
env = Monitor(env, log_dir)

# Create Callback
callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir, verbose=1)

model = DQN('MlpPolicy', env, learning_rate=0.0003, batch_size=64, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128, 128, 128]))

# train the model
model.learn(total_timesteps=800000, callback=callback, progress_bar=True)  # 显示进度条

# test the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

save_path = os.path.join(log_dir, "final_model")
model.save(save_path)
