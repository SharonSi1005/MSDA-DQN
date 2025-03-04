import os
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from CustomCb import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.evaluation import evaluate_policy


# Create log dir
log_dir = "result"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = DynChgEnv()
env = FlattenObservation(env)
env = Monitor(env, log_dir)

# Create Callback
callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir, verbose=1)

model = PPO('MlpPolicy', env, learning_rate=0.0003, batch_size=64, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128, 128, 128]))

# train the model
model.learn(total_timesteps=800000, callback=callback, progress_bar=True)  # 显示进度条

save_path = os.path.join(log_dir, "final_model")
model.save(save_path)

test_env = DynChgEnv()
test_env = FlattenObservation(test_env)
log_eval = "result/eval"
test_env = Monitor(test_env, log_eval)

# test the model
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10000)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


