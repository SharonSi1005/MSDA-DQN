import os
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# 按顺序依次执行4个scenario的主程序

'''
# %%%%%%%%%%%%%%%%%%%%%%Scenario 1%%%%%%%%%%%%%%%%%%%%%%%%%

# Create log dir
log_dir_1 = ".../Case4/PartA/S1/result"
os.makedirs(log_dir_1, exist_ok=True)

save_path_1 = os.path.join(log_dir_1, "saved_model")
checkpoint_callback_1 = CheckpointCallback(
  save_freq=100000,
  save_path=save_path_1,
)

# Create and wrap the environment
env_1 = DynChgEnv()
env_1 = FlattenObservation(env_1)
env_1 = Monitor(env_1, log_dir_1)

# Build the model
model = DQN('MlpPolicy', env_1, learning_rate=0.0003, batch_size=64, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128, 128, 128]))

# train the model
model.learn(total_timesteps=600000, callback=checkpoint_callback_1, progress_bar=True)  # 显示进度条


# %%%%%%%%%%%%%%%%Scenario 2%%%%%%%%%%%%%%%%%%%%%%%%%

# Create log dir
log_dir_2 = ".../Case4/PartA/S2/result"
os.makedirs(log_dir_2, exist_ok=True)

save_path_2 = os.path.join(log_dir_2, "saved_model")
checkpoint_callback_2 = CheckpointCallback(
    save_freq=50000,
    save_path=save_path_2,
)

env_2 = DynChgEnv(EnableBlockWCLs=True)
env_2 = FlattenObservation(env_2)
env_2 = Monitor(env_2, log_dir_2)

log_dir_1 = "/home/sharon/Chaoran-mycode/IOT_R1/Case4/PartA/S1/result"
save_path_1 = os.path.join(log_dir_1, "saved_model")

# Load the model
model = DQN.load(f"{save_path_1}/rl_model_600000_steps.zip", env=env_2)

# train the model
model.learn(total_timesteps=200000, callback=checkpoint_callback_2, progress_bar=True)  # 显示进度条



# %%%%%%%%%%%%%%%%Scenario 3%%%%%%%%%%%%%%%%%%%%%%

log_dir_3 = ".../Case4/PartA/S3/result"
os.makedirs(log_dir_3, exist_ok=True)

save_path_3 = os.path.join(log_dir_3, "saved_model")
checkpoint_callback_3 = CheckpointCallback(
    save_freq=50000,
    save_path=save_path_3,
)

env_3 = DynChgEnv(EnableDynPrices=True)
env_3 = FlattenObservation(env_3)
env_3 = Monitor(env_3, log_dir_3)

# Load the model
save_path_2 = "/home/sharon/Chaoran-mycode/IOT_R1/Case4/PartA/S2/result/saved_model"
model = DQN.load(f"{save_path_2}/rl_model_200000_steps.zip", env=env_3)

# train the model
model.learn(total_timesteps=200000, callback=checkpoint_callback_3, progress_bar=True)  # 显示进度条

'''
# %%%%%%%%%%%%%%%%Scenario 4%%%%%%%%%%%%%%%%%%%%%%
"""如果Vmn小于下限，则给进入WCL(m,n)的GPL加收拥堵费"""

log_dir_4 = ".../Case4/PartA/S4/result"
os.makedirs(log_dir_4, exist_ok=True)

save_path_4 = os.path.join(log_dir_4, "saved_model")
checkpoint_callback_4 = CheckpointCallback(
    save_freq=50000,
    save_path=save_path_4,
)

env_4 = DynChgEnv(EnableTollGPLs=True)
env_4 = FlattenObservation(env_4)
env_4 = Monitor(env_4, log_dir_4)

# Load the model
save_path_3 = "/home/sharon/Chaoran-mycode/IOT_R1/Case4/PartA/S3/result/saved_model"
model = DQN.load(f"{save_path_3}/rl_model_200000_steps.zip", env=env_4)

# train the model
model.learn(total_timesteps=200000, callback=checkpoint_callback_4, progress_bar=True)  # 显示进度条
