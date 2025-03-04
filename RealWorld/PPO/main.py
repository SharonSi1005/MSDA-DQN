import os
from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from CustomCb import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.evaluation import evaluate_policy
import utils

# 定义全局步数文件
GLOBAL_TIMESTEP_FILE = "./global_timestep.txt"


def load_global_timestep():
    """从文件加载全局步数"""
    if os.path.exists(GLOBAL_TIMESTEP_FILE):
        with open(GLOBAL_TIMESTEP_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def save_global_timestep(timestep):
    """保存全局步数"""
    with open(GLOBAL_TIMESTEP_FILE, "w") as f:
        f.write(str(timestep))


# 读取上次的全局时间步
global_timestep = load_global_timestep()

# Create log dir
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix = \
    utils.paramInit()
env = DynChgEnv(sorted_nodes, WCL)
env = FlattenObservation(env)
env = Monitor(env, log_dir, allow_early_resets=True)

# Create Callback
callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir, verbose=1)

# 加载模型
CHECKPOINT_DIR = "./checkpoints"
LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "last_model.zip")

if os.path.exists(LAST_CHECKPOINT):
    print(f"Loading model from {LAST_CHECKPOINT}...")
    model = PPO.load(LAST_CHECKPOINT, env=env)
else:
    model = PPO('MlpPolicy', env, learning_rate=0.0003, batch_size=64, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128, 128, 128]))

# train the model
TIMESTEPS = 40000
TOTAL_TIMESTEPS = 2000000

while global_timestep < TOTAL_TIMESTEPS:
    print(f"Training from global timestep: {global_timestep}")

    # 训练 & 断点续训（确保时间步数不重置）
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=callback, progress_bar=True)
    global_timestep += TIMESTEPS
    save_global_timestep(global_timestep)

    # 保存模型
    model.save(LAST_CHECKPOINT)
    print(f"Checkpoint saved at {LAST_CHECKPOINT}, global timestep: {global_timestep}")

# 保存最终模型
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_model.zip")
model.save(FINAL_MODEL_PATH)
print(f"Training complete! Final model saved at {FINAL_MODEL_PATH}")
