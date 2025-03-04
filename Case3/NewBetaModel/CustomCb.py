import os
import csv
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)

        return True


class RiskPreferenceCallback(BaseCallback):
    """
    自定义回调，用于监督并记录每个 episode 的 St和每次新行程开始时的 β。
    """

    def __init__(self, log_file: str = "risk_preference_log.csv", verbose: int = 0):
        super(RiskPreferenceCallback, self).__init__(verbose)
        self.log_file = log_file
        self.records = []

    def _on_training_start(self):
        # 初始化日志文件
        with open(self.log_file, "w") as f:
            f.write("episode,Se,beta\n")
        if self.verbose > 0:
            print(f"Logging to {self.log_file}")

    def _on_step(self) -> bool:
        # print(f"Callback Locals: {self.locals.keys()}")
        # 检查行程是否结束
        infos = self.locals.get("infos", [{}])
        # print(f"Callback Infos: {infos}")
        for info in infos:
            if "custom_info" in info:  # 检查行程结束
                custom_info = info["custom_info"]
                episode = custom_info.get("episode", 0)

                Se = custom_info.get("Se", 0)
                beta = custom_info.get("beta", 0)

                # 记录数据
                self.records.append((episode, Se, beta))

                # 写入日志文件
                with open(self.log_file, "a") as f:
                    f.write(f"{episode},{Se},{beta}\n")

                if self.verbose > 0:
                    print(f"Episode {episode} End: Se={Se}, beta={beta}")

        return True

    def _on_training_end(self):
        # 输出总结信息
        if self.verbose > 0:
            print(f"Training finished. Log saved to {self.log_file}")
