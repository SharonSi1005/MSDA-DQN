import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy


# 指定日志文件夹（包含 monitor.csv）
log_folder = "./logs"

# 加载日志数据（load_results 会自动读取 monitor.csv，并合并所有数据）
results = load_results(log_folder)

# 使用 ts2xy 提取数据，其中 'episodes' 选项会返回一个连续的 episode 序号数组和对应的 reward 数组
x, y = ts2xy(results, "episodes")

# monitor.csv 中的 episode 顺序不连续，可以手动构造连续的 episode 编号：
x_continuous = np.arange(1, len(y) + 1)

results = np.column_stack((x, x_continuous, y))

np.savetxt(log_folder + "/trainingCurve_realworld_0217.csv", results, delimiter=",")