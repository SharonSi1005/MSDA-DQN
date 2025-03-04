import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

log_dir = "result"

x, y = ts2xy(load_results(log_dir), "episodes")


results = np.column_stack((x, y))

np.savetxt(log_dir + "/trainingCurve_SP-A2C.csv", results, delimiter=",")