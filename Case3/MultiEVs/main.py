from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
import numpy as np
import pandas as pd
import utils
import global_var

data = pd.read_csv("output_data.csv")    # 读取travel data

model = DQN.load(".../best_model.zip")  # 加载训练好的模型（risk-prone或者risk averse）

n_steps = 20
n_DWERs = 12
n_nodes = 41
chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Mch = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Nch = [13, 1, 23, 12, 21, 9, 27, 26, 38, 37, 7, 6]  # WCL终点
WCLs = [(1, 13), (13, 1), (12, 23), (23, 12), (9, 21), (21, 9), (26, 27), (27, 26), (37, 38), (38, 37), (6, 7), (7, 6)]
allNodes = list(range(41))
comNodes = [node for node in allNodes if node not in chgNodes]  # 非充电节点
emax = 80  # 电池最大容量
alpha = 0.15  # kwh/km 每km耗电量
rho_ave = 0.7673

n_comNodes = len(comNodes)
startLoc = data['上车点编号'].values
endLoc = data['下车点编号'].values
Total_reward = np.empty(len(startLoc))
WCL_count = {}
Mch_chosenWCL = np.empty(len(startLoc))
Nch_chosenWCL = np.empty(len(startLoc))

for WCL in WCLs:
    WCL_count[WCL] = 0

# np.random.seed(42)  # 设置随机数

for k in range(len(startLoc)):

    ns = int(startLoc[k])
    ne = int(endLoc[k])
    print('起点和终点分别为：', ns, ne)

    loc = ns
    SOC = np.array([np.random.uniform(low=0.4, high=0.6, )])
    # print("初始SOC：", SOC)
    curTime = np.zeros(1)
    velocity = np.zeros(4)

    G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix \
        = utils.paramInit()  # 第一次用到random seed
    # print("初始化的Vmn为", Vmn)

    global_var._init()
    Wij = G_eqv_adj_matrix  # 初始化全局变量Wij
    global_var.set_value('Wij', Wij)

    idx = 0
    for adjNode in G_velocity.neighbors(loc):
        velocity[idx] = G_velocity_adj_matrix[loc, adjNode]
        idx = idx + 1

    Ctr, L, Cch = utils.getFeatures(
        rho_ch_s,
        Vmn,
        G_length_adj_matrix,
        G_eqv_adj_matrix,
        et=SOC * emax,
        start_node=loc,
        end_node=ne)

    Ctr = np.array(list(Ctr.values()))  # 视图对象（不支持索引）→list → array
    Cch = np.array(list(Cch.values()))
    L2 = np.empty([n_DWERs, 2], dtype=int)
    idx = 0
    for key in L.keys():
        L2[idx, 0] = L[key][0]
        if len(L[key]) >= 2:
            L2[idx, 1] = L[key][1]
        else:
            L2[idx, 1] = L2[idx, 0]
        idx = idx + 1

    obs_dict = {"loc": loc, "SOC": SOC, "curTime": curTime, "velocity": velocity,
                "travelCost": Ctr, "chgCost": Cch, "optPath": L2}
    # print("Initial obs_dict =", obs_dict)

    env = DynChgEnv()

    env.set_obs(obs_dict, ne)  # 令self._loc = loc

    env = FlattenObservation(env)
    obs = FlattenObservation.observation(self=env, observation=obs_dict)  # 把dict型flatten
    # print("Initial obs =", obs)

    reward1 = []
    for step in range(n_steps):
        # print(f"Step {step + 1}")
        action_tuple = model.predict(obs, deterministic=True)  # 由model.predict给定
        action = int(action_tuple[0])
        # print("action = ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        # print("The", step + 1, "step reward is", reward)
        reward1.append(reward)
        done = terminated or truncated
        # print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            # print("Goal reached!", "info=", info)
            break

    total_reward = sum(reward1)   # reward求和

    print('The chosen DWER is:', (Mch[action], Nch[action]), 'The total reward is', total_reward)

    Total_reward[k] = total_reward
    WCL_count[(Mch[action], Nch[action])] = WCL_count[(Mch[action], Nch[action])] + 1
    Mch_chosenWCL[k] = Mch[action]
    Nch_chosenWCL[k] = Nch[action]


results = np.column_stack((startLoc, endLoc, Mch_chosenWCL, Nch_chosenWCL, Total_reward))

print(WCL_count)

np.savetxt("multiEV_results.csv", results, delimiter=",", header="StartLoc,EndLoc, Mch, Nch, TotalReward")
