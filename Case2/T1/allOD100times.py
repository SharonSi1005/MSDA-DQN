from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
import numpy as np
import utils
import math
import global_var
from Dijkstra import Graph
import csv

'''
测试不同电池容量
需修改的参数：
1. utils.getFeatures()的emax
2. CustomEnv的self.emax
3. 本文件emax
'''

model = DQN.load(".../best_model.zip")  # 加载训练好的模型

n_steps = 10
n_DWERs = 12
n_nodes = 41
chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Mch = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Nch = [13, 1, 23, 12, 21, 9, 27, 26, 38, 37, 7, 6]  # WCL终点
allNodes = list(range(41))
comNodes = [node for node in allNodes if node not in chgNodes]  # 非充电节点
emax = 40  # 电池最大容量
alpha = 0.15  # kwh/km 每km耗电量
rho_ave = 0.7673

n_comNodes = len(comNodes)
startLoc = np.empty(n_comNodes * (n_comNodes - 1))
endLoc = np.empty(n_comNodes * (n_comNodes - 1))
cost = np.empty(n_comNodes * (n_comNodes - 1))


with open('data_DQN_E1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['StartLoc', 'EndLoc', 'Cost'])

    k = 0
    for ns in comNodes:
        for ne in comNodes:
            if ns != ne:
                startLoc[k] = ns
                endLoc[k] = ne
                print('起点和终点分别为：', ns, ne)

                Obj_list = []

                for seed in range(100):
                    np.random.seed(seed)   # 更新随机数
                    loc = ns
                    SOC = np.array([np.random.uniform(low=0.4, high=0.6, )])
                    # SOC = np.array([0.5], dtype=np.float64)  # 给定初始SOC为40kWh
                    curTime = np.zeros(1)
                    velocity = np.zeros(4)

                    G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix \
                        = utils.paramInit()  # 第一次用到random seed
                    # print("WCL电价：", rho_ch_s)
                    # print("WCL速度：", Vmn)

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

                    env = DynChgEnv(G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix)
                    env.set_obs(obs_dict, ne)  # 令self._loc = loc

                    # env = TimeLimit(env, max_episode_steps=20)
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

                    # print("从起点到m的reward：", reward1)
                    # print("Cch1_lastStep is:", info["Cch1_lastStep"])

                    if bool(info):  # info不为空：结果正常
                        Ctr1 = - sum(reward1[0:-1]) + info["Cch1_lastStep"]  # 除节点m外reward求和，取相反数
                        # print("Ctr1 is:", Ctr1)

                        g = Graph(41)
                        g.graph = G_eqv_adj_matrix

                        dist, prev = g.dijkstra(Nch[action])

                        Ctr2 = dist[ne]
                        # print("Ctr2 is", Ctr2)
                        # 从n到ne的路线
                        L2 = []
                        u = ne
                        while math.isnan(prev[u]) is False:
                            L2.insert(0, u)
                            u = prev[u]
                        # print("从n到ne的路线为：", L2)

                        # print("选择的WCL：", (Mch[action], Nch[action]))
                        # print("Travel Cost:", Ctr1 + Ctr2)
                        # print("Charging Cost:", info["Cch_mn"])

                        # Ctr_mn = alpha * rho_ave * G_length_adj_matrix[Mch[action], Nch[action]]
                        # print("Ctr_mn is:", Ctr_mn)

                        Obj = Ctr1 + info["Cch_mn"] + Ctr2
                        # print("总obj为：", Obj)
                        print('The chosen WCL is:', (Mch[action], Nch[action]), 'The optimal value of the obj. is', Obj)

                    else:  # info为空：结果异常
                        Obj = np.nan
                        print('出现异常值')

                    Obj_list.append(Obj)

                cost[k] = np.nanmean(Obj_list)
                print("平均cost:", cost[k])

                writer.writerow([ns, ne, cost[k]])

                k = k + 1





