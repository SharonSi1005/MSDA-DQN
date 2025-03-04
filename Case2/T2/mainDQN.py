from CustomEnv import DynChgEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
import numpy as np
import utils
import math
import global_var
from Dijkstra import Graph

'''
测试不同的电池每公里耗电量alpha
需修改的参数：
1. utils.paramInit()和utils.getFeatures()的alpha
2. CustomEnv的self.alpha
3. 本文件alpha
'''

np.random.seed(42)

model = DQN.load(".../best_model.zip")  # 加载训练好的模型

n_steps = 100
n_DWERs = 12
n_nodes = 41
chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Mch = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Nch = [13, 1, 23, 12, 21, 9, 27, 26, 38, 37, 7, 6]  # WCL终点
allNodes = list(range(41))
comNodes = [node for node in allNodes if node not in chgNodes]  # 非充电节点
emax = 80  # 电池最大容量
alpha = 0.1  # kwh/km 每km耗电量
rho_ave = 0.7673

startLoc = 4
endLoc = 30
'''
startLoc = int(np.random.choice(comNodes))
endLoc = startLoc
while np.array_equal(endLoc, startLoc):
    endLoc = int(np.random.choice(comNodes))
'''
print("start node =", startLoc, "; end node =", endLoc)

loc = startLoc
SOC = np.array([np.random.uniform(low=0.4, high=0.6, )])
# SOC = np.array([0.5], dtype=np.float64)  # 给定初始SOC为40kWh
curTime = np.zeros(1)
velocity = np.zeros(4)

G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix \
    = utils.paramInit()  # 第一次用到random seed
print("WCL电价：", rho_ch_s)
print("WCL速度：", Vmn)

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
    end_node=endLoc)

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
print("Initial obs_dict =", obs_dict)

env = DynChgEnv(G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix)
env.set_obs(obs_dict, endLoc)  # 令self._loc = loc

# env = TimeLimit(env, max_episode_steps=20)
env = FlattenObservation(env)
obs = FlattenObservation.observation(self=env, observation=obs_dict)  # 把dict型flatten
print("Initial obs =", obs)

reward1 = []
for step in range(n_steps):
    print(f"Step {step + 1}")
    action_tuple = model.predict(obs, deterministic=True)  # 由model.predict给定
    action = int(action_tuple[0])
    print("action = ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    print("The", step + 1, "step reward is", reward)
    reward1.append(reward)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    if done:
        print("Goal reached!", "info=", info)
        break

print("从起点到m的reward：", reward1)
print("Cch1_lastStep is:", info["Cch1_lastStep"])
Ctr1 = - sum(reward1[0:-1]) + info["Cch1_lastStep"]  # 除节点m外reward求和，取相反数
print("Ctr1 is:", Ctr1)

g = Graph(41)
g.graph = G_eqv_adj_matrix

dist, prev = g.dijkstra(Nch[action])

Ctr2 = dist[endLoc]
print("Ctr2 is", Ctr2)
# 从n到ne的路线
L2 = []
u = endLoc
while math.isnan(prev[u]) is False:
    L2.insert(0, u)
    u = prev[u]
print("从n到ne的路线为：", L2)

print("选择的WCL：", (Mch[action], Nch[action]))
print("Travel Cost:", Ctr1+Ctr2)
print("Charging Cost:", info["Cch_mn"])

# Ctr_mn = alpha * rho_ave * G_length_adj_matrix[Mch[action], Nch[action]]
# print("Ctr_mn is:", Ctr_mn)

Obj = Ctr1 + info["Cch_mn"] + Ctr2
print("总obj为：", Obj)
