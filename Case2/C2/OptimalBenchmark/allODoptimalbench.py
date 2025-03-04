import pandas as pd
import numpy as np
import scipy.stats as stats
from gurobipy import *
import csv

'''
本脚本直接用Gurobi求解P1，作为Optimal Benchmark
'''

df = pd.read_excel('BeijingTrafficNetwork_T2.xlsx')

start_node = df['From'].tolist()
end_node = df['To'].tolist()
Length = df['length (km)'].tolist()
speed_limit_old = df['speed limit (km/h)']

arcs = []
for i in df.index:
    arcs.append((start_node[i], end_node[i]))
    arcs.append((end_node[i], start_node[i]))

# print(arcs)
# 常参数
alpha = 0.15
rho_ave = 0.7673
q_t = 25
p_ch = 30
emax = 80
emin = 0
Vmax = 40
Vmin = 20
beta = 1

WCLs = [(1, 13), (13, 1), (12, 23), (23, 12), (9, 21), (21, 9), (26, 27), (27, 26), (37, 38), (38, 37), (6, 7),
        (7, 6)]
Mch = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
Nch = [13, 1, 23, 12, 21, 9, 27, 26, 38, 37, 7, 6]
rho_ch_mu = [0.4, 0.4, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.3, 0.3]
speedWCL_mu = [32, 32, 27, 27, 34, 34, 30, 30, 23, 23, 36, 36]

nodes = list(range(41))
chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
comNodes = [node for node in nodes if node not in chgNodes]  # 非充电节点

n_comNodes = len(comNodes)
startLoc = np.empty(n_comNodes * (n_comNodes - 1))
endLoc = np.empty(n_comNodes * (n_comNodes - 1))
cost = np.empty(n_comNodes * (n_comNodes - 1))

with open('data_OB_T2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['StartLoc', 'EndLoc', 'Cost'])

    # generate and write data
    k = 0
    for ns in comNodes:
        for ne in comNodes:
            if ns != ne:
                startLoc[k] = ns
                endLoc[k] = ne

                print('起点和终点分别为：', ns, ne)

                sumObj = 0

                for seed in range(100):

                    np.random.seed(seed)

                    SOC = np.array([np.random.uniform(low=0.4, high=0.6, )])
                    e_ini = emax * SOC[0]
                    print("初始SOC：", SOC[0])

                    # 生成GPL数据
                    length = {}
                    Vij = {}
                    # speed_limit = speed_limit_old   # 赋初值
                    for i in df.index:
                        length[arcs[2 * i]] = Length[i]
                        length[arcs[2 * i + 1]] = length[arcs[2 * i]]

                        speed_limit = speed_limit_old[i]

                        mu = 0.9 * speed_limit
                        sigma = 0.05 * speed_limit
                        lower = mu - 2 * sigma
                        upper = mu + 2 * sigma
                        velocity = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        # 使用随机数
                        Vij[arcs[2 * i]] = velocity.rvs(1)[0]
                        Vij[arcs[2 * i + 1]] = Vij[arcs[2 * i]]  # 正反向车道速度相等

                    # 生成WCL数据
                    rho_ch_s = {}  # CNY/kwh 各WCL实时充电价格
                    Vmn = {}  # 各WCL上行驶速度
                    C = {}  # 道路拥堵程度
                    ech_max = {}  # 各WCL上最大充电量

                    for k in range(12):
                        mu = rho_ch_mu[k]
                        sigma = 0.15 * mu
                        lower = mu - 2 * sigma
                        upper = mu + 2 * sigma
                        rho_ch = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        rho_ch_s[WCLs[k]] = rho_ch.rvs(1)[0]

                        speedWCL_sigma = 0.15 * speedWCL_mu[k]
                        Vmn_lower = 0
                        Vmn_upper = 40
                        Vmn_dist = stats.truncnorm((Vmn_lower - speedWCL_mu[k]) / speedWCL_sigma,
                                                   (Vmn_upper - speedWCL_mu[k]) / speedWCL_sigma,
                                                   loc=speedWCL_mu[k], scale=speedWCL_sigma)
                        Vmn[WCLs[k]] = Vmn_dist.rvs(1)[0]

                        # 计算充电车道的拥堵程度C
                        if Vmn[WCLs[k]] < Vmin:
                            C[WCLs[k]] = 1 - Vmn[WCLs[k]] / Vmin
                        else:
                            C[WCLs[k]] = 0

                        ech_max[WCLs[k]] = length[WCLs[k]] / Vmn[WCLs[k]] * p_ch

                    # print('路长', length)
                    # print('GPL速度：', Vij)
                    # print('WCL电价', rho_ch_s)
                    # print('WCL速度', Vmn)
                    # print('拥堵偏好', C)
                    # print('WCL最大充电量', ech_max)

                    # MIP  model formulation
                    m = Model('optimal benchmark')

                    # Add variables
                    x = m.addVars(arcs, vtype=GRB.BINARY, name="xij")
                    y = m.addVars(arcs, vtype=GRB.BINARY, name="yij")
                    z = m.addVars(WCLs, vtype=GRB.BINARY, name="zmn")
                    Em = m.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=-GRB.INFINITY, name="EVenergy")
                    ech = m.addVar(vtype=GRB.CONTINUOUS, name="ChargingEnergy")
                    Travel = m.addVar(name="TravelCost")
                    Charge = m.addVar(ub=GRB.INFINITY, lb=-GRB.INFINITY, name="ChargeCost")
                    delta = m.addVar(vtype=GRB.BINARY, name="AuxiliaryVariables")

                    # Set objective
                    m.setObjective(Travel + Charge, GRB.MINIMIZE)

                    # Add constraints
                    m.addConstr(Travel == alpha * rho_ave * quicksum(length[ar] * (x[ar] + y[ar]) for ar in arcs)
                                + q_t * quicksum(length[ar] / Vij[ar] * (x[ar] + y[ar]) for ar in arcs),
                                name="Travelcost")

                    m.addConstr(Charge == ech * quicksum(rho_ch_s[WCL] * z[WCL] * (1 - beta * C[WCL]) for WCL in WCLs),
                                name="Chargecost")

                    m.addConstr(Em == e_ini - alpha * quicksum(length[ar] * x[ar] for ar in arcs),
                                name="EV energy before charging")

                    m.addConstr(Em >= emin, name="min energy before charging")

                    M = 100000  # 大M法处理min约束
                    m.addConstr(ech <= (emax - Em) + M * (1 - delta), name="min-1")
                    m.addConstr(ech >= (emax - Em) - M * (1 - delta), name="min-2")
                    m.addConstr(ech <= quicksum(ech_max[WCL] * z[WCL] for WCL in WCLs) + M * delta, name="min-3")
                    m.addConstr(ech >= quicksum(ech_max[WCL] * z[WCL] for WCL in WCLs) - M * delta, name="min-4")
                    m.addConstr(quicksum(ech_max[WCL] * z[WCL] for WCL in WCLs) <= (emax - Em) + M * delta,
                                name="min-5")
                    m.addConstr((emax - Em) <= quicksum(ech_max[WCL] * z[WCL] for WCL in WCLs) + M * (1 - delta),
                                name="min-6")

                    nodes_x = [node for node in nodes if node not in Mch]
                    nodes_x.remove(ns)

                    m.addConstr(x.sum(ns, '*') - x.sum('*', ns) == 1, name="Route Continuity for x-1")
                    m.addConstrs((x.sum(WCL[0], '*') - x.sum('*', WCL[0]) == -z[WCL] for WCL in WCLs),
                                 name="Route Continuity for x-2")
                    m.addConstrs((x.sum(i, '*') - x.sum('*', i) == 0 for i in nodes_x), name="Route Continuity for x-3")

                    nodes_y = [node for node in nodes if node not in Nch]
                    nodes_y.remove(ne)

                    m.addConstrs((y.sum(WCL[1], '*') - y.sum('*', WCL[1]) == z[WCL] for WCL in WCLs),
                                 name="Route Continuity for y-1")
                    m.addConstr(y.sum(ne, '*') - y.sum('*', ne) == -1, name="Route Continuity for y-2")
                    m.addConstrs((y.sum(i, '*') - y.sum('*', i) == 0 for i in nodes_y), name="Route Continuity for y-3")

                    m.addConstr(quicksum(z[WCL] for WCL in WCLs) == 1, name="Choose one WCL")

                    m.update()
                    m.setParam('OutputFlag', 0)
                    # m.write('optimal_benchmark.lp')
                    m.optimize()

                    Obj_op = m.objVal

                    print("目标函数最优值:", Obj_op)

                    sumObj = sumObj + Obj_op

                cost[k] = sumObj / 100
                print("平均cost:", cost[k])

                writer.writerow([ns, ne, cost[k]])

                k = k + 1
