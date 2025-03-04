import pandas as pd  # read file
import networkx as nx
import numpy as np
import scipy.stats as stats
from Dijkstra import Graph
import math
from stable_baselines3.common.utils import set_random_seed


def paramInit(df=None, WCL=None, rho_ave=0.7673, q_t=25, alpha=0.2, rho_ch_mu=None, speedWCL_mu=None):
    # 随机生成各DWER充电价格（截断正态分布）
    if df is None:
        df = pd.read_excel('BeijingTrafficNetwork.xlsx')
    if WCL is None:
        WCL = [(1, 13), (13, 1), (12, 23), (23, 12), (9, 21), (21, 9), (26, 27), (27, 26), (37, 38), (38, 37), (6, 7),
               (7, 6)]
    if rho_ch_mu is None:
        rho_ch_mu = [0.4, 0.4, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.3, 0.3]
    if speedWCL_mu is None:
        speedWCL_mu = [32, 32, 27, 27, 34, 34, 30, 30, 23, 23, 36, 36]


    # GPL数据
    # 随机生成各条路段行驶速度（截断正态分布），计算各条路段等效权重
    G_length = nx.Graph()
    G_velocity = nx.Graph()
    G_eqv = nx.Graph()
    for index, row in df.iterrows():
        source_node = int(row['From'])
        target_node = int(row['To'])
        length = row['length (km)']  # 对距离取整
        G_length.add_edge(source_node, target_node, weight=length)

        speed_limit = row['speed limit (km/h)']
        mu = 0.9 * speed_limit
        sigma = 0.05 * speed_limit
        lower = mu - 2 * sigma
        upper = mu + 2 * sigma
        velocity = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        velocity_s = velocity.rvs(1)[0]  # 用截断正态分布随机生成每条路段的“最高行驶车速”vij，sample size=1
        # print(velocity_s)
        G_velocity.add_edge(source_node, target_node, weight=velocity_s)
        weight_eqv = alpha * rho_ave * length + q_t * length / velocity_s  # 计算等效边权重
        G_eqv.add_edge(source_node, target_node, weight=weight_eqv)

    G_length_adj_matrix = nx.to_numpy_matrix(G_length, nodelist=list(range(0, 41)))
    G_velocity_adj_matrix = nx.to_numpy_matrix(G_velocity, nodelist=list(range(0, 41)))
    G_eqv_adj_matrix = nx.to_numpy_matrix(G_eqv, nodelist=list(range(0, 41)))

    # WCL数据
    rho_ch_s = {}  # CNY/kwh 各WCL实时充电价格
    Vmn = {}  # 各WCL上行驶速度

    for k in range(12):
        mu = rho_ch_mu[k]
        sigma = 0.15 * mu
        lower = mu - 2 * sigma
        upper = mu + 2 * sigma
        rho_ch = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        rho_ch_s[WCL[k]] = rho_ch.rvs(1)[0]

        speedWCL_sigma = 0.15 * speedWCL_mu[k]
        Vmn_lower = 0
        Vmn_upper = 40
        Vmn_dist = stats.truncnorm((Vmn_lower - speedWCL_mu[k]) / speedWCL_sigma,
                                   (Vmn_upper - speedWCL_mu[k]) / speedWCL_sigma,
                                   loc=speedWCL_mu[k], scale=speedWCL_sigma)
        Vmn[WCL[k]] = Vmn_dist.rvs(1)[0]

    return G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix


def getFeatures(
        rho_ch_s,
        Vmn,
        G_length_adj_matrix,
        G_eqv_adj_matrix,
        Mch=None,
        Nch=None,
        WCL=None,
        beta=1,  # 对WCL拥堵的偏好（取值范围-1~1，大于0偏好拥堵，小于0回避拥堵）
        Vmin=20,  # WCL不拥堵时的最低速度
        alpha=0.2,
        p_ch=30,
        start_node=1,
        end_node=5,
        et=40,
        emax=80
):
    if Mch is None:
        Mch = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
    if Nch is None:
        Nch = [13, 1, 23, 12, 21, 9, 27, 26, 38, 37, 7, 6]
    if WCL is None:
        WCL = [(Mch[i], Nch[i]) for i in range(12)]

    g = Graph(41)
    g.graph = G_eqv_adj_matrix

    dist1, prev1 = g.dijkstra(start_node)
    dist1_to_Mch = [dist1[m] for m in Mch]  # start node 到所有WCL起点的距离

    dist2, prev2 = g.dijkstra(end_node)
    dist2_to_Nch = [dist2[n] for n in Nch]  # 所有WCL终点到end node的距离

    Ctr = {}  # travel cost regarding WCL k
    L = {}  # the optimal path from the current location to WCL k
    em = {}  # 到达WCL起点时的电量
    ech_max = {}  # 每个WCL最大充电量
    ech = {}  # 实际充电量
    Cch = {}  # charging cost regarding WCL k
    for k in range(12):
        Ctr_ndarray = dist1_to_Mch[k] + dist2_to_Nch[k]
        Ctr[WCL[k]] = float(Ctr_ndarray)
        '''print('Ctr类型', type(Ctr[WCL[k]]))'''

        L[Mch[k]] = []
        u = Mch[k]
        while math.isnan(prev1[u]) is False:
            L[Mch[k]].insert(0, u)
            u = prev1[u]

        if L[Mch[k]]:
            em[Mch[k]] = et - alpha * sum(
                [G_length_adj_matrix[L[Mch[k]][i], L[Mch[k]][i + 1]] for i in range(len(L[Mch[k]]) - 1)]
                , G_length_adj_matrix[start_node, L[Mch[k]][0]])
        else:
            em[Mch[k]] = et

        ech_max[WCL[k]] = G_length_adj_matrix[Mch[k], Nch[k]] / Vmn[WCL[k]] * p_ch
        ech[WCL[k]] = min(emax - em[Mch[k]], ech_max[WCL[k]])

        # 计算充电车道的拥堵程度C
        if Vmn[WCL[k]] < Vmin:
            C = 1 - Vmn[WCL[k]] / Vmin
        else:
            C = 0

        Cch_ndarray = rho_ch_s[WCL[k]] * ech[WCL[k]] * (1 - beta * C)
        Cch[WCL[k]] = float(Cch_ndarray)
        '''print('Cch类型', type(Cch[WCL[k]]))'''

    '''print("充电量", ech)'''

    Obj = {}
    for k in WCL:
        Obj[k] = Ctr[k] + Cch[k]

    # 选择的最优WCL，以及对应最优目标函数
    WCL_op = min(Obj, key=Obj.get)
    Obj_op = Obj[WCL_op]

    '''print('The chosen WCL is:', WCL_op, 'The optimal value of the obj. is', Obj_op)

    print('Ctr is', Ctr)
    print('The optimal path is', L)

    print('Cch is', Cch)'''

    return Ctr, L, Cch


if __name__ == "__main__":
    '''G_length = nx.Graph()
    G_velocity = nx.Graph()
    G_eqv = nx.Graph()'''
    set_random_seed(100)
    loc = 0
    G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix = paramInit()
    print("路长矩阵", G_length_adj_matrix)
    print("WCL速度是", Vmn)
    print("充电价格是", rho_ch_s)
    Ctr, L, Cch = getFeatures(
        rho_ch_s,
        Vmn,
        G_length_adj_matrix,
        G_eqv_adj_matrix,
        start_node=loc)
    print("Ctr is", Ctr)
    print("L is", L)
    print("Cch is", Cch)

    L2 = np.empty([12, 2])
    idx = 0
    for key in L.keys():
        if len(L[key]) >= 2:
            L2[idx, 0] = L[key][0]
            L2[idx, 1] = L[key][1]
        elif len(L[key]) == 1:
            L2[idx, 0] = L[key][0]
            L2[idx, 1] = L2[idx, 0]
        else:
            L2[idx] = [loc, loc]
        idx = idx + 1

    print("L2为", L2)
