import pandas as pd  # read file
import networkx as nx
import numpy as np
import scipy.stats as stats


def paramInit(df=None, DWER=None, rho_ave=0.7673, q_t=30, alpha=0.15, rho_ch_mu=None, speedWCL_mu=None):
    # 随机生成各DWER充电价格（截断正态分布）
    if df is None:
        df = pd.read_excel('BeijingTrafficNetwork.xlsx')
    if DWER is None:
        DWER = [(1, 13), (13, 1), (12, 23), (23, 12), (9, 21), (21, 9), (26, 27), (27, 26), (37, 38), (38, 37), (6, 7),
                (7, 6)]
    if rho_ch_mu is None:
        rho_ch_mu = [0.4, 0.4, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.3, 0.3]
    if speedWCL_mu is None:
        speedWCL_mu = [32, 32, 27, 27, 34, 34, 30, 30, 23, 23, 36, 36]

    # WCL数据
    rho_ch_s = {}  # CNY/kwh 各WCL实时充电价格
    Vmn = {}  # 各WCL上行驶速度

    for k in range(12):
        mu = rho_ch_mu[k]
        sigma = 0.15 * mu
        lower = mu - 2 * sigma
        upper = mu + 2 * sigma
        rho_ch = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        rho_ch_s[DWER[k]] = rho_ch.rvs(1)[0]

        speedWCL_sigma = 0.15 * speedWCL_mu[k]
        Vmn_lower = 0
        Vmn_upper = 40
        Vmn_dist = stats.truncnorm((Vmn_lower - speedWCL_mu[k]) / speedWCL_sigma,
                                   (Vmn_upper - speedWCL_mu[k]) / speedWCL_sigma,
                                   loc=speedWCL_mu[k], scale=speedWCL_sigma)
        Vmn[DWER[k]] = Vmn_dist.rvs(1)[0]

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

    return rho_ch_s, Vmn, G_velocity_adj_matrix, G_length_adj_matrix, G_eqv_adj_matrix


if __name__ == "__main__":
    '''G_length = nx.Graph()
    G_velocity = nx.Graph()
    G_eqv = nx.Graph()'''

    rho_ch_s, Vmn, G_velocity_adj_matrix, G_length_adj_matrix, G_eqv_adj_matrix = paramInit()
    print("车速为：", G_velocity_adj_matrix)
    print("充电价格为：", rho_ch_s)
    print("充电车道速度：", Vmn)

