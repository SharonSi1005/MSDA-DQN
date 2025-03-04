import pandas as pd  # read file
import networkx as nx
import numpy as np
import scipy.stats as stats
from Dijkstra import Graph
import math
from stable_baselines3.common.utils import set_random_seed
import getSpeed
import getPrice


def paramInit(length_file_path='chaoyang_edge_whole.csv', speed_file_path="linkID_time_speed_12-7.csv",
              price_file_path="charging_prices_processed.csv", time=0, rho_ave=0.7673, q_t=25, alpha=0.15):
    """
    从chaoyang_edge_whole.csv，linkID_time_speed.csv，charging_prices_processed.csv文件中得到格式化的输入参数
    :param price_file_path:
    :param length_file_path:
    :param speed_file_path:
    :param time:
    :param rho_ave:
    :param q_t:
    :param alpha:
    :return:
    """

    # GPL数据
    length_data = pd.read_csv(length_file_path)
    speed_data = pd.read_csv(speed_file_path)

    # 创建速度插值函数
    speed_functions = getSpeed.create_speed_functions(speed_data)

    # 初始化有向图
    G_length = nx.DiGraph()
    G_speed = nx.DiGraph()
    G_eqv = nx.DiGraph()

    # 遍历每条边，添加到图中
    for _, row in length_data.iterrows():
        linkID = row['LinkID']
        from_node = row['FromNode']
        to_node = row['ToNode']
        length = row['Length']
        oneway = row['Oneway']

        # 添加单向边
        G_length.add_edge(from_node, to_node, weight=length)  # length

        speed = getSpeed.get_speed(speed_functions, linkID, time)
        G_speed.add_edge(from_node, to_node, weight=speed)  # speed

        weight_eqv = alpha * rho_ave * length + q_t * length / speed  # 计算等效边权重
        G_eqv.add_edge(from_node, to_node, weight=weight_eqv)  # equivalent

        # 如果不是单向（Oneway != 'T'），添加逆向边
        if oneway != 'T':
            G_length.add_edge(to_node, from_node, weight=length)
            G_speed.add_edge(to_node, from_node, weight=speed)
            G_eqv.add_edge(to_node, from_node, weight=weight_eqv)

    # 处理G_length
    # 找到所有强连通分量
    strongly_connected_components = list(nx.strongly_connected_components(G_length))
    # 找到最大的强连通子图
    largest_scc = max(strongly_connected_components, key=len)
    # 创建包含最大强连通分量的子图
    G_largest = G_length.subgraph(largest_scc).copy()

    # # 输出节点和边的数量
    # num_nodes = G_largest.number_of_nodes()  # 获取节点数量
    # num_edges = G_largest.number_of_edges()  # 获取边数量
    #
    # print(f"G_length 的节点数量: {num_nodes}")
    # print(f"G_length 的边数量: {num_edges}")

    # 替换原图为处理后的图
    G_length = G_largest

    # 处理G_speed
    # 找到所有强连通分量
    strongly_connected_components = list(nx.strongly_connected_components(G_speed))
    # 找到最大的强连通子图
    largest_scc = max(strongly_connected_components, key=len)
    # 创建包含最大强连通分量的子图
    G_largest = G_speed.subgraph(largest_scc).copy()
    # 替换原图为处理后的图
    G_speed = G_largest

    # 处理G_eqv
    # 找到所有强连通分量
    strongly_connected_components = list(nx.strongly_connected_components(G_eqv))
    # 找到最大的强连通子图
    largest_scc = max(strongly_connected_components, key=len)
    # 创建包含最大强连通分量的子图
    G_largest = G_eqv.subgraph(largest_scc).copy()
    # 替换原图为处理后的图
    G_eqv = G_largest

    # 获取节点列表并升序排序
    sorted_nodes = sorted(G_length.nodes())

    # 将图的邻接矩阵导出为 NumPy 矩阵
    G_length_adj_matrix = nx.to_numpy_matrix(G_length, nodelist=sorted_nodes)
    G_speed_adj_matrix = nx.to_numpy_matrix(G_speed, nodelist=sorted_nodes)
    G_eqv_adj_matrix = nx.to_numpy_matrix(G_eqv, nodelist=sorted_nodes)

    #  WCL数据
    prices_data = pd.read_csv(price_file_path)
    unique_ids = prices_data['ID'].unique().tolist()

    # 创建结果列表
    WCL = []  # 存储充电车道元组
    rho_ch_s = {}  # CNY/kwh 各WCL实时充电价格
    Vmn = {}  # 各WCL上行驶速度

    # 将 edges_data 转换为字典 {LinkID: (FromNode, ToNode, Oneway)}
    edges_dict = length_data.set_index('LinkID')[['FromNode', 'ToNode', 'Oneway']].to_dict('index')

    for link_id in unique_ids:
        from_node = edges_dict[link_id]['FromNode']
        to_node = edges_dict[link_id]['ToNode']
        oneway = edges_dict[link_id]['Oneway']

        # 添加单向边
        WCL.append((from_node, to_node))
        price = getPrice.get_price(link_id, time)
        rho_ch_s[(from_node, to_node)] = price

        speed = getSpeed.get_speed(speed_functions, link_id, time)
        Vmn[(from_node, to_node)] = speed

        # 如果是双向车道，添加反向边
        if oneway != 'T':
            WCL.append((to_node, from_node))
            rho_ch_s[(to_node, from_node)] = price
            Vmn[(to_node, from_node)] = speed

    return sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix


def getFeatures(
        sorted_nodes,
        WCL,
        G_eqv,
        rho_ch_s,
        Vmn,
        G_length_adj_matrix,
        G_eqv_adj_matrix,
        beta=1,  # 对WCL拥堵的偏好（取值范围-1~1，大于0偏好拥堵，小于0回避拥堵）
        Vmin=30,  # WCL不拥堵时的最低速度
        alpha=0.15,
        p_ch=100,
        start_node=6773,
        end_node=1525,
        et=40,
        emax=80
):
    Mch = [lane[0] for lane in WCL]  # 提取 FromNode
    Nch = [lane[1] for lane in WCL]  # 提取 ToNode

    g = Graph(len(sorted_nodes))
    g.graph = G_eqv_adj_matrix

    # 转换为邻接矩阵中的index（范围0~517）
    start_node_index = sorted_nodes.index(start_node)
    end_node_index = sorted_nodes.index(end_node)
    # print("New start node =", start_node_index, "; end node =", end_node_index)

    dist1, prev1 = g.dijkstra(start_node_index)
    dist1_to_Mch = [dist1[sorted_nodes.index(m)] for m in Mch]  # start node 到所有WCL起点的距离
    # print("到WCL起点的Ctr", dist1_to_Mch)
    # dist1_op = min(dist1_to_Mch)
    # m_op = dist1.index(dist1_op)

    reversed_graph = G_eqv.reverse(copy=True)
    # dist2 = nx.single_source_dijkstra_path_length(reversed_graph, end_node)
    dist2, paths = nx.single_source_dijkstra(reversed_graph, source=end_node)

    dist2_to_Nch = [dist2[n] for n in Nch]  # 所有WCL终点到end node的距离
    # print("到WCL终点的Ctr", dist2_to_Nch)
    # dist2_op = min(dist2_to_Nch)
    # n_op = dist2.index(dist2_op)

    Ctr = {}  # travel cost regarding WCL k
    L = {}  # the optimal path from the current location to WCL k
    em = {}  # 到达WCL起点时的电量
    ech_max = {}  # 每个WCL最大充电量
    ech = {}  # 实际充电量
    Cch = {}  # charging cost regarding WCL k
    # L_rev = {}  # n到end node的路径

    n_WCLs = len(WCL)  # WCL总数

    for k in range(n_WCLs):
        Ctr_ndarray = dist1_to_Mch[k] + dist2_to_Nch[k]
        Ctr[WCL[k]] = float(Ctr_ndarray)
        '''print('Ctr类型', type(Ctr[WCL[k]]))'''

        L[Mch[k]] = []
        u = sorted_nodes.index(Mch[k])
        while math.isnan(prev1[u]) is False:
            L[Mch[k]].insert(0, u)
            u = prev1[u]

        # n到end node的路径
        # L_rev[Mch[k]] = [sorted_nodes.index(i) for i in paths[Nch[k]]]

        if L[Mch[k]]:
            em[Mch[k]] = et - alpha * sum(
                [G_length_adj_matrix[L[Mch[k]][i], L[Mch[k]][i + 1]] for i in range(len(L[Mch[k]]) - 1)]
                , G_length_adj_matrix[start_node_index, L[Mch[k]][0]])
        else:

            em[Mch[k]] = et

        ech_max[WCL[k]] = G_length_adj_matrix[sorted_nodes.index(Mch[k]), sorted_nodes.index(Nch[k])] / Vmn[
            WCL[k]] * p_ch
        ech[WCL[k]] = min(emax - em[Mch[k]], ech_max[WCL[k]])

        # 计算充电车道的拥堵程度C
        if Vmn[WCL[k]] < Vmin:
            C = 1 - Vmn[WCL[k]] / Vmin
        else:
            C = 0

        Cch_ndarray = rho_ch_s[WCL[k]] * ech[WCL[k]] * (1 - beta * C)
        Cch[WCL[k]] = float(Cch_ndarray)
        '''print('Cch类型', type(Cch[WCL[k]]))'''

    # print("充电量", ech)

    Obj = {}
    for k in WCL:
        Obj[k] = Ctr[k] + Cch[k]

    # 选择的最优WCL，以及对应最优目标函数
    WCL_op = min(Obj, key=Obj.get)
    Obj_op = Obj[WCL_op]

    # print('The chosen WCL is:', WCL_op, 'The optimal value of the obj. is', Obj_op)
    #
    # print('Ctr is:', Ctr)
    # print('L is:', L)
    # print('L_rev is', L_rev)
    #
    # print('Cch is:', Cch)

    return Ctr, L, Cch


if __name__ == "__main__":
    speed_file_path = "linkID_time_speed_12-11.csv"
    curTime = np.array([201])
    sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix = \
        paramInit(speed_file_path=speed_file_path, time=int(curTime[0]))
    print("sorted_nodes:", sorted_nodes)
    print("WCL:", WCL)
    print("路长矩阵:", G_length_adj_matrix)
    print("WCL速度:", Vmn)
    print("充电价格:", rho_ch_s)
    print("节点27的原始ID：", sorted_nodes[27])
    print("节点129的原始ID：", sorted_nodes[129])
    print("(27,129)速度为：", G_speed_adj_matrix[27, 129])

    # 指定要查询的边
    from_node = 27
    to_node = 129

    # 检查边是否存在并获取其权重
    if G_eqv.has_edge(from_node, to_node):
        weight = G_eqv[from_node][to_node].get('weight', None)
        print(f"边 ({from_node}, {to_node}) 的权重为: {weight}")
    else:
        print(f"边 ({from_node}, {to_node}) 不存在。")
    # print("节点393的原索引是：", sorted_nodes[393])

    csv_filename = 'G_eqv_adj_matrix.csv'
    np.savetxt(csv_filename, G_eqv_adj_matrix, delimiter=',', fmt='%.6f')

    print(f"邻接矩阵已保存为 {csv_filename}")

    Ctr, L, Cch = getFeatures(
        sorted_nodes,
        WCL,
        G_eqv,
        rho_ch_s,
        Vmn,
        G_length_adj_matrix,
        G_eqv_adj_matrix,
        start_node=sorted_nodes[27])

    print("Ctr is", Ctr)
    print("L is", L)
    print("Cch is", Cch)

    '''
    L2 = np.empty([19, 2])
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
'''
