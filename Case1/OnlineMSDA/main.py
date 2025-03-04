import numpy as np
import utils

# 根据需要main.py中p_ch的取值

n_DWERs = 12
n_nodes = 41
chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
allNodes = list(range(41))
comNodes = [node for node in allNodes if node not in chgNodes]  # 非充电节点
DWER = [(chgNodes[2 * i], chgNodes[2 * i + 1]) for i in range(6)]
emax = 80  # 电池最大容量
alpha = 0.15  # kwh/km 每km耗电量
p_ch = 30  # kw wireless charging power

cost = [0] * 10000
# total_cost = 0


for k in range(10000):
    # np.random.seed(66)
    startLoc = int(np.random.choice(comNodes))
    # We will sample the target's location randomly until it does not coincide with the agent's location
    endLoc = startLoc
    while np.array_equal(endLoc, startLoc):
        endLoc = int(np.array(np.random.choice(comNodes)))
    print('第', k, '轮')
    print('起点和终点分别为：', startLoc, endLoc)
    DWER_op = DWER[0]
    # et = 40
    et = np.random.uniform(low=0.4, high=0.6, )

    while not np.array_equal(startLoc, DWER_op[0]):
        # 随机生成车速和充电费用
        rho_ch_s, Vmn, G_velocity_adj_matrix, G_length_adj_matrix, G_eqv_adj_matrix = utils.paramInit()
        # print('WCL电价', rho_ch_s)
        # print('WCL速度', Vmn)
        DWER_op, L_op, oneStepCtr, Cch, Ctr = utils.getFeatures(
            rho_ch_s,
            Vmn,
            G_length_adj_matrix,
            G_eqv_adj_matrix,
            start_node=startLoc,
            end_node=endLoc,
            et=et
        )
        # print("单步行驶cost为", oneStepCtr)
        cost[k] = cost[k] + oneStepCtr
        # total_cost = total_cost + oneStepCtr

        if len(L_op):  # 判断L_op不为空
            et = et - alpha * G_length_adj_matrix[startLoc, L_op[0]]
            startLoc = L_op[0]

        #  elif np.array_equal(startLoc, L_op[0]):
        #    et = et
        else:
            et = et

        # print('EV位置为', startLoc, '选择的充电节点为', DWER_op[0])

    # print("充电cost为", Cch[DWER_op])

    Ctr2 = Ctr[DWER_op]
    # print("剩余travel cost为", Ctr2)

    # total_cost = total_cost + Cch[DWER_op] + Ctr2

    cost[k] = cost[k] + Cch[DWER_op] + Ctr2
    print("Total cost为", cost[k])


mean_cost = np.mean(cost)
std_cost = np.std(cost)
# print('The costs are:', cost)
print(f"mean_cost:{mean_cost:.2f} +/- {std_cost:.2f}")
