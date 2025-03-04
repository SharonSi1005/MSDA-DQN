import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete, MultiDiscrete
import utils
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
import global_var


class DynChgEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, sorted_nodes, WCL, isTesting=False):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.WCL = WCL
        self.n_DWERs = len(WCL)
        self.n_nodes = len(sorted_nodes)
        chgNodes = list(set(node for lane in WCL for node in lane))
        self.chgNodes = chgNodes
        self.allNodes = sorted_nodes
        self.comNodes = [node for node in self.allNodes if node not in self.chgNodes]  # 非充电节点（使用原始索引）
        self.emax = 80  # 电池最大容量
        self.alpha = 0.15  # kwh/km 每km耗电量
        self.p_ch = 30  # kw wireless charging power
        self.speed_file_path = "linkID_time_speed_12-7.csv"
        self.isTesting = isTesting

        # initialize obs
        self._startLoc_NewID = 0
        self._startLoc_OldID = 6773
        self._endLoc = self._startLoc_OldID
        self._loc_oldID = self._startLoc_OldID
        self._loc_newID = self._startLoc_NewID
        self._SOC = np.array([0.5], dtype=np.float64)
        self._curTime = np.zeros(1)
        self._velocity = np.zeros(4)
        self._Ctr = np.zeros(self.n_DWERs)  # 视图对象（不支持索引）→list → array
        self._Cch = np.zeros(self.n_DWERs)
        self._L2 = np.zeros([self.n_DWERs, 2], dtype=int)

        # t时刻的Wij，用于计算t时刻的reward
        self.Wij = np.zeros((self.n_nodes, self.n_nodes))

        # 定义动作空间和状态空间
        # 动作为选择某个DWER
        self.action_space = Discrete(self.n_DWERs)  # {0,1,2,...,18}

        # 状态空间7类
        Lk = np.array([[self.n_nodes, self.n_nodes]])  # 取最优路径的前两段
        self.observation_space = Dict({"loc": Discrete(self.n_nodes), "SOC": Box(low=0, high=1, dtype=np.float64),
                                       "curTime": Box(low=0, high=1440, dtype=np.int64),
                                       "velocity": Box(low=0, high=200, shape=(5,), dtype=np.float64),
                                       "travelCost": Box(low=float('-inf'), high=float('inf'), shape=(self.n_DWERs,),
                                                         dtype=np.float64),
                                       "chgCost": Box(low=float('-inf'), high=float('inf'), shape=(self.n_DWERs,),
                                                      dtype=np.float64),
                                       "optPath": MultiDiscrete(Lk.repeat(self.n_DWERs, axis=0))
                                       })

    def _get_obs(self):

        return {"loc": self._loc_newID, "SOC": self._SOC, "curTime": self._curTime, "velocity": self._velocity,
                "travelCost": self._Ctr, "chgCost": self._Cch, "optPath": self._L2}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 为self.np_random提供seed

        if self.isTesting is True:
            self.speed_file_path = "linkID_time_speed_12-14.csv"
        else:
            random_int = self.np_random.choice([1, 2, 3, 4])

            # 根据随机值设置 speed_file_path
            if random_int == 1:
                self.speed_file_path = "linkID_time_speed_12-7.csv"
            elif random_int == 2:
                self.speed_file_path = "linkID_time_speed_12-8.csv"
            elif random_int == 3:
                self.speed_file_path = "linkID_time_speed_12-10.csv"
            else:  # random_int == 4
                self.speed_file_path = "linkID_time_speed_12-11.csv"

        # print("速度文件：", self.speed_file_path)

        sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix = \
            utils.paramInit(speed_file_path=self.speed_file_path, time=int(self._curTime[0]))

        # self._startLoc_OldID = int(self.np_random.choice(self.comNodes))
        self._startLoc_OldID = 2780
        self._startLoc_NewID = sorted_nodes.index(self._startLoc_OldID)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._endLoc = self._startLoc_OldID
        while np.array_equal(self._endLoc, self._startLoc_OldID):
            self._endLoc = int(np.array(self.np_random.choice(self.comNodes)))

        # print("Old start node =", self._startLoc_OldID, "; end node =", self._endLoc)

        self._loc_oldID = self._startLoc_OldID  # loc是原始ID
        self._loc_newID = self._startLoc_NewID
        self._SOC = np.array([self.np_random.uniform(low=0.4, high=0.6, )])
        self._curTime = np.array([self.np_random.choice(list(range(1440)))])
        # print("curTime is:", self._curTime)

        global_var._init()
        self.Wij = G_eqv_adj_matrix  # 初始化全局变量Wij
        global_var.set_value('Wij', self.Wij)

        self._velocity = np.zeros(5)
        idx = 0
        for adjNode in G_speed.neighbors(self._loc_oldID):
            self._velocity[idx] = G_speed_adj_matrix[self._loc_newID, sorted_nodes.index(adjNode)]
            idx = idx + 1

        Ctr, L, Cch = utils.getFeatures(
            sorted_nodes,
            WCL,
            G_eqv,
            rho_ch_s,
            Vmn,
            G_length_adj_matrix,
            G_eqv_adj_matrix,
            et=self._SOC * self.emax,
            start_node=self._loc_oldID,
            end_node=self._endLoc)  # L为newID

        self._Ctr = np.array(list(Ctr.values()))  # 视图对象（不支持索引）→list → array
        self._Cch = np.array(list(Cch.values()))
        self._L2 = np.empty([self.n_DWERs, 2], dtype=int)
        idx = 0
        for key in L.keys():
            self._L2[idx, 0] = L[key][0]
            if len(L[key]) >= 2:
                self._L2[idx, 1] = L[key][1]
            else:
                self._L2[idx, 1] = self._L2[idx, 0]
            idx = idx + 1

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):  # action={0,1,2,...,18}
        # count = global_var.get_value('count')
        # count = count + 1
        # global_var.set_value('count', count)
        sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix = \
            utils.paramInit(speed_file_path=self.speed_file_path, time=int(self._curTime[0]))

        # MDP process
        lastLoc = self._loc_newID  # lastloc为newID

        self._loc_newID = self._L2[action, 0]
        self._loc_oldID = sorted_nodes[self._loc_newID]

        et = self._SOC * self.emax - self.alpha * G_length_adj_matrix[lastLoc, self._loc_newID]
        self._SOC = et / self.emax

        # 把时间从小时换算分钟，取整后为取余
        if lastLoc != self._loc_newID:
            delta_T_hour = G_length_adj_matrix[lastLoc, self._loc_newID] / G_speed_adj_matrix[lastLoc, self._loc_newID]
            # if np.isnan(delta_T_hour):
            #     print("时间更新出错！")
            #
            #     print("speed_file_path=", self.speed_file_path)
            #     print("time=", int(self._curTime[0]))
            #     print("出错的边为：", (lastLoc, self._loc_newID))
            #     print("endLoc=", self._endLoc)
            #     print("action=", WCL[action])
            #     print("L2=", self._L2)
            #     print("length=", G_length_adj_matrix[lastLoc, self._loc_newID])
            #     print("speed=", G_speed_adj_matrix[lastLoc, self._loc_newID])

            self._curTime = (self._curTime + int(round(delta_T_hour * 60))) % 1440

        self._velocity = np.zeros(5)
        idx = 0
        for adjNode in G_speed.neighbors(self._loc_oldID):
            self._velocity[idx] = G_speed_adj_matrix[self._loc_newID, sorted_nodes.index(adjNode)]
            idx = idx + 1

        Ctr, L, Cch = utils.getFeatures(
            sorted_nodes,
            WCL,
            G_eqv,
            rho_ch_s,
            Vmn,
            G_length_adj_matrix,
            G_eqv_adj_matrix,
            et=et,
            start_node=self._loc_oldID,
            end_node=self._endLoc)

        self._Ctr = np.array(list(Ctr.values()))  # 视图对象（不支持索引）→list → array
        self._Cch = np.array(list(Cch.values()))
        self._L2 = np.empty([self.n_DWERs, 2], dtype=int)
        idx = 0
        for key in L.keys():
            if len(L[key]) >= 2:
                self._L2[idx, 0] = L[key][0]
                self._L2[idx, 1] = L[key][1]
            elif len(L[key]) == 1:
                self._L2[idx, 0] = L[key][0]
                self._L2[idx, 1] = self._L2[idx, 0]
            else:
                self._L2[idx] = [self._loc_newID, self._loc_newID]
            idx = idx + 1

        observation = self._get_obs()

        # print("loc_oldID：", self._loc_oldID)
        # print("充电节点：", self.WCL[action][0])

        terminated = np.array_equal(self._loc_oldID, self.WCL[action][0])

        self.Wij = global_var.get_value('Wij')

        if terminated:
            Cch = self._Cch[action]
            # print("充电cost为", Cch)

            Ctr2 = self._Ctr[action]
            # print("行驶cost为", Ctr2)
            
            if self._SOC > 0:
                reward = - self.Wij[lastLoc, self._loc_newID] - Cch - Ctr2
            else:  # 到达WCL使电量小于0
                reward = -100
        else:
            reward = -self.Wij[lastLoc, self._loc_newID]  # reward用上一时刻的G_eqv_adj_matrix算

        self.Wij = G_eqv_adj_matrix  # 更新t+1时刻
        global_var.set_value('Wij', self.Wij)

        truncated = False  # we do not limit the number of steps here

        info = {}

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    np.random.seed(42)

    sorted_nodes, WCL, G_speed, G_eqv, rho_ch_s, Vmn, G_length_adj_matrix, G_speed_adj_matrix, G_eqv_adj_matrix = \
        utils.paramInit()
    env = DynChgEnv(sorted_nodes, WCL, isTesting=True)
    #  env = FlattenObservation(env)
    check_env(env, warn=True)
    obs, _ = env.reset()
    print("Initial obs =", obs)
    '''print("obs space shape", env.observation_space.shape)'''

    n_steps = 10

    for step in range(n_steps):
        print(f"Step {step + 1}")
        action = 7
        # action = np.random.choice(range(env.n_DWERs))
        print("action = ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("The", step + 1, "step reward is", reward)
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            print("Goal reached!", "reward=", reward)
            break
