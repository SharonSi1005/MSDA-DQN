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

    def __init__(self, n_DWERs=12, n_nodes=41, chgNodes=None, allNodes=None, df=None,
                 error_probability=0, error_percentage=0):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.n_DWERs = n_DWERs
        self.n_nodes = n_nodes
        if chgNodes is None:
            chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
        self.chgNodes = chgNodes
        if allNodes is None:
            allNodes = list(range(self.n_nodes))
        self.allNodes = allNodes
        if df is None:
            df = pd.read_excel('BeijingTrafficNetwork.xlsx')
        self.df = df
        self.comNodes = [node for node in self.allNodes if node not in self.chgNodes]  # 非充电节点
        self.emax = 80  # 电池最大容量
        self.alpha = 0.15  # kwh/km 每km耗电量
        self.p_ch = 30  # kw wireless charging power
        self.error_probability = error_probability
        self.error_percentage = error_percentage

        # initialize obs
        self._startLoc = 0
        self._endLoc = self._startLoc
        self._loc = self._startLoc
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
        self.action_space = Discrete(self.n_DWERs)  # {0,1,2,...,11}

        # 状态空间7类
        Lk = np.array([[self.n_nodes, self.n_nodes]])  # 取最优路径的前两段
        self.observation_space = Dict({"loc": Discrete(self.n_nodes), "SOC": Box(low=0, high=1, dtype=np.float64),
                                       "curTime": Box(low=0, high=float('inf'), dtype=np.float64),
                                       "velocity": Box(low=0, high=120, shape=(4,), dtype=np.float64),
                                       "travelCost": Box(low=float('-inf'), high=float('inf'), shape=(self.n_DWERs,),
                                                         dtype=np.float64),
                                       "chgCost": Box(low=float('-inf'), high=float('inf'), shape=(self.n_DWERs,),
                                                      dtype=np.float64),
                                       "optPath": MultiDiscrete(Lk.repeat(self.n_DWERs, axis=0))
                                       })

    def _get_obs(self):

        return {"loc": self._loc, "SOC": self._SOC, "curTime": self._curTime, "velocity": self._velocity,
                "travelCost": self._Ctr, "chgCost": self._Cch, "optPath": self._L2}

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)  # 为self.np_random提供seed
        self._startLoc = int(self.np_random.choice(self.comNodes))
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._endLoc = self._startLoc
        while np.array_equal(self._endLoc, self._startLoc):
            self._endLoc = int(np.array(self.np_random.choice(self.comNodes)))

        # print("start node =", self._startLoc, "; end node =", self._endLoc)

        self._loc = self._startLoc
        self._SOC = np.array([self.np_random.uniform(low=0.4, high=0.6, )])
        self._curTime = np.zeros(1)

        G_velocity_measured, rho_ch_s, Vmn, G_length_adj_matrix, \
            G_velocity_adj_matrix_real, G_velocity_adj_matrix_measured, \
            G_eqv_adj_matrix_real, G_eqv_adj_matrix_measured \
            = utils.paramInit(error_probability=self.error_probability, error_percentage=self.error_percentage)

        global_var._init()
        self.Wij = G_eqv_adj_matrix_real  # 初始化全局变量Wij，保存真值
        global_var.set_value('Wij', self.Wij)

        self._velocity = np.zeros(4)   # observation中的速度为测量值
        idx = 0
        for adjNode in G_velocity_measured.neighbors(self._loc):
            self._velocity[idx] = G_velocity_adj_matrix_measured[self._loc, adjNode]
            idx = idx + 1

        Ctr, L, Cch = utils.getFeatures(
            rho_ch_s,
            Vmn,
            G_length_adj_matrix,
            G_eqv_adj_matrix_measured,
            et=self._SOC * self.emax,
            start_node=self._loc,
            end_node=self._endLoc)

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

    def step(self, action):  # action={0,1,2,...,11}
        G_velocity_measured, rho_ch_s, Vmn, G_length_adj_matrix, \
            G_velocity_adj_matrix_real, G_velocity_adj_matrix_measured, \
            G_eqv_adj_matrix_real, G_eqv_adj_matrix_measured \
            = utils.paramInit(error_probability=self.error_probability, error_percentage=self.error_percentage)

        # MDP process
        lastLoc = self._loc
        self._loc = self._L2[action, 0]

        et = self._SOC * self.emax - self.alpha * G_length_adj_matrix[lastLoc, self._loc]
        self._SOC = et / self.emax

        if lastLoc != self._loc:
            delta_T_hour = G_length_adj_matrix[lastLoc, self._loc] / G_velocity_adj_matrix_real[lastLoc, self._loc]
            self._curTime = self._curTime + delta_T_hour

        self._velocity = np.zeros(4)
        idx = 0
        for adjNode in G_velocity_measured.neighbors(self._loc):
            self._velocity[idx] = G_velocity_adj_matrix_measured[self._loc, adjNode]
            idx = idx + 1

        Ctr, L, Cch = utils.getFeatures(
            rho_ch_s,
            Vmn,
            G_length_adj_matrix,
            G_eqv_adj_matrix_measured,
            et=et,
            start_node=self._loc,
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
                self._L2[idx] = [self._loc, self._loc]
            idx = idx + 1

        observation = self._get_obs()

        # print("当前位置为：", self._loc)
        # print("选择的充电节点为：", self.chgNodes[action])
        terminated = np.array_equal(self._loc, self.chgNodes[action])

        self.Wij = global_var.get_value('Wij')

        if terminated:
            Cch = self._Cch[action]
            '''print("充电cost为", Cch)'''

            Ctr2 = self._Ctr[action]
            '''print("行驶cost为", Ctr2)'''

            if self._SOC > 0:
                reward = - self.Wij[lastLoc, self._loc] - Cch - Ctr2
            else:  # 到达WCL使电量小于0
                reward = -100
        else:
            reward = -self.Wij[lastLoc, self._loc]  # reward用上一时刻的G_eqv_adj_matrix_real算

        self.Wij = G_eqv_adj_matrix_real  # 更新t+1时刻
        global_var.set_value('Wij', self.Wij)

        truncated = False  # we do not limit the number of steps here

        info = {}

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    np.random.seed(42)
    env = DynChgEnv()
    #  env = FlattenObservation(env)
    check_env(env, warn=True)
    obs, _ = env.reset()
    print("Initial obs =", obs)
    '''print("obs space shape", env.observation_space.shape)'''

    n_steps = 10

    for step in range(n_steps):
        print(f"Step {step + 1}")
        action = 0
        # action = np.random.choice(range(env.n_DWERs))
        print("action = ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("The", step + 1, "step reward is", reward)
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            print("Goal reached!", "reward=", reward)
            break
