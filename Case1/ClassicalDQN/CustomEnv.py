import gymnasium as gym
import pandas as pd
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete
import utils
from Dijkstra import Graph
import math

from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation


class DynChgEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_DWERs=12, n_nodes=41, chgNodes=None, allNodes=None, df=None):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.n_DWERs = n_DWERs
        self.n_nodes = n_nodes
        if chgNodes is None:
            chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
        self.chgNodes = chgNodes
        if allNodes is None:
            allNodes = list(range(41))
        self.allNodes = allNodes
        if df is None:
            df = pd.read_excel('BeijingTrafficNetwork.xlsx')
        self.df = df
        self.comNodes = [node for node in self.allNodes if node not in self.chgNodes]  # 非充电节点
        self.emax = 80  # 电池最大容量
        self.alpha = 0.15  # kwh/km 每km耗电量
        self.p_ch = 30  # kw wireless charging power
        self.beta = 1   # 风险偏好

        # 定义动作和状态空间
        # 动作为选择某个DWER
        self.action_space = Discrete(self.n_DWERs)  # {0,1,2,...,11}

        # 状态空间7类

        self.observation_space = Dict({"loc": Discrete(self.n_nodes), "SOC": Box(low=0, high=1, dtype=np.float64),
                                       "curTime": Box(low=0, high=float('inf'), dtype=np.float64),
                                       "all_velocity": Box(low=0, high=120, shape=(self.n_nodes, self.n_nodes),
                                                           dtype=np.float64),
                                       "WCL_velocity": Box(low=0, high=120, shape=(self.n_DWERs,),
                                                           dtype=np.float64),
                                       "chgPrice": Box(low=0, high=float('inf'), shape=(self.n_DWERs,),
                                                       dtype=np.float64)
                                       })

    def _get_obs(self):

        return {"loc": self._loc, "SOC": self._SOC, "curTime": self._curTime, "all_velocity": self._velocity,
                "WCL_velocity": self._Vmn, "chgPrice": self._chgPrice}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 为self.np_random提供seed
        self._startLoc = int(self.np_random.choice(self.comNodes))
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._endLoc = self._startLoc
        while np.array_equal(self._endLoc, self._startLoc):
            self._endLoc = int(np.array(self.np_random.choice(self.comNodes)))

        '''print("start node =", self._startLoc, "; end node =", self._endLoc)'''

        self._loc = self._startLoc
        self._SOC = np.array([self.np_random.uniform(low=0.4, high=0.6, )])
        self._curTime = np.zeros(1)

        rho_ch_s, Vmn, G_velocity_adj_matrix, G_length_adj_matrix, G_eqv_adj_matrix = utils.paramInit(df=self.df)

        self.Wij = G_eqv_adj_matrix
        self._velocity = G_velocity_adj_matrix
        self._Vmn = np.array(list(Vmn.values()))
        self._chgPrice = np.array(list(rho_ch_s.values()))  # 视图对象（不支持索引）→list → array

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):  # action={0,1,2,...,11}

        rho_ch_s, Vmn, G_velocity_adj_matrix, G_length_adj_matrix, G_eqv_adj_matrix = utils.paramInit(df=self.df)

        self._velocity = G_velocity_adj_matrix
        self._Vmn = np.array(list(Vmn.values()))
        self._chgPrice = np.array(list(rho_ch_s.values()))  # 视图对象（不支持索引）→list → array

        g = Graph(41)
        g.graph = G_eqv_adj_matrix

        # 将action对应到图中DWER节点
        Mch = self.chgNodes[action]
        if (action % 2) == 0:
            Nch = self.chgNodes[action + 1]
        else:
            Nch = self.chgNodes[action - 1]

        dist1, prev1 = g.dijkstra(self._loc)
        dist1_to_Mch = dist1[Mch]  # 当前位置到选择的DWER的距离
        dist2, prev2 = g.dijkstra(self._endLoc)
        dist2_to_Nch = dist2[Nch]  # 选择的DWER到终点的距离
        Ctr = dist1_to_Mch + dist2_to_Nch

        # 由当前位置到选择的DWER起点的最短路径
        L = []
        u = Mch
        while math.isnan(prev1[u]) is False:
            L.insert(0, u)
            u = prev1[u]

        # MDP process
        lastLoc = self._loc
        if L:
            self._loc = L[0]
            et = self._SOC * self.emax - self.alpha * G_length_adj_matrix[lastLoc, self._loc]
            self._curTime = self._curTime + G_length_adj_matrix[lastLoc, self._loc] / G_velocity_adj_matrix[
                lastLoc, self._loc]
        else:
            et = self._SOC * self.emax

        self._SOC = et / self.emax

        observation = self._get_obs()

        terminated = np.array_equal(self._loc, Mch)

        if terminated:
            ech_max = G_length_adj_matrix[Mch, Nch] / self._Vmn[action] * self.p_ch
            ech = min(self.emax - et, ech_max)

            # 计算充电车道的拥堵程度C, beta=1
            if self._Vmn[action] < 20:
                C = 1 - self._Vmn[action] / 20
            else:
                C = 0

            Cch = self._chgPrice[action] * ech * (1 - self.beta * C)

            if self._SOC > 0:
                reward = -Ctr - Cch

            else:
                reward = -100

        else:
            reward = -self.Wij[lastLoc, self._loc]

        self.Wij = G_eqv_adj_matrix  # 更新t+1时刻

        truncated = False  # we do not limit the number of steps here

        info = {}

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = DynChgEnv()
    # env = FlattenObservation(env)
    check_env(env, warn=True)
    obs, _ = env.reset()
    print("Initial obs =", obs)
    print("obs space shape", env.observation_space.shape)

    n_steps = 20

    for step in range(n_steps):
        print(f"Step {step + 1}")
        action = 0
        # action = np.random.choice(range(env.n_DWERs))
        print("action = ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            print("Goal reached!", "reward=", reward)
            break
