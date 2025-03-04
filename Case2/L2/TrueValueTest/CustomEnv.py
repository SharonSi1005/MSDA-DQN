import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete, MultiDiscrete
import utils
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FlattenObservation
import global_var


class DynChgEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, G_velocity, rho_ch_s, Vmn, G_length_adj_matrix,
                 G_velocity_adj_matrix, G_eqv_adj_matrix, n_DWERs=12, n_nodes=41, chgNodes=None, allNodes=None):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.n_DWERs = n_DWERs
        self.n_nodes = n_nodes
        if chgNodes is None:
            chgNodes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.chgNodes = chgNodes
        if allNodes is None:
            allNodes = list(range(41))
        self.allNodes = allNodes
        self.comNodes = [node for node in self.allNodes if node not in self.chgNodes]  # 非充电节点
        self.emax = 80  # 电池最大容量
        self.alpha = 0.15  # kwh/km 每km耗电量
        self.p_ch = 30  # kw wireless charging power

        # 输入参数初始化(固定参数)
        self.G_velocity = G_velocity
        self.rho_ch_s = rho_ch_s
        self.Vmn = Vmn
        self.G_length_adj_matrix = G_length_adj_matrix
        self.G_velocity_adj_matrix = G_velocity_adj_matrix
        self.G_eqv_adj_matrix = G_eqv_adj_matrix

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

        # 定义动作和状态空间
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

    def set_obs(self, obs_dict=None, endLoc=None):
        self._loc = obs_dict["loc"]
        self._SOC = obs_dict["SOC"]
        self._curTime = obs_dict["curTime"]
        self._velocity = obs_dict["velocity"]
        self._Ctr = obs_dict["travelCost"]
        self._Cch = obs_dict["travelCost"]
        self._L2 = obs_dict["optPath"]
        self._endLoc = endLoc

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 为self.np_random提供seed

        self._startLoc = int(10)
        self._endLoc = int(29)

        print("start node =", self._startLoc, "; end node =", self._endLoc)

        self._loc = self._startLoc
        self._SOC = np.array([0.5], dtype=np.float64)  # 给定初始SOC为40kWh
        self._curTime = np.zeros(1)

        self._velocity = np.zeros(4)
        G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix \
            = utils.paramInit()  # 用到random seed
        print("初始化的Vmn为", Vmn)
        idx = 0
        for adjNode in G_velocity.neighbors(self._loc):
            self._velocity[idx] = G_velocity_adj_matrix[self._loc, adjNode]
            idx = idx + 1

        Ctr, L, Cch = utils.getFeatures(
            rho_ch_s,
            Vmn,
            G_length_adj_matrix,
            G_eqv_adj_matrix,
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
        # MDP process
        lastLoc = self._loc
        self._loc = self._L2[action, 0]
        print("当前位置：", self._loc)

        et = self._SOC * self.emax - self.alpha * self.G_length_adj_matrix[lastLoc, self._loc]
        self._SOC = et / self.emax

        if lastLoc != self._loc:
            self._curTime = self._curTime + self.G_length_adj_matrix[lastLoc, self._loc] / \
                            self.G_velocity_adj_matrix[lastLoc, self._loc]

        self._velocity = np.zeros(4)
        idx = 0
        for adjNode in self.G_velocity.neighbors(self._loc):
            self._velocity[idx] = self.G_velocity_adj_matrix[self._loc, adjNode]
            idx = idx + 1

        Ctr, L, Cch = utils.getFeatures(
            rho_ch_s=self.rho_ch_s,
            Vmn=self.Vmn,
            G_length_adj_matrix=self.G_length_adj_matrix,
            G_eqv_adj_matrix=self.G_eqv_adj_matrix,
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

        terminated = np.array_equal(self._loc, self.chgNodes[action])

        info = {}

        Wij = global_var.get_value('Wij')

        if terminated:
            Cch = self._Cch[action]
            info["Cch_mn"] = Cch
            # print("充电cost为", Cch_mn, "类型为", type(Cch_mn))
            info["Cch1_lastStep"] = Wij[lastLoc, self._loc]

            Ctr2 = self._Ctr[action]
            '''print("行驶cost为", Ctr2)'''

            if self._SOC > 0:
                reward = - Wij[lastLoc, self._loc] - Cch - Ctr2
            else:  # 到达WCL使电量小于0
                reward = -100
        else:
            reward = - Wij[lastLoc, self._loc]

        Wij = self.G_eqv_adj_matrix  # 更新t+1时刻
        global_var.set_value('Wij', Wij)

        truncated = False  # we do not limit the number of steps here

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":

    np.random.seed(38)
    env = DynChgEnv(G_velocity=1)

    n_steps = 100
    # obs, _ = env.reset(seed=38)
    n_DWERs = 12
    chgNodes = [1, 13, 12, 23, 9, 21, 26, 27, 37, 38, 6, 7]
    allNodes = list(range(41))
    comNodes = [node for node in allNodes if node not in chgNodes]  # 非充电节点
    emax = 80  # 电池最大容量
    alpha = 0.15  # kwh/km 每km耗电量
    p_ch = 30  # kw wireless charging power

    startLoc = 18
    endLoc = 19
    '''
    startLoc = int(np.random.choice(comNodes))
    endLoc = startLoc
    while np.array_equal(endLoc, startLoc):
        endLoc = int(np.random.choice(comNodes))'''

    print("start node =", startLoc, "; end node =", endLoc)

    loc = startLoc
    SOC = np.array([0.5], dtype=np.float64)  # 给定初始SOC为40kWh
    curTime = np.zeros(1)

    velocity = np.zeros(4)
    G_velocity, rho_ch_s, Vmn, G_length_adj_matrix, G_velocity_adj_matrix, G_eqv_adj_matrix \
        = utils.paramInit()  # 用到random seed
        
    print("初始化的Vmn为", Vmn)

    global_var._init()
    Wij = G_eqv_adj_matrix  # 初始化全局变量Wij
    global_var.set_value('Wij', Wij)

    # print("初始化的Wij为", global_var.get_value('Wij'))
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

    env.set_obs(obs_dict, endLoc)  # 令self._loc = loc

    env = FlattenObservation(env)
    obs = FlattenObservation.observation(self=env, observation=obs_dict)  # 把dict型flatten
    print("Initial obs =", obs)

    for step in range(n_steps):
        print(f"Step {step + 1}")
        action = 0
        # action = np.random.choice(range(env.n_DWERs))
        print("action = ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("The", step + 1, "step reward is", reward)
        # print('Wij', global_var.get_value('Wij'))
        done = terminated or truncated
        print("obs=", obs, "reward=", reward, "done=", done)
        if done:
            print("Goal reached!", "info is", info)
            break

