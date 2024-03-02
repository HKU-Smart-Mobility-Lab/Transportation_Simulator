import os
import pickle
from config import *
from utilities.utilities import *

class PricingAgent(object):
    def __init__(self, **params):
        """
        Params:
        - learning_rate
        - discount_rate
        """
        # 状态空间定义
        self.time_slices = [i for i in range(int(LEN_TIME / LEN_TIME_SLICE))]
        self.supply_bins = [i for i in range(0, 100, 10)]
        self.demand_bins = [i for i in range(0, 100, 10)]

        self.price_options = [0.45, 0.5, 0.55, 0.6, 0.65]

        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']
        self.strategy = params['strategy']

        # 初始化 Q 表 {(time, supply_bin, demand_bin): [Q-values for actions]}
        self.q_value_table = {}
        for t in self.time_slices:
            for s_bin in self.supply_bins:
                for d_bin in self.demand_bins:
                    state_key = (t, s_bin, d_bin)
                    self.q_value_table[state_key] = [0.0] * len(self.price_options)


    def _discretize_state(self, raw_state):
        """
        离散化当前状态用于 Q 表索引。
        state = [time_slice, supply, demand]
        """
        time_slice = raw_state[0]

        supply = int(raw_state[1])
        demand = int(raw_state[2])

        # ⚠️ 将 supply/demand 限定在最大 bin = 90（或更高）
        supply_bin = min((supply // 10) * 10, 90)
        demand_bin = min((demand // 10) * 10, 90)

        # # ✅ 可选：调试输出，观察 supply/demand 分布
        # if np.random.rand() < 0.01:  # 只打印少量，避免刷屏
        #     print(f"[DEBUG] Discretizing state: time={time_slice}, supply={supply}, demand={demand} -> bin=({supply_bin}, {demand_bin})")

        return (time_slice, supply_bin, demand_bin)
    

    def update_q_value_table(self, s0, s1, action_idx, reward):
        key0 = self._discretize_state(s0)
        key1 = self._discretize_state(s1)

        if key0 not in self.q_value_table:
            self.q_value_table[key0] = [0.0] * len(self.price_options)
        if key1 not in self.q_value_table:
            self.q_value_table[key1] = [0.0] * len(self.price_options)

        q_old = self.q_value_table[key0][action_idx]

        # 判断 s1 是否为终止状态（仿真结束）
        is_terminal = s1[0] >= int(LEN_TIME / LEN_TIME_SLICE)  # s1[0] 是 time_slice

        if is_terminal:
            q_new = (1 - self.learning_rate) * q_old + self.learning_rate * reward
        else:
            max_q_next = max(self.q_value_table[key1])
            q_new = (1 - self.learning_rate) * q_old + self.learning_rate * (reward + self.discount_rate * max_q_next)

        self.q_value_table[key0][action_idx] = q_new


    def perceive(self, transitions: list):
        """
        transitions = [state_array, action_idx_array, next_state_array, reward_array]
        """
        current_states = transitions[0]
        action_indices = transitions[1]
        next_states = transitions[2]
        rewards = transitions[3]
    
        for i in range(len(current_states)):
            self.update_q_value_table(current_states[i], next_states[i], action_indices[i], rewards[i])


    def get_action(self, pricing_state, epsilon=0.1):
        """
        输入 pricing_state，输出每个订单的 designed_reward（价格）。
        """
        trip_distances = pricing_state["trip_distances"]  # Series
        supply = pricing_state["supply"]
        demand = pricing_state["demand"]
        time_slice = pricing_state["time_slice"]

        if self.strategy == "static":
            # 基于距离线性定价
            return 1, 2.5 + 0.5 * ((1000 * trip_distances - 322).clip(lower=0) / 322)

        elif self.strategy == "dynamic":
            state_key = self._discretize_state([time_slice, supply, demand])
            if state_key not in self.q_value_table:
                self.q_value_table[state_key] = [0.0] * len(self.price_options)

            if np.random.rand() < epsilon:
                action_idx = np.random.randint(len(self.price_options))
            else:
                action_idx = np.argmax(self.q_value_table[state_key])
            
            price_per_km = self.price_options[action_idx]
            price_array = 2.5 + price_per_km * ((1000 * trip_distances - 322).clip(lower=0) / 322)

            # 返回每个订单的价格（按距离映射）
            return action_idx, price_array

        else:
            raise ValueError("Unsupported pricing strategy")


    def save_parameters(self, epoch: int):
        # 修改保存路径为当前路径下的 models 文件夹
        base_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models')
        folder = os.path.join(base_folder, f'episode_{epoch}')
        
        # 如果文件夹不存在，则创建
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 保存文件路径
        file_path = os.path.join(folder, f'pricing_q_table_epoch_{epoch}.pickle')
        
        # 保存 Q 表到文件
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_value_table, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load_parameters(self, file_name):
        self.q_value_table = pickle.load(open(file_name, 'rb'))
