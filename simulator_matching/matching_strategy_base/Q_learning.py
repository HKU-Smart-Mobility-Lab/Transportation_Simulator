import os
import pickle
from utilities import State
from config import env_params, LEN_TIME, LEN_TIME_SLICE, START_TIMESTAMP

class QLearningAgent:
    def __init__(self, **params):
        """
        Q-learning Agent for Matching
        """
        self.grid_ids = [i for i in range(env_params['grid_num'])]
        self.time_slices = [i for i in range(int(LEN_TIME / LEN_TIME_SLICE))]

        # ✅ 明确动作空间为 [0,1,2,3,4]
        self.actions = [0, 1, 2, 3, 4] # 后续可以改为action_size

        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']

        self.q_value_table = {
            (State(t, g), a): 0.0
            for t in self.time_slices
            for g in self.grid_ids
            for a in self.actions
        }

    def update_q_value_table(self, s0: State, a0: int, s1: State, reward: float):
        key0 = (s0, a0)

        if s1.time_slice >= int(LEN_TIME / LEN_TIME_SLICE):  # 终止状态
            try:
                self.q_value_table[key0] = (1 - self.learning_rate) * self.q_value_table[key0] + \
                                           self.learning_rate * reward
            except:
                print("State:", s0.time_slice, s0.grid_id, "Action:", a0)
        else:
            # print("State:", s0.time_slice, s0.grid_id, "Action:", a0)
            max_q_s1 = max(self.q_value_table[(s1, a)] for a in self.actions)
            discounted = (self.discount_rate ** (s1.time_slice - s0.time_slice)) * max_q_s1
            self.q_value_table[key0] = (1 - self.learning_rate) * self.q_value_table[key0] + \
                                       self.learning_rate * (reward + discounted)
            # print("update q value:", self.q_value_table[key0])



    def perceive(self, transitions: list):
        current_states = transitions[0]
        actions = transitions[1]
        print("actions", actions)
        next_states = transitions[2]
        rewards = transitions[3]
        num = len(current_states)

        for i in range(num):
            t0 = int((current_states[i][0] - START_TIMESTAMP - 1) / LEN_TIME_SLICE)
            g0 = int(current_states[i][1])
            s0 = State(t0, g0)

            t1 = int((next_states[i][0] - START_TIMESTAMP - 1) / LEN_TIME_SLICE)
            g1 = int(next_states[i][1])
            s1 = State(t1, g1)

            a0 = int(actions[i])  # ✅ 确保是整数动作
            self.update_q_value_table(s0, a0, s1, rewards[i])



    def save_parameters(self, epoch: int):
        # file path
        root_file_path = os.path.abspath(os.path.dirname(__file__))
        folder_path = os.path.join(root_file_path, 'episode_' + str(epoch))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)  # create a folder
        file_path = os.path.join(folder_path, 'Qlearning_q_value_table_epoch_' + str(epoch) + '.pickle')

        v = {
            t: {
                g: {a: self.q_value_table[(State(t, g), a)] for a in self.actions}
                for g in self.grid_ids
            }
            for t in self.time_slices
        }

        with open(file_path, 'wb') as f:
            pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, file_name):
        q_table = pickle.load(open(file_name, 'rb'))
        for t in self.time_slices:
            for g in self.grid_ids:
                for a in self.actions:
                    self.q_value_table[(State(t, g), a)] = q_table[t][g][a]
