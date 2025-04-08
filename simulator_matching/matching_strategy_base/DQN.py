import numpy as np
import os
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from config import START_TIMESTAMP, LEN_TIME_SLICE


class DQNAgent:
    def __init__(self, state_dim=2, learning_rate=0.005, gamma=0.95,
                 memory_size=200, batch_size=16, target_update_freq=10):
        self.state_dim = state_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.memory = deque(maxlen=memory_size)

        # Main and target networks
        self.model = self.build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')  # Predict Q(s) as scalar
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
        return model

    def _convert_state(self, raw_state):
        if not isinstance(raw_state, (list, np.ndarray)):
            raise TypeError(f"Expected list/array for raw_state, got {type(raw_state)}: {raw_state}")
        if len(raw_state) != 2:
            raise ValueError(f"State must have 2 elements [timestamp, grid_id], got: {raw_state}")

        t = int((raw_state[0] - START_TIMESTAMP - 1) / LEN_TIME_SLICE)
        g = int(raw_state[1])
        return np.array([t, g], dtype=np.float32)

    def _is_terminal_state(self, state_array):
        t = int(state_array[0])
        max_slice = int(6 * 60 * 60 / LEN_TIME_SLICE)
        return t >= max_slice

    def get_q_value(self, raw_state):
        try:
            s = self._convert_state(raw_state)
            q = self.model.predict(s[np.newaxis, :], verbose=0)[0][0]
            return q
        except Exception as e:
            print(f"[DQN get_q_value] Invalid state {raw_state}: {e}")
            return 0.0

    def perceive(self, transitions: list):
        state_array      = transitions[0]
        reward_array     = transitions[1]
        next_state_array = transitions[2]

        for i in range(len(state_array)):
            try:
                s  = self._convert_state(state_array[i])
                s_ = self._convert_state(next_state_array[i])
            except Exception as e:
                print(f"[DQN perceive] Skipping index {i} due to state conversion error: {e}")
                continue

            r = reward_array[i]
            done = self._is_terminal_state(s_)
            self.memory.append((s, r, s_, done))

        self.train()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch_idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, targets = [], []

        for idx in minibatch_idx:
            s, r, s_, done = self.memory[idx]

            if done:
                q_target = r
            else:
                q_next = self.target_model.predict(s_[np.newaxis, :], verbose=0)[0][0]
                q_target = r + self.gamma * q_next

            states.append(s)
            targets.append([q_target])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())