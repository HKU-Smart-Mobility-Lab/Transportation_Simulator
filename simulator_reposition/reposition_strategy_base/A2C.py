<<<<<<< HEAD:simulator_reposition/reposition_strategy_base/A2C.py
import numpy as np
import keras.models
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Reshape, concatenate, Conv2D, Flatten
# from keras.layers.merge import Add, Multiply
from keras.layers import add, multiply
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, MeanSquaredError
#import keras.backend as K
#import tensorflow as tf
import random
from copy import deepcopy
from path import *
from collections import deque
import os
import datetime
#from scipy import exp

# rl for repositioning
# A2C
local_random_gen = np.random.default_rng()

class A2C:
    def __init__(self, action_dim, state_dim, available_directions, load_model, load_path, model_name='', logger=None, **kwargs):
        self.discount_factor = kwargs.pop('discount_factor', 0.99)
        self.actor_lr = kwargs.pop('actor_lr', 0.001)
        self.critic_lr = kwargs.pop('critic_lr', 0.005)
        self.logger = logger

        self.state_indexes = list(range(0, state_dim))
        self.state_size = len(self.state_indexes)
        self.action_size = action_dim
        self.value_size = 1

        self.render = False
        self.load_model = load_model

        self.total_step = 0
        self.total_step_threshold = -1

        self.load_path = load_path

        self.actor_structure = kwargs['actor_structure']
        self.critic_structure = kwargs['critic_structure']
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.df_available_directions = available_directions

        self.model_name = model_name
        if self.load_model:
            # load model for test
            actor_path = os.path.join(self.load_path, "20250315102853", f"{self.model_name}_epoch600_actor")
            critic_path = os.path.join(self.load_path, "20250315102853", f"{self.model_name}_epoch600_critic")
            
            if os.path.exists(actor_path) and os.path.exists(critic_path):
                # Chanes from load_weights to load_model
                self.actor = keras.models.load_model(actor_path)
                self.critic = keras.models.load_model(critic_path)
                print(f"Models loaded successfully")
            else:
                print(f"Model loading failed. Check the paths: '{actor_path}' and '{critic_path}'")

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(self.actor_structure[0], input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        for layer_size in self.actor_structure[1:]:
            actor.add(Dense(layer_size, activation='relu',kernel_initializer='he_uniform'))
        # 激活函数为softmax，用于输出动作的概率分布
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        loss_fn = CategoricalCrossentropy()
        actor.compile(loss=loss_fn, optimizer=Adam(learning_rate=self.actor_lr))
        return actor


    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(self.critic_structure[0], input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        for layer_size in self.critic_structure[1:]:
            critic.add(Dense(layer_size, activation='relu',kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',kernel_initializer='he_uniform'))
        critic.summary()
        loss_fn = MeanSquaredError()
        critic.compile(loss=loss_fn, optimizer=Adam(learning_rate=self.critic_lr))
        return critic

    # def get_action(self, state, batch_size=1):
    #     # not available in this experiment
    #     state = np.array(state)
    #     policy = self.actor.predict(state, batch_size=batch_size).flatten()
    #     action = np.random.choice(self.action_size, 1, p=policy)[0]
    #     return action

    def train_model(self, state_array, action_array, reward_array, next_state_array, done_array):
        #训练时未对direction做严格限制
        batch_size = len(state_array)
        # critic 目标值
        target_array = np.zeros([batch_size, self.value_size])
        # actor 优势值
        advantages_array = np.zeros([batch_size, self.action_size])
        con_done = done_array == 1
        con_not_done = done_array == 0

        # update critic
        # 预测下一个状态的值：V(S')
        next_value_array = self.critic.predict(next_state_array).flatten()

        if np.any(con_done):  # 只要有终止状态
            target_array[con_done, 0] = reward_array[con_done]

        if np.any(con_not_done):  # 只要有未终止状态
            # V(S) = R + γV(S')
            target_array[con_not_done, 0] = reward_array[con_not_done] + self.discount_factor * next_value_array[con_not_done]

        # 训练critic：通过MSE更新
        # state_array是输入值
        # target_array是目标值(作为真实值-----当前的reward＋S'状态的估计值)，critic.predict(state_array)是预测值
        # critic的loss function：目的是让critic的预测值和TD目标值尽可能接近-----使用MSE
        critic_history = self.critic.fit(state_array, target_array, batch_size=batch_size, epochs=1, verbose=2)
        critic_loss = critic_history.history['loss'][0]

        # update actor
        # 使用Critic预测当前状态的值：V(S)，并将其作为Actor的指导
        value_array = self.critic.predict(state_array).flatten()

        if np.any(con_done):  
            advantages_array[con_done, action_array[con_done]] = reward_array[con_done] - value_array[con_done]

        if np.any(con_not_done):  
            # A(S, A) = R + γV(S') - V(S)：如果A(S, A) > 0，说明这个动作是好的-----fit过程增加该动作的概率
            advantages_array[con_not_done, action_array[con_not_done]] = \
                reward_array[con_not_done] + self.discount_factor * next_value_array[con_not_done] - value_array[con_not_done]
        # 训练actor：通过CategoricalCrossentropy更新
        # advantages_array是目标值（作为伪标签去训练actor）-----使用CategoricalCrossentropy
        # advantages_array只在对应的action上有值，其他位置为0（可以理解为多分类任务中的伪标签）
        actor_history = self.actor.fit(state_array, advantages_array, batch_size=batch_size, epochs=1, verbose=2)
        actor_loss = actor_history.history['loss'][0]
        self.logger.debug(f"Step: {self.total_step}, Critic Loss: {critic_loss}, Actor Loss: {actor_loss}")

    def perceive(self, transition_list):
        # updated version
        if self.total_step < self.total_step_threshold:
            self.total_step += 1
            return None
        else:
            pass
        state_array  = transition_list[0]
        action_array = transition_list[1]
        reward_array = transition_list[2]
        next_state_array = transition_list[3]
        done_array = transition_list[4]

        # train model
        
        self.train_model(state_array, action_array, reward_array, next_state_array, done_array)

    def egreedy_actions(self, origin_state, epsilon, grid_id_index):

        state = deepcopy(origin_state)
        state = state[np.newaxis, :]
        # 根据grid_id_index获取可行的方向
        available_directions = self.df_available_directions.iloc[grid_id_index, 1:].values

        if local_random_gen.random() <= epsilon:
            policy = 1 / self.action_size
        else:
            state_action = self.actor.predict(state, batch_size=1).flatten()
            policy = deepcopy(state_action)
            #policy[np.isnan(policy)] = 0.00001
            policy = np.nan_to_num(policy)
            policy[policy < 0] = 0
            if sum(policy) == 0:
                policy = 1 / self.action_size
        
        policy = policy * available_directions
        if sum(policy) == 0:
            policy = available_directions / sum(available_directions)
        # 对policy归一化
        policy = policy / sum(policy)
        # choice返回的是一个数组------所以需要[0]
        # actor是通过策略分布来选取动作的（概率大的动作被选取的可能性大，但不是一定）
        action = local_random_gen.choice(self.action_size, 1, p=policy)[0]
        np.set_printoptions(precision=3)

        return action

    def save_model(self, epoch, timestamp):
         # Assuming load_path is supposed to be a class attribute
        save_path = os.path.join(load_path, timestamp)
        
        # Make the directory for the timestamp if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the actor and critic models
        self.actor.save(os.path.join(save_path, f"{self.model_name}_epoch{epoch}_actor"))
        self.critic.save(os.path.join(save_path, f"{self.model_name}_epoch{epoch}_critic"))





=======
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Reshape, concatenate, Conv2D, Flatten
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, MeanSquaredError
#import keras.backend as K
#import tensorflow as tf
import random
from copy import deepcopy
from path import *
from collections import deque
#from scipy import ex


# rl for repositioning
# A2C
class A2C:
    def __init__(self, action_dim, state_dim, available_directions, load_model, model_name='', **kwargs):
        self.discount_factor = kwargs.pop('discount_factor', 0.99)
        self.actor_lr = kwargs.pop('actor_lr', 0.001)
        self.critic_lr = kwargs.pop('critic_lr', 0.005)

        self.state_indexes = list(range(0, state_dim))
        self.state_size = len(self.state_indexes)
        self.action_size = action_dim
        self.value_size = 1

        self.render = False
        self.load_model = load_model

        self.total_step = 0
        self.total_step_threshold = -1

        self.actor_structure = kwargs['actor_structure']
        self.critic_structure = kwargs['critic_structure']
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.df_available_directions = available_directions

        self.model_name = model_name
        if self.load_model:
            self.actor.load_weights(load_path + "\\save_model\\" + model_name + "_actor")
            self.critic.load_weights(load_path + "\\save_model\\" + model_name + "_critic")

    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(self.actor_structure[0], input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        for layer_size in self.actor_structure[1:]:
            actor.add(Dense(layer_size, activation='relu',kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        loss_fn = CategoricalCrossentropy()
        actor.compile(loss=loss_fn, optimizer=adam_v2.Adam(learning_rate=self.actor_lr))
        return actor

    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(self.critic_structure[0], input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        for layer_size in self.critic_structure[1:]:
            critic.add(Dense(layer_size, activation='relu',kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',kernel_initializer='he_uniform'))
        critic.summary()
        loss_fn = MeanSquaredError()
        critic.compile(loss=loss_fn, optimizer=adam_v2.Adam(learning_rate=self.critic_lr))
        return critic

    def get_action(self, state, batch_size=1):
        # not available in this experiment
        state = np.array(state)
        policy = self.actor.predict(state, batch_size=batch_size).flatten()
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def train_model(self, state_array, action_array, reward_array, next_state_array, done_array):
        #训练时未对direction做严格限制
        batch_size = len(state_array)
        target_array = np.zeros([batch_size, self.value_size])
        advantages_array = np.zeros([batch_size, self.action_size])
        con_done = done_array == 1
        con_not_done = ~con_done

        #update critic
        next_value_array = self.critic.predict(next_state_array).flatten()
        if not np.all(~con_done):
            target_array[con_done, 0] = reward_array[con_done]
        if not np.all(~con_not_done):
            target_array[con_not_done, 0] = reward_array[con_not_done] + self.discount_factor * next_value_array[
                con_not_done]
        self.critic.fit(state_array, target_array, batch_size=batch_size, epochs=1, verbose=0)

        #update actor
        value_array = self.critic.predict(state_array).flatten()
        next_value_array = self.critic.predict(next_state_array).flatten()
        if not np.all(~con_done):
            advantages_array[con_done, action_array[con_done]] = reward_array[con_done] - value_array[con_done]
        if not np.all(~con_not_done):
            advantages_array[con_not_done, action_array[con_not_done]] = \
                reward_array[con_not_done] + self.discount_factor * next_value_array[con_not_done] - value_array[con_not_done]
        self.actor.fit(state_array, advantages_array, batch_size=batch_size, epochs=1, verbose=0)

    def perceive(self, transition_list):
        # updated version
        if self.total_step < self.total_step_threshold:
            self.total_step += 1
            return None
        else:
            pass

        state_array  = transition_list[0]
        action_array = transition_list[1]
        reward_array = transition_list[2]
        next_state_array = transition_list[3]
        done_array = transition_list[4]

        # train model
        self.train_model(state_array, action_array, reward_array, next_state_array, done_array)

    def egreedy_actions(self, origin_state, epsilon, grid_id_index):

        state = deepcopy(origin_state)
        state = state[np.newaxis, :]
        available_directions = self.df_available_directions.iloc[grid_id_index, 1:].values

        if random.random() <= epsilon:
            policy = 1 / self.action_size
        else:
            state_action = self.actor.predict(state, batch_size=1).flatten()
            policy = deepcopy(state_action)
            #policy[np.isnan(policy)] = 0.00001
            policy = np.nan_to_num(policy)
            policy[policy < 0] = 0
            if sum(policy) == 0:
                policy = 1 / self.action_size

        policy = policy * available_directions
        if sum(policy) == 0:
            policy = available_directions / sum(available_directions)
        policy = policy / sum(policy)
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        np.set_printoptions(precision=3)

        return action

    def save_model(self, epoch):
        self.actor.save(load_path + "\\save_model\\" + str(epoch) + "\\" + self.model_name + "_actor")
        self.critic.save(load_path + "\\save_model\\" + str(epoch) + "\\" + self.model_name + "_critic")





>>>>>>> 6a9d3e4 (fix config, fix keras import):simulator/A2C.py
