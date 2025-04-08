from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
from utilities import *
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
from pricing_agent import PricingAgent

if __name__ == "__main__":
    simulator = Simulator(**env_params)

    if env_params['rl_mode'] == "pricing":
        epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, NUM_EPOCH, decay=DECAY, pre_steps=PRE_STEP)
        pricing_agent = PricingAgent(strategy="dynamic_q", learning_rate=0.01)

        for epoch in range(NUM_EPOCH):
            date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
            simulator.experiment_date = date
            simulator.reposition_method = "random_cruise"  # 固定reposition策略
            simulator.reset()

            for step in tqdm(range(simulator.finish_run_step)):
                # Step 1: 获取定价状态
                pricing_state = simulator.get_pricing_state()
                trip_distances = pricing_state['trip_distances']
                
                # Step 2: agent生成action（即价格）
                pricing_action_array = []
                for dist in trip_distances:
                    state = (dist, pricing_state['supply'], pricing_state['demand'])
                    price = pricing_agent.get_action(state, epsilon=epsilons[epoch])
                    pricing_action_array.append(price)
                
                simulator.exectue_pricing_action(pricing_action_array)

                # Step 3: simulator运行匹配逻辑 + 状态更新
                simulator.step1()
                simulator.step2(np.array([]))  # 空动作表示使用默认的reposition策略

                # Step 4: 构造reward（乘客是否接受）并更新pricing agent
                reward_array = []
                for price, req in zip(pricing_action_array, simulator.requests.itertuples()):
                    distance = getattr(req, 'trip_distance')
                    pickup_dis = getattr(req, 'pickup_distance', 0)

                    # 构造顾客接受函数（高价格和远pickup降低接受率）
                    cancel_prob = min(1.0, 0.1 + 0.3 * (pickup_dis / 3.0) + 0.6 * (price / 20.0))
                    accept = np.random.rand() > cancel_prob
                    reward_array.append(1.0 if accept else -1.0)

                # Step 5: agent更新
                for dist, price, reward in zip(trip_distances, pricing_action_array, reward_array):
                    state = (dist, pricing_state['supply'], pricing_state['demand'])
                    pricing_agent.update(state, price, reward)

            print(f"Epoch {epoch} finished, total_reward: {simulator.total_reward}")

        # Save final pricing agent
        pricing_agent.save_parameters("pricing_agent_final.pkl")
