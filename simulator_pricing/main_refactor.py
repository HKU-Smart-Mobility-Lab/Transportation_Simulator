from utilities.utilities import *
from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
from utilities import *
from matplotlib import pyplot as plt
from pricing_agent import PricingAgent
import wandb
# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [100]
    max_distance_num = [1]

    cruise_flag = [True]
    pickup_flag = ['rg']
    delivery_flag = ['rg']

    wandb.login()
    pricing_refactor = wandb.init(project="simulator_pricing_refactor",
                                       config={"pricing_method": "dynamic",
                                               "driver_num": 100,
                                               "EPOCH":601},)      

    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:
                for single_driver_num in driver_num:
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['maximal_pickup_distance'] = single_max_distance_num

                        simulator = Simulator(**env_params)
                        simulator.reset()
                        track_record = []
                        t = time.time()

                        if env_params['rl_mode'] == "pricing":
                            if simulator.experiment_mode == 'test':
                                column_list = ['total_reward', 'matched_request_num',
                                               'long_request_num',
                                               'matched_long_request_num', 'matched_medium_request_num',
                                               'medium_request_num',
                                               'matched_short_request_num',
                                               'short_request_num', 'total_request_num',
                                               'waiting_time','pickup_time','occupancy_rate','occupancy_rate_no_pickup',
                                               'matched_long_request_ratio', 'matched_medium_request_ratio',
                                               'matched_short_request_ratio',
                                               'matched_request_ratio']
                                test_num = 10
                                test_interval = 20
                                threshold = 5
                                df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
                                # df = pickle.load(open(load_path + 'performance_record_test_' + env_params['method'] + '.pickle', 'rb'))
                                remaining_index_array = np.where(df['total_reward'].values == 0)[0]
                                if len(remaining_index_array > 0):
                                    last_stopping_index = remaining_index_array[0]
                                ax,ay = [],[]
                                
                                epoch = 0
                                for num in range(last_stopping_index, test_num):
                                    print('num: ', num)
                                    # simulator = Simulator(**env_params)
                                    pricing_agent = {}
                                    pricing_agent = PricingAgent(**pricing_params)
                                    if env_params['pricing_strategy']=='dynamic':
                                        pricing_agent.load_parameters(
                                            load_path + '/episode_300/pricing_q_table_epoch_300.pickle')
    
                                    total_reward = 0
                                    total_request_num = 0
                                    long_request_num = 0
                                    medium_request_num = 0
                                    short_request_num = 0
                                    matched_request_num = 0
                                    matched_long_request_num = 0
                                    matched_medium_request_num = 0
                                    matched_short_request_num = 0
                                    occupancy_rate = 0
                                    occupancy_rate_no_pickup = 0
                                    pickup_time = 0
                                    waiting_time = 0
                                    for date in TEST_DATE_LIST:
                                        simulator.experiment_date = date
                                        simulator.reset()
                                        start_time = time.time()
                                        for step in range(simulator.finish_run_step):
                                            dispatch_transitions = simulator.rl_step(pricing_agent=pricing_agent)
                                        end_time = time.time()

                                        total_reward += simulator.total_reward
                                        total_request_num += simulator.total_request_num
                                        occupancy_rate += simulator.occupancy_rate
                                        matched_request_num += simulator.matched_requests_num
                                        long_request_num += simulator.long_requests_num
                                        medium_request_num += simulator.medium_requests_num
                                        short_request_num += simulator.short_requests_num
                                        matched_long_request_num += simulator.matched_long_requests_num
                                        matched_medium_request_num += simulator.matched_medium_requests_num
                                        matched_short_request_num += simulator.matched_short_requests_num
                                        occupancy_rate_no_pickup += simulator.occupancy_rate_no_pickup
                                        pickup_time += simulator.pickup_time / simulator.matched_requests_num
                                        waiting_time += simulator.waiting_time / simulator.matched_requests_num
                                    
                                    
                                    epoch += 1
                                    total_reward = total_reward / len(TEST_DATE_LIST)
                                    ax.append(epoch)
                                    ay.append(total_reward)
                                    print("------------test after epoch",epoch)
                                    print("total reward",total_reward)
                                    total_request_num = total_request_num / len(TEST_DATE_LIST)
                                    occupancy_rate = occupancy_rate / len(TEST_DATE_LIST)
                                    matched_request_num = matched_request_num / len(TEST_DATE_LIST)
                                    long_request_num = long_request_num / len(TEST_DATE_LIST)
                                    medium_request_num = medium_request_num / len(TEST_DATE_LIST)
                                    short_request_num = short_request_num / len(TEST_DATE_LIST)
                                    matched_long_request_num = matched_long_request_num / len(TEST_DATE_LIST)
                                    matched_medium_request_num = matched_medium_request_num / len(TEST_DATE_LIST)
                                    matched_short_request_num = matched_short_request_num / len(TEST_DATE_LIST)
                                    occupancy_rate_no_pickup = occupancy_rate_no_pickup / len(TEST_DATE_LIST)
                                    pickup_time = pickup_time / len(TEST_DATE_LIST)
                                    waiting_time = waiting_time / len(TEST_DATE_LIST)
                                    print("pick",pickup_time)
                                    print("wait",waiting_time)
                                    print("matching ratio",matched_request_num/total_request_num)
                                    print("ocu",occupancy_rate)
                                    
                                    record_array = np.array(
                                        [total_reward, matched_request_num,
                                          long_request_num, matched_long_request_num,
                                         matched_medium_request_num, medium_request_num, matched_short_request_num,
                                         short_request_num, total_request_num,waiting_time,pickup_time,occupancy_rate,occupancy_rate_no_pickup])
                                    # record_array = np.array([total_reward])

                                    if num == 0:
                                        df.iloc[0, :13] = record_array
                                    else:
                                        df.iloc[num, :13] = (df.iloc[(num - 1), :13].values * num + record_array) / (
                                                    num + 1)

                                    if num % 120 == 0:  # save the result every 10
                                        pickle.dump(df, open(
                                            load_path + 'performance_record_test_' + env_params['method'] + '.pickle',
                                            'wb'))

                                    if num >= (test_interval - 1):
                                        profit_array = df.loc[(num - test_interval):num, 'total_reward'].values
                                        # print(profit_array)
                                        error = np.abs(np.max(profit_array) - np.min(profit_array))
                                        print('error: ', error)
                                        if error < threshold:
                                            index = num
                                            print('converged at index ', index)
                                            break

                                df.loc[:(num), 'matched_long_request_ratio'] = df.loc[:(num),
                                                                               'matched_long_request_num'].values / df.loc[
                                                                                                                    :(num),
                                                                                                                    'long_request_num'].values
                                df.loc[:(num), 'matched_medium_request_ratio'] = df.loc[:(num),
                                                                                 'matched_medium_request_num'].values / df.loc[
                                                                                                                        :(
                                                                                                                            num),
                                                                                                                        'medium_request_num'].values
                                df.loc[:(num), 'matched_short_request_ratio'] = df.loc[:(num),
                                                                                'matched_short_request_num'].values / df.loc[
                                                                                                                      :(
                                                                                                                          num),
                                                                                                                      'short_request_num'].values
                                df.loc[:(num), 'matched_request_ratio'] = df.loc[:(num),
                                                                          'matched_request_num'].values / df.loc[:(num),
                                
                                                                                                     'total_request_num'].values
                                print(df.columns) 
                                pickle.dump(df,
                                            open(load_path + 'performance_record_test_' + env_params['method'] + '.pickle',
                                                 'wb'))
                                print(df.iloc[test_num-1, :])

                                # np.savetxt(load_path + "supply_dist_" + simulator.method + ".csv", simulator.driver_spatial_dist, delimiter=",")

                            elif simulator.experiment_mode == 'train':
                                print("training process")
                                epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, 500, decay=DECAY,
                                                                    pre_steps=PRE_STEP)
                                epsilons = np.concatenate([epsilons, np.zeros(NUM_EPOCH - 500)])
                                # epsilons = np.zeros(NUM_EPOCH)
                                total_reward_record = np.zeros(NUM_EPOCH)

                                for epoch in range(NUM_EPOCH):
                                    date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                                    simulator.experiment_date = date
                                    simulator.reset()
                                    start_time = time.time()
                                    for step in range(simulator.finish_run_step):
                                        # print("At step {}".format(step)) # TODO: delete this test print
                                        # step_start_time = time.time()
                                        pricing_transitions = simulator.rl_step(pricing_agent=pricing_agent, epsilon=epsilons[epoch])
                                        if pricing_agent is not None:
                                            pricing_agent.perceive(pricing_transitions)
                                        # step_end_time = time.time()
                                        # print("step time",step_end_time-step_start_time)
                                    end_time = time.time()
                                    total_reward_record[epoch] = simulator.total_reward
                                    # pickle.dump(simulator.order_status_all_time,open("1106a-order.pkl","wb"))
                                    # pickle.dump(simulator.driver_status_all_time,open("1106a-driver.pkl","wb"))
                                    # pickle.dump(simulator.used_driver_status_all_time,open("1106a-used-driver.pkl","wb"))
                                    print('epoch:', epoch)
                                    print('epoch running time: ', end_time - start_time)
                                    print('epoch total reward: ', simulator.total_reward)
                                    print("total orders",simulator.total_request_num)
                                    print("matched orders",simulator.matched_requests_num)
                                    matching_ratio = simulator.matched_requests_num/simulator.total_request_num
                                    print("macthing ratio",matching_ratio)
                                    
                                    # 使用 wandb 记录指标
                                    pricing_refactor.log({
                                        "epoch": epoch,
                                        "total_reward": simulator.total_reward,
                                        "matching_ratio": matching_ratio,
                                        "epoch_running_time": end_time - start_time,
                                    })

                                    if epoch % 50 == 0:  # save the result every 200 epochs
                                        pricing_agent.save_parameters(epoch)

