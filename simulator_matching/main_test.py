from simulator_env_baseline import Simulator
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
from matching_strategy_base.sarsa import SarsaAgent
from matplotlib import pyplot as plt
# from A2C import * # you may comment this import if you are running matching
# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [100]
    max_distance_num = [1]

    cruise_flag = [True if env_params['rl_mode'] == 'matching' else False]
    pickup_flag = ['rg']
    delivery_flag = ['rg']

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
                        # Comment simulator.reset() below if you are not running matching with instant_reward_no_subway
                        simulator.reset()
                        track_record = []
                        t = time.time()

                        if env_params['rl_mode'] == "matching":
                            if simulator.experiment_mode == 'test':
                                
                                column_list = ['total_reward', 'matched_transfer_request_num', 'matched_request_num',
                                               'transfer_request_num',
                                               'long_request_num',
                                               'matched_long_request_num', 'matched_medium_request_num',
                                               'medium_request_num',
                                               'matched_short_request_num',
                                               'short_request_num', 'total_request_num',
                                               'matched_transfer_request_ratio', 'transfer_long_request_ratio',
                                               'matched_long_request_ratio', 'matched_medium_request_ratio',
                                               'matched_short_request_ratio',
                                               'matched_request_ratio','waiting_time','pickup_time','occupancy_rate','occupancy_rate_no_pickup']
                                
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
                                    agent = {}
                                    if simulator.method in ['sarsa', 'sarsa_no_subway', 'sarsa_travel_time',
                                                            'sarsa_travel_time_no_subway',
                                                            'sarsa_total_travel_time', 'sarsa_total_travel_time_no_subway']:
                                        agent = SarsaAgent(**qTable_params)
                                        agent.load_parameters(
                                            load_path + 'episode_4000\\sarsa_q_value_table_epoch_4000.pickle')

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
                                    transfer_request_num = 0
                                    matched_transfer_request_num = 0
                                    for date in TEST_DATE_LIST:
                                        simulator.experiment_date = date
                                        simulator.reset()
                                        start_time = time.time()
                                        for step in range(simulator.finish_run_step):
                                            dispatch_transitions = simulator.rl_step(agent)
                                        end_time = time.time()

                                        total_reward += simulator.total_reward
                                        total_request_num += simulator.total_request_num
                                        transfer_request_num += simulator.transfer_request_num
                                        occupancy_rate += simulator.occupancy_rate
                                        print("occupancy_rate",occupancy_rate)
                                        matched_request_num += simulator.matched_requests_num
                                        matched_transfer_request_num += simulator.matched_transferred_requests_num
                                        long_request_num += simulator.long_requests_num
                                        medium_request_num += simulator.medium_requests_num
                                        short_request_num += simulator.short_requests_num
                                        matched_long_request_num += simulator.matched_long_requests_num
                                        matched_medium_request_num += simulator.matched_medium_requests_num
                                        matched_short_request_num += simulator.matched_short_requests_num
                                        occupancy_rate_no_pickup += simulator.occupancy_rate_no_pickup
                                        pickup_time += simulator.pickup_time / simulator.matched_requests_num
                                        print("pick",pickup_time)
                                        waiting_time += simulator.waiting_time / simulator.matched_requests_num
                                        print("wait",waiting_time)
                                    
                                    
                                    epoch += 1
                                    total_reward = total_reward / len(TEST_DATE_LIST)
                                    ax.append(epoch)
                                    ay.append(total_reward)
                                    print("total reward",total_reward)
                                    total_request_num = total_request_num / len(TEST_DATE_LIST)
                                    transfer_request_num = transfer_request_num / len(TEST_DATE_LIST)
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
                                    # print("pick",pickup_time)
                                    # print("wait",waiting_time)
                                    # print("matching ratio",matched_request_num/total_request_num)
                                    # print("ocu",occupancy_rate)
                                    record_array = np.array(
                                        [total_reward, matched_transfer_request_num, matched_request_num,
                                         transfer_request_num, long_request_num, matched_long_request_num,
                                         matched_medium_request_num, medium_request_num, matched_short_request_num,
                                         short_request_num, total_request_num,waiting_time,pickup_time,occupancy_rate,occupancy_rate_no_pickup])
                                    
                                    # record_array = np.array([total_reward])

                                    if num == 0:
                                        df.iloc[0, :15] = record_array
                                    else:
                                        df.iloc[num, :15] = (df.iloc[(num - 1), :1].values * num + record_array) / (
                                                    num + 1)

                                    if num % 10 == 0:  # save the result every 10
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
                                plt.plot(ax,ay)
                                plt.plot(ax,ay,'r+')
                                plt.show()
                                df.loc[:num, 'matched_transfer_request_ratio'] = df.loc[:(num),
                                                                                 'matched_transfer_request_num'].values / df.loc[
                                                                                                                          :(
                                                                                                                              num),
                                                                                                                          'matched_request_num'].values
                                df.loc[:(num), 'transfer_long_request_ratio'] = df.loc[:(num),
                                                                                'transfer_request_num'].values / df.loc[
                                                                                                                 :(num),
                                                                                                                 'long_request_num'].values
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
