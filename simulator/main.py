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
from utilities import get_exponential_epsilons
from sarsa import SarsaAgent
# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [500]
    max_distance_num = [5]

    cruise_flag = [False]
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
                        # simulator.reset()
                        track_record = []
                        t = time.time()
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
                                           'matched_request_ratio']
                            test_num = 500
                            test_interval = 20
                            threshold = 5
                            df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
                            # df = pickle.load(open(load_path + 'performance_record_test_' + env_params['method'] + '.pickle', 'rb'))
                            remaining_index_array = np.where(df['total_reward'].values == 0)[0]
                            if len(remaining_index_array > 0):
                                last_stopping_index = remaining_index_array[0]

                            for num in range(last_stopping_index, test_num):
                                print('num: ', num)
                                # simulator = Simulator(**env_params)
                                agent = {}
                                if simulator.method in ['sarsa', 'sarsa_no_subway', 'sarsa_travel_time',
                                                        'sarsa_travel_time_no_subway',
                                                        'sarsa_total_travel_time', 'sarsa_total_travel_time_no_subway']:
                                    agent = SarsaAgent(**sarsa_params)
                                    agent.load_parameters(
                                        load_path + 'episode_4000\\sarsa_q_value_table_epoch_4000.pickle')

                                total_reward = 0
                                total_request_num = 0
                                long_request_num = 0
                                medium_request_num = 0
                                short_request_num = 0
                                transfer_request_num = 0
                                matched_request_num = 0
                                matched_long_request_num = 0
                                matched_medium_request_num = 0
                                matched_short_request_num = 0
                                matched_transfer_request_num = 0
                                occupancy_rate = 0
                                occupancy_rate_no_pickup = 0

                                for date in TEST_DATE_LIST:
                                    simulator.experiment_date = date
                                    simulator.reset()
                                    start_time = time.time()
                                    for step in range(simulator.finish_run_step):
                                        dispatch_transitions = simulator.step(agent)
                                    end_time = time.time()

                                    total_reward += simulator.total_reward
                                    total_request_num += simulator.total_request_num
                                    transfer_request_num += simulator.transfer_request_num
                                    occupancy_rate += simulator.occupancy_rate
                                    matched_request_num += simulator.matched_requests_num
                                    matched_transfer_request_num += simulator.matched_transferred_requests_num
                                    long_request_num += simulator.long_requests_num
                                    medium_request_num += simulator.medium_requests_num
                                    short_request_num += simulator.short_requests_num
                                    matched_long_request_num += simulator.matched_long_requests_num
                                    matched_medium_request_num += simulator.matched_medium_requests_num
                                    matched_short_request_num += simulator.matched_short_requests_num
                                    occupancy_rate_no_pickup += simulator.occupancy_rate_no_pickup
                                total_reward = total_reward / len(TEST_DATE_LIST)
                                total_request_num = total_request_num / len(TEST_DATE_LIST)
                                transfer_request_num = transfer_request_num / len(TEST_DATE_LIST)
                                occupancy_rate = occupancy_rate / len(TEST_DATE_LIST)
                                matched_request_num = matched_request_num / len(TEST_DATE_LIST)
                                matched_transfer_request_num = matched_transfer_request_num / len(TEST_DATE_LIST)
                                long_request_num = long_request_num / len(TEST_DATE_LIST)
                                medium_request_num = medium_request_num / len(TEST_DATE_LIST)
                                short_request_num = short_request_num / len(TEST_DATE_LIST)
                                matched_long_request_num = matched_long_request_num / len(TEST_DATE_LIST)
                                matched_medium_request_num = matched_medium_request_num / len(TEST_DATE_LIST)
                                matched_short_request_num = matched_short_request_num / len(TEST_DATE_LIST)
                                occupancy_rate_no_pickup = occupancy_rate_no_pickup / len(TEST_DATE_LIST)

                                record_array = np.array(
                                    [total_reward, matched_transfer_request_num, matched_request_num,
                                     transfer_request_num, long_request_num, matched_long_request_num,
                                     matched_medium_request_num, medium_request_num, matched_short_request_num,
                                     short_request_num, total_request_num])

                                if num == 0:
                                    df.iloc[0, :11] = record_array
                                else:
                                    df.iloc[num, :11] = (df.iloc[(num - 1), :11].values * num + record_array) / (
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
                            pickle.dump(df,
                                        open(load_path + 'performance_record_test_' + env_params['method'] + '.pickle',
                                             'wb'))
                            print(df.iloc[index, :])

                            # np.savetxt(load_path + "supply_dist_" + simulator.method + ".csv", simulator.driver_spatial_dist, delimiter=",")

                        elif simulator.experiment_mode == 'train':
                            print("training process")
                            epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, 2500, decay=DECAY,
                                                                pre_steps=PRE_STEP)
                            epsilons = np.concatenate([epsilons, np.zeros(NUM_EPOCH - 2500)])
                            # epsilons = np.zeros(NUM_EPOCH)
                            total_reward_record = np.zeros(NUM_EPOCH)
                            if simulator.method in ['sarsa', 'sarsa_no_subway', 'sarsa_travel_time',
                                                    'sarsa_travel_time_no_subway', 'sarsa_total_travel_time',
                                                    'sarsa_total_travel_time_no_subway']:
                                agent = SarsaAgent(**sarsa_params)
                                if FLAG_LOAD:
                                    agent.load_parameters(
                                        load_path + 'episode_1800\\sarsa_q_value_table_epoch_1800.pickle')
                            for epoch in range(NUM_EPOCH):
                                date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                                simulator.experiment_date = date
                                simulator.reset()
                                start_time = time.time()
                                for step in range(simulator.finish_run_step):
                                    dispatch_transitions = simulator.rl_step(agent, epsilons[epoch])
                                    print("dispatch_transitions",dispatch_transitions)
                                    agent.perceive(dispatch_transitions)
                                end_time = time.time()
                                total_reward_record[epoch] = simulator.total_reward
                                print('epoch:', epoch)
                                print('epoch running time: ', end_time - start_time)
                                print('epoch total reward: ', simulator.total_reward)

                                if epoch % 200 == 0:  # save the result every 200 epochs
                                    agent.save_parameters(epoch)

                                if epoch % 200 == 0:  # plot and save training curve
                                    # plt.plot(list(range(epoch)), total_reward_record[:epoch])
                                    pickle.dump(total_reward_record, open(load_path + 'training_results_record', 'wb'))

                        for step in tqdm(range(simulator.finish_run_step)):
                            new_tracks = simulator.step()
                            track_record.append(new_tracks)

                        match_and_cancel_track_list = simulator.match_and_cancel_track
                        file_path = './output3/' + pc_flag + "_" + dl_flag + "_" + "cruise="+str(cr_flag)
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                        pickle.dump(track_record, open(file_path + '/records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))
                        pickle.dump(simulator.requests, open(file_path + '/passenger_records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))

                        pickle.dump(match_and_cancel_track_list,open(file_path+'/match_and_cacel_'+str(single_driver_num)+'.pickle','wb'))
                        file = open(file_path + '/time_statistic.txt', 'a')
                        file.write(str(time.time()-t)+'\n')




