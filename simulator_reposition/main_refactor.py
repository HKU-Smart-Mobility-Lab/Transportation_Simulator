
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
from sarsa import SarsaAgent
from A2C import *
from matplotlib import pyplot as plt
import logging
import datetime
import argparse
from reposition_agent import RepositionAgent
from simulator_trainer import RepositionTrainer

if __name__ == "__main__":
    model_timestamp = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    loss_logger = logging.getLogger(f'RL_loss_logger')
    loss_logger.setLevel(logging.DEBUG)

    logger = logging.getLogger(f'my_logger_simulator')
    logger.setLevel(logging.DEBUG)

    # 创建一个文件处理器，并设置日志级别为DEBUG
    os.makedirs("log/loss", exist_ok=True)
    log_file = f"{model_timestamp}_grids{env_params['grid_num']}_drivers{env_params['driver_num']}_repo2any={env_params['repo2any']}_radius{env_params['maximal_pickup_distance']}.log"
    file_handler = logging.FileHandler(os.path.join("log", log_file))
    file_handler.setLevel(logging.DEBUG)

    loss_log_file = f"{model_timestamp}_grids{env_params['grid_num']}_drivers{env_params['driver_num']}_repo2any={env_params['repo2any']}_loss.log"
    loss_file_handler = logging.FileHandler(os.path.join("log","loss", loss_log_file))
    loss_file_handler.setLevel(logging.DEBUG)

    # 创建一个控制台处理器，并设置日志级别为INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个日志格式器，并设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    loss_file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    loss_logger.addHandler(loss_file_handler)

    simulator = Simulator(**env_params)
    # simulator.reset()

    # **训练参数**
    train_config = {
        'num_epochs': NUM_EPOCH,
        'stop_epoch': STOP_EPOCH,
        'init_epsilon': INIT_EPSILON,
        'final_epsilon': FINAL_EPSILON,
        'decay': DECAY,
        'pre_steps': PRE_STEP,
        'save_interval': 20,
        'output_path': "saved_models",
        'model_timestamp': model_timestamp,
        'logger': logger,
        'epsilons': get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, NUM_EPOCH, decay=DECAY, pre_steps=PRE_STEP),
    }

    if not os.path.exists(train_config['output_path']):
        os.makedirs(train_config['output_path'])

    if env_params['rl_mode'] == "reposition":
        if not os.path.exists(load_path):
            os.makedirs(load_path)
        len_time_binary = 18
        len_grid_binary = 7
        if simulator.experiment_mode == 'train':
            epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, NUM_EPOCH, decay=DECAY,
                                                pre_steps=PRE_STEP)
            total_reward_record = np.zeros(NUM_EPOCH)

            # parameters
            # here 8 is the length of binary total time steps
            # 6 is the the length of binary total grids
            # 37 is the length of global state vetors, equal to the numeber of grids
            if simulator.reposition_method == 'A2C':
                if env_params['repo2any'] == True:
                    # df_available_directions are initialized in the utilities.py
                    agent_params = dict(action_dim=env_params['grid_num'], state_dim=(len_time_binary + len_grid_binary),
                                    available_directions=df_available_directions,
                                    load_model=False, 
                                    load_path=load_path,
                                    discount_factor=DISCOUNT_FACTOR,
                                    actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                                    actor_structure=ACTOR_STRUCTURE,
                                    critic_structure=CRITIC_STRUCTURE,
                                    model_name=simulator.reposition_method,
                                    logger=loss_logger)
                else:
                    agent_params = dict(action_dim=5, state_dim=(len_time_binary + len_grid_binary),
                                        available_directions=df_available_directions,
                                        load_model=False, 
                                        load_path=load_path,
                                        discount_factor=DISCOUNT_FACTOR,
                                        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                                        actor_structure=ACTOR_STRUCTURE,
                                        critic_structure=CRITIC_STRUCTURE,
                                        model_name=simulator.reposition_method,
                                        logger=loss_logger)
            elif simulator.reposition_method == 'A2C_global_aware':
                if env_params['repo2any'] == True:
                    agent_params = dict(action_dim=env_params['grid_num'], state_dim=(len_time_binary + len_grid_binary),
                                    available_directions=df_available_directions,
                                    load_model=False, 
                                    load_path=load_path,
                                    discount_factor=DISCOUNT_FACTOR,
                                    actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                                    actor_structure=ACTOR_STRUCTURE,
                                    critic_structure=CRITIC_STRUCTURE,
                                    model_name=simulator.reposition_method,
                                    logger=loss_logger)
                else:
                    agent_params = dict(action_dim=5, state_dim=(
                                len_time_binary + len_grid_binary + 2 *  env_params['grid_num']),
                                        available_directions=df_available_directions,
                                        load_model=False, 
                                        load_path=load_path,discount_factor=DISCOUNT_FACTOR,
                                        actor_lr=ACTOR_LR,
                                        critic_lr=CRITIC_LR,
                                        actor_structure=ACTOR_STRUCTURE,
                                        critic_structure=CRITIC_STRUCTURE,
                                        model_name=simulator.reposition_method,
                                        logger=loss_logger)

            reposition_agent = RepositionAgent(agent_params)
            # **创建 `RepositionTrainer` 并运行训练**
            trainer = RepositionTrainer(simulator, reposition_agent, train_config)
            trainer.train()


        elif simulator.experiment_mode == 'test':
            simulator = Simulator(**env_params)
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
            test_num = 1
            test_interval = 3
            threshold = 10

            df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
            remaining_index_array = np.where(df['total_reward'].values == 0)[0]
            if len(remaining_index_array > 0):
                last_stopping_index = remaining_index_array[0]

            if simulator.reposition_method == 'A2C':
                agent_params = dict(action_dim=5, state_dim=(len_time_binary + len_grid_binary),
                                    available_directions=df_available_directions,
                                    load_model=True, 
                                    load_path=load_path,discount_factor=DISCOUNT_FACTOR,
                                    actor_lr=ACTOR_LR,
                                    critic_lr=CRITIC_LR,
                                    actor_structure=ACTOR_STRUCTURE,
                                    critic_structure=CRITIC_STRUCTURE,
                                    model_name=simulator.reposition_method)
            elif simulator.reposition_method == 'A2C_global_aware':
                agent_params = dict(action_dim=5,
                                    state_dim=(
                                                len_time_binary + len_grid_binary + 2 *  env_params['grid_num']),
                                    available_directions=df_available_directions,
                                    load_model=True, 
                                    load_path=load_path,discount_factor=DISCOUNT_FACTOR,
                                    actor_lr=ACTOR_LR,
                                    critic_lr=CRITIC_LR,
                                    actor_structure=ACTOR_STRUCTURE,
                                    critic_structure=CRITIC_STRUCTURE,
                                    model_name=simulator.reposition_method)
            elif simulator.reposition_method == 'random_cruise' or simulator.reposition_method == 'stay' or simulator.reposition_method == "nearby":
                agent_params = dict(action_dim=5,
                                    state_dim=(
                                                len_time_binary + len_grid_binary + 2 *  env_params['grid_num']),
                                    available_directions=df_available_directions,
                                    load_model=False, 
                                    load_path=load_path,discount_factor=DISCOUNT_FACTOR,
                                    actor_lr=ACTOR_LR,
                                    critic_lr=CRITIC_LR,
                                    actor_structure=ACTOR_STRUCTURE,
                                    critic_structure=CRITIC_STRUCTURE,
                                    model_name='')
            repo_agent = A2C(**agent_params)

            for num in range(last_stopping_index, test_num):
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
                wait_time = 0
                pickup_time = 0
                for date in TEST_DATE_LIST:
                    simulator.experiment_date = date
                    print(date)
                    simulator.reset()
                    for step in tqdm(list(range(simulator.finish_run_step))):
                        # print(step)
                        grid_array, time_array, idle_drivers_by_grid, waiting_orders_by_grid = simulator.step1()

                        action_array = np.array([])
                        if len(grid_array) > 0:
                            # state transformation
                            index_grid = np.where(
                                grid_array.reshape(grid_array.size, 1) == simulator.zone_id_array)[
                                1]
                            index_time = (time_array - simulator.t_initial) // simulator.delta_t
                            binary_index_grid = s2e(index_grid, total_len=len_grid_binary)
                            binary_index_time = s2e(index_time, total_len=len_time_binary)
                            state_array = np.hstack([binary_index_grid, binary_index_time])
                            if simulator.reposition_method == 'A2C_global_aware':
                                global_idle_driver_array = np.tile(idle_drivers_by_grid,
                                                                    [state_array.shape[0], 1])
                                global_wait_orders_array = np.tile(waiting_orders_by_grid,
                                                                    [state_array.shape[0], 1])
                                state_array = np.hstack([state_array, global_idle_driver_array,
                                                            global_wait_orders_array])

                            # make actions
                            action_array = np.zeros(state_array.shape[0])
                            for i in range(len(action_array)):
                                if simulator.reposition_method == 'A2C' or simulator.reposition_method == 'A2C_global_aware':
                                    action = repo_agent.egreedy_actions(state_array[i], -1,
                                                                        index_grid[i])
                                elif simulator.reposition_method == 'random_cruise':
                                    action = repo_agent.egreedy_actions(state_array[i], 2,
                                                                        index_grid[i])
                                elif simulator.reposition_method == 'stay':
                                    action = 0
                                elif simulator.reposition_method == 'nearby':
                                    grid_id = index_grid[i]
                                    target = [grid_id]
                                    
                                    if grid_id - side > 0:
                                        target.append(grid_id - side)                                            
                                    elif int((grid_id + 1) / side) == int(grid_id / side) and grid_id + 1 < side * side:
                                        target.append(grid_id + 1)
                                    elif grid_id + side < side * side:
                                        target.append(grid_id + side)
                                    elif int((grid_id - 1) / side) == int(grid_id / side) and grid_id - 1 > 0:
                                        target.append(grid_id - 1)

                                    action = 0
                                    max_value = -1
                                    for idx,item in enumerate(target):
                                        if item in simulator.grid_value.keys() and simulator.grid_value[item] > max_value:
                                            max_value = simulator.grid_value[item]
                                            action = idx
                        
                                action_array[i] = action

                        simulator.step2(action_array)

                    total_reward += simulator.total_reward
                    total_request_num += simulator.total_request_num
                    matched_request_num += simulator.matched_requests_num
                    long_request_num += simulator.long_requests_num
                    medium_request_num += simulator.medium_requests_num
                    short_request_num += simulator.short_requests_num
                    matched_long_request_num += simulator.matched_long_requests_num
                    matched_medium_request_num += simulator.matched_medium_requests_num
                    matched_short_request_num += simulator.matched_short_requests_num
                    occupancy_rate += simulator.occupancy_rate
                    occupancy_rate_no_pickup = simulator.occupancy_rate_no_pickup
                    wait_time += simulator.waiting_time / simulator.matched_requests_num
                    pickup_time += simulator.pickup_time / simulator.matched_requests_num

                total_reward = total_reward / len(TEST_DATE_LIST)
                total_request_num = total_request_num / len(TEST_DATE_LIST)
                matched_request_num = matched_request_num / len(TEST_DATE_LIST)
                long_request_num /= len(TEST_DATE_LIST)
                medium_request_num /= len(TEST_DATE_LIST)
                short_request_num /= len(TEST_DATE_LIST)
                matched_long_request_num /= len(TEST_DATE_LIST)
                matched_medium_request_num /= len(TEST_DATE_LIST)
                matched_short_request_num /= len(TEST_DATE_LIST)
                occupancy_rate /= len(TEST_DATE_LIST)
                occupancy_rate_no_pickup /= len(TEST_DATE_LIST)
                wait_time /= len(TEST_DATE_LIST)
                pickup_time /= len(TEST_DATE_LIST)

                record_array = np.array(
                    [total_reward, matched_request_num,
                        long_request_num,matched_long_request_num, 
                        matched_medium_request_num,medium_request_num,
                        matched_short_request_num,short_request_num, total_request_num,
                        wait_time,pickup_time,occupancy_rate,occupancy_rate_no_pickup,
                            ])

                if num == 0:
                    df.iloc[0, :13] = record_array
                else:
                    df.iloc[num, :13] = (df.iloc[(num - 1), :13].values * num + record_array) / (
                                num + 1)

                if num % 1 == 0:  # save the result every 10
                    pickle.dump(df, open(load_path + 'performance_record_test_' + env_params[
                        'reposition_method'] + '.pickle', 'wb'))

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



            # pickle.dump(df, open(load_path + 'performance_record_test_' + env_params[
            #     'reposition_method'] + '.pickle', 'wb'))
            print(df.iloc[num, :])
            # print(f"simulator.occupancy_rate_repo:{simulator.occupancy_rate_repo}")




