import pickle
from functools import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from config import *

# records = pickle.load(open("./output/multi_thread_order.pickle",'rb'))
# order = pickle.load(open("input/dataset.pickle",'rb'))
# data = pd.read_csv("input/multi_thread.csv")
# records = pickle.load(open("toy_records_price_limitation.pickle",'rb'))


# for item in tqdm(records):
#     for record in item.values():
#         if isinstance(record[0],list):
#             print(record[0][-1])

# calculate matching rate
def get_matching_rate(records,order):
    total_num = 0
    matched_num = 0
    # calculate the number of orders
    for time in range(env_params['t_initial'],env_params['t_end'],env_params['delta_t']):
        if time in order.keys():
            total_num += len(order[time])
    # calculate the number of orders that are matched
    for time_item in records:
        for k,v in time_item.items():
            if isinstance(v[0],list):
                matched_num += 1
    return matched_num * 1.0 / total_num


# print("匹配率:",get_matching_rate(records,order))

# 计算乘客平均等待时间（pre_matching time)
def get_avg_prematching_waiting_time(records,order):
    result_list = []
    new_order = []
    for time_item in order:
        for item in order[time_item]:
            new_order.append(item)
    column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                   'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                   'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
    new_order = pd.DataFrame(new_order, columns=column_name)
    order = new_order
    for time_item in records:
        for k,v in time_item.items():
            if isinstance(v[0],list):
                matching_time = v[0][-1]
                start_time = order[order['order_id'] == int(v[0][2])].start_time.values[0]
                result_list.append(matching_time - start_time)
    if len(result_list) != 0:
        return sum(result_list) / len(result_list)
    else:return 0

# print("乘客平均等待时间",get_avg_prematching_waiting_time(records,data))

# 计算乘客接单时间(可选择平均时间）
def get_postmatching_pickup_time(records,avg = True):
    result = []
    for i,time_item in enumerate(records):
        for k,v in time_item.items():
            if isinstance(v[0],list):
                start_time = i*env_params['delta_t'] + env_params['t_initial']
                end_time = v[0][-1]
                for i in range(1,len(v)):
                    # print(v[i])
                    if v[i][-2] == 2.0:
                        end_time = v[i][-1]
                    elif v[i][-2] == 1.0:
                        end_time = v[i][-1]
                        break
                # sys.exit()
                # print(end_time-start_time)
                end_time = min(end_time,env_params['t_end'])
                if end_time > start_time:
                    result.append(end_time - start_time)
    if len(result) != 0:
        if avg is True:

            return sum(result) / len(result)
        else: return sum(result)
    else: return 0

# print("乘客平均接单时间",get_postmatching_pickup_time(records))

# 司机利用率（包括pickup及delivery）
def get_driver_usage_rate(records,start_time,end_time,driver_num):
    occupied_time = 0.0
    for i,time_item in enumerate(records):
        for k,v in time_item.items():
            if isinstance(v[0],list):

                start_time_ = v[0][-1]#i * env_params['delta_t'] + env_params['t_initial']
                end_time_ = v[0][-1]
                for i in range(1,len(v)):
                    if v[i][-1] < end_time:
                        end_time_ = v[i][-1]
                    else:
                        end_time_ = end_time
                        break

                if start_time_ > end_time:
                    print(v)
                    print(start_time_,end_time)
                    sys.exit()
                occupied_time += (end_time - start_time_)

    print("ocu",occupied_time)
    return (occupied_time / (end_time-start_time)) /  driver_num

# print("司机利用率",get_driver_usage_rate(records,36000,79200,500))

# 司机接单率（pickup time / total time)
def get_driver_pickup_ratio(records,start_time,end_time,driver_num):
    return get_postmatching_pickup_time(records) / (end_time - start_time)

# print("司机接单率",get_driver_pickup_ratio(records,36000,79200,500))


def plot_figure():
    start_time = env_params['t_initial']
    end_time = env_params['t_end']
    driver_num = [500,1000,1500]
    matching_rate = []
    pre_matching_waiting_time = []
    post_matching_pickup_time = []
    driver_usage_rate = []
    driver_pickup_rate = []
    for driver in driver_num:
        records = pickle.load(open("statistic/toy_records_no_price_" +str(driver) +".pickle", 'rb'))
        matching_rate.append(get_matching_rate(records,order))
        pre_matching_waiting_time.append((get_avg_prematching_waiting_time(records,data)))
        post_matching_pickup_time.append(get_postmatching_pickup_time(records))
        driver_usage_rate.append(get_driver_usage_rate(records,start_time,end_time,driver))
        driver_pickup_rate.append(get_driver_pickup_ratio(records,start_time,end_time,driver))
    plt.xlabel("The number of driver")
    plt.ylabel("Matching rate")
    plt.plot(driver_num,matching_rate)
    plt.show()
    plt.xlabel("The number of driver")
    plt.ylabel("pre_matching_waiting_time")
    plt.plot(driver_num, pre_matching_waiting_time)
    plt.show()
    plt.xlabel("The number of driver")
    plt.ylabel("post_matching_pickup_time")
    plt.plot(driver_num, post_matching_pickup_time)
    plt.show()
    plt.xlabel("The number of driver")
    plt.ylabel("driver_usage_rate")
    plt.plot(driver_num, driver_usage_rate)
    plt.show()
    plt.xlabel("The number of driver")
    plt.ylabel("driver_pickup_rate")
    plt.plot(driver_num, driver_pickup_rate)
    plt.show()
# plot_figure()