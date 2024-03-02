import pandas as pd
import numpy as np
import pickle
import random
from copy import deepcopy
from path import *
from utilities import *
from find_closest_point import *
import math
import matplotlib.pyplot as plt
import matplotlib

center = (
(env_params['east_lng'] + env_params['west_lng']) / 2, (env_params['north_lat'] + env_params['south_lat']) / 2)
radius = max(abs(env_params['east_lng'] - env_params['west_lng']) / 2,
             abs(env_params['north_lat'] - env_params['south_lat']) / 2)
side = env_params['side']
interval = 2 * radius / side


def get_zone(lat, lng):
    """
    :param lat: the latitude of coordinate
    :type : float
    :param lng: the longitude of coordinate
    :type lng: float
    :return: the id of zone that the point belongs to
    :rtype: float
    """
    if lat < center[1]:
        i = math.floor(side / 2) - math.ceil((center[1] - lat) / interval) + side % 2
    else:
        i = math.floor(side / 2) + math.ceil((lat - center[1]) / interval) - 1

    if lng < center[0]:
        j = math.floor(side / 2) - math.ceil((center[0] - lng) / interval) + side % 2
    else:
        j = math.floor(side / 2) + math.ceil((lng - center[0]) / interval) - 1
    return i * side + j


# Driver file
def generate_driver(gdf_nodes, result):
    df_driver_info = pd.DataFrame(columns=['driver_id', 'start_time', 'end_time', 'lng', 'lat','node_id', 'grid_id', 'status',
                                           'target_loc_lng', 'target_loc_lat', 'target_node_id','target_grid_id', 'remaining_time',
                                           'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                                           'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list'])

    gdf_nodes = gdf_nodes.sample(n=env_params['driver_num'] * 2, replace = True)
    lng_list = gdf_nodes['x'].tolist()
    lat_list = gdf_nodes['y'].tolist()
    id_list = gdf_nodes.index.tolist()
    df_driver_info['lng'] = lng_list[:env_params['driver_num']]
    df_driver_info['lat'] = lat_list[:env_params['driver_num']]
    origin_id_list = id_list[:env_params['driver_num']]
    df_driver_info['driver_id'] = [str(i) for i in range(env_params['driver_num'])]
    df_driver_info['start_time'] = env_params['t_initial']
    df_driver_info['end_time'] = env_params['t_end']
    df_driver_info['node_id'] = origin_id_list
    df_driver_info['grid_id'] = [int(result[result['node_id'] == x].iloc[0]['grid_id']) for x in origin_id_list]
    df_driver_info['status'] = 0
    df_driver_info['target_loc_lng'] = lng_list[env_params['driver_num']:]
    df_driver_info['target_loc_lat'] = lat_list[env_params['driver_num']:]
    target_id_list = id_list[env_params['driver_num']:]
    df_driver_info['target_node_id'] = target_id_list
    df_driver_info['target_grid_id'] = [int(result[result['node_id'] == x].iloc[0]['grid_id']) for x in target_id_list]
    df_driver_info['remaining_time'] = 0
    df_driver_info['matched_order_id'] = 'None'
    df_driver_info['total_idle_time'] = 0
    df_driver_info['time_to_last_cruising'] = 0
    df_driver_info['current_road_node_index'] = 0
    df_driver_info['remaining_time_for_current_node'] = 0
    df_driver_info['itinerary_node_list'] = [[] for _ in range(len(df_driver_info))]
    df_driver_info['itinerary_segment_dis_list'] = [[] for _ in range(len(df_driver_info))]
    print(df_driver_info.shape)
    pickle.dump(df_driver_info, open(data_path + 'hongkong_driver_info' + '.pickle', 'wb'))


def sample_by_order_distribution(driver_num):
    zone_num = dict()
    order_num = 0
    order = pickle.load(open('./input1/new_order.pickle', 'rb'))
    for date in order.keys():
        for i in range(10800, 14400):
            if i in order[date].keys():
                for single_order in order[date][i]:
                    order_num += 1
                    zone = get_zone(single_order[2], single_order[3])
                    if zone in zone_num:
                        zone_num[zone] += 1
                    else:
                        zone_num[zone] = 1

    driver = pickle.load(open('./input/driver_info.pickle', 'rb'))
    driver_info = pd.DataFrame()
    for i, zone in enumerate(zone_num):
        if i == len(zone_num)-1:
            driver_sample = driver_num - len(driver_info)
            driver_sample = min(driver_sample, len(driver[driver['grid_id'] == zone]))
            temp_driver_info = driver[driver['grid_id'] == zone].sample(n=driver_sample)
            driver = driver[driver['grid_id'] != zone]
            driver_info = pd.concat([driver_info, temp_driver_info], axis=0)
            left_driver = driver.sample(n=driver_num-len(driver_info))
            driver_info = pd.concat([driver_info, left_driver], axis=0)
            break
        frac = zone_num[zone]/order_num
        driver_sample = int(driver_num * frac)
        driver_sample = min(driver_sample, len(driver[driver['grid_id'] == zone]))
        temp_driver_info = driver[driver['grid_id'] == zone].sample(n=driver_sample)
        driver = driver[driver['grid_id'] != zone]
        driver_info = pd.concat([driver_info, temp_driver_info], axis=0)
    print(driver_info.shape)
    driver_info['driver_id'] = [i for i in range(driver_num)]
    pickle.dump(driver_info, open('./input1/driver_distribution_3am-4am' + '.pickle', 'wb'))


def statistic_order_num_per_hour():
    order = pickle.load(open('./input/order.pickle', 'rb'))
    hour = dict()
    for i in tqdm(range(0, 86400)):
        if i in order.keys():
            if int(i/3600) in hour.keys():
                hour[int(i/3600)] += 1
            else:
                hour[int(i / 3600)] = 1
    print(hour)
    plt.plot(hour.keys(), hour.values())
    plt.show()


if __name__ == '__main__':
    # generate_driver(gdf_nodes=gdf_nodes, result=result)
    sample_by_order_distribution(5000)
    # statistic_order_num_per_hour()