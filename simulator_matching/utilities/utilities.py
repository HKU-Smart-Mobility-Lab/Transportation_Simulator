# from re import I
# from socket import if_indextoname
import json
import os

import numpy as np
from copy import deepcopy
import random
from random import choice
from shapely.geometry import Point, Polygon
from pymongo.errors import ConnectionFailure

from Transportation_Simulator.simulator_matching.matching_algorithm.dispatch_alg import LD
from math import radians, sin, atan2, cos, acos
from config import *
import math
import pickle
import osmnx as ox
from tqdm import tqdm
import pandas as pd
import sys
from collections import Counter
import pymongo
import time
import scipy.stats as st
from scipy.stats import skewnorm
from collections import defaultdict
import networkx as nx
from config import env_params
import geopandas as gpd
from path import *
from math import sin, cos, sqrt, atan2, radians, degrees, asin, pi
import os
import sys

"""
Here, we load the information of graph network from graphml file.
"""
G = ox.load_graphml(os.path.join(data_path, 'manhattan.graphml'))
gdf_nodes, _ = ox.graph_to_gdfs(G)
lat_list = gdf_nodes['y'].tolist()
lng_list = gdf_nodes['x'].tolist()
node_id = gdf_nodes.index.tolist()

shp_file_path = os.path.join(data_path,f"new_grids_{env_params['grid_num']}", f"new_grids_{env_params['grid_num']}.shp")
result = gpd.read_file(shp_file_path)
result = result.rename(columns={"osmid": "node_id", "x": "lng", "y": "lat"})
# map id to coordinate; map coordinate to node_id
node_id_to_coord = result.set_index('node_id')[['lng', 'lat']].apply(tuple, axis=1).to_dict()
node_coord_to_id = {value: key for key, value in node_id_to_coord.items()}

map_from_node_to_grid = {}
map_from_grid_to_nodes = defaultdict(list)
map_from_grid_to_centroid = {}
for index, row in result.iterrows():
    node_id = row['node_id']
    grid_id = row['grid_id']
    map_from_node_to_grid[node_id] = grid_id
    map_from_grid_to_nodes[grid_id].append(node_id)
    map_from_grid_to_centroid[grid_id] = (row['centroid_x'], row['centroid_y'])

"""
Here, we build the connection to mongodb, which will be used to speed up access to road network information.
"""
myclient = pymongo.MongoClient("mongodb://localhost:27017/") 
try:
    # The ismaster command is cheap and does not require auth.
    myclient.admin.command('ismaster')
    print("MongoDB is connected!")
except ConnectionFailure:
    print("Server not available")
mydb = myclient["manhattan_island"]
mycollection = mydb['od_shortest_path']


df_neighbor_centroid = pd.DataFrame()
zone_id = []
centroid_lng = []
centroid_lat = []
up_b = []
down_b = []
left_b = []
right_b = []

if env_params['repo2any'] == True:
    for id in range(env_params['grid_num']):
        zone_id.append(id)
        current_centroid = map_from_grid_to_centroid[id]
        centroid_lng.append(current_centroid[0])
        centroid_lat.append(current_centroid[1])
    df_neighbor_centroid['zone_id'] = zone_id
    df_neighbor_centroid['centroid_lng'] = centroid_lng
    df_neighbor_centroid['centroid_lat'] = centroid_lat
    direction_0 = [1] * len(zone_id)
    df_available_directions = pd.DataFrame([direction_0] * len(direction_0)).transpose()
    df_available_directions.insert(0,'zone_id',range(len(df_available_directions)))
else:
    if env_params['grid_num'] == 8:
        up = [0, 1, 0, 1, 2, 3, 4, 5]
        down = [2, 3, 4, 5, 6, 7, 6, 7]
        left = [0, 2, 4, 6, 0, 2, 4, 6]
        right = [1, 3, 5, 7, 1, 3, 5, 7]
    elif env_params['grid_num'] == 35:    
        up = [1, 3, 4, 6, 7, 8, 9, 10, 13, 14, 12, 12, 15, 16, 20, 21, 17, 18, 24, 25, 23, 22, 23, 26, 27, 29, 29, 28, 30, 31, 32, 34, 33, 33, 34]
        down = [0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 10, 8, 9, 12, 13, 16, 17, 14, 14, 15, 21, 20, 18, 18, 23, 24, 27, 25, 28, 29, 30, 32, 31]
        left = [0, 1, 1, 3, 3, 5, 5, 6, 8, 8, 9, 10, 14, 13, 13, 14, 16, 17, 18, 17, 20, 20, 20, 25, 24, 24, 25, 27, 28, 27, 30, 30, 32, 33, 33]
        right = [0, 2, 2, 4, 4, 6, 7, 7, 9, 10, 11, 11, 12, 14, 15, 15, 19, 19, 19, 20, 21, 21, 22, 23, 25, 26, 26, 29, 29, 26, 31, 31, 31, 34, 34]

    for id in range(env_params['grid_num']):
        zone_id.append(id)
        current_centroid = map_from_grid_to_centroid[id]
        centroid_lng.append(current_centroid[0])
        centroid_lat.append(current_centroid[1])
        up_b.append(1 if up[id] != id else 0)
        down_b.append(1 if down[id] != id else 0)
        left_b.append(1 if left[id] != id else 0)
        right_b.append(1 if right[id] != id else 0)

    df_neighbor_centroid['zone_id'] = zone_id
    df_neighbor_centroid['centroid_lng'] = centroid_lng
    df_neighbor_centroid['centroid_lat'] = centroid_lat
    df_neighbor_centroid['stay'] = zone_id
    df_neighbor_centroid['up'] = up
    df_neighbor_centroid['right'] = right
    df_neighbor_centroid['down'] = down
    df_neighbor_centroid['left'] = left

    direction_0 = [1] * len(zone_id)
    df_available_directions = pd.DataFrame({
        'zone_id': zone_id,
        'direction_0': direction_0,
        'direction_1': up_b,  # Up
        'direction_2': down_b,  # Down
        'direction_3': left_b,  # Left
        'direction_4': right_b  # Right
    }
    )

# rl for matching
def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    obtain exponential decay epsilons
    :param initial_epsilon:
    :param final_epsilon:
    :param steps:
    :param decay: decay rate
    :param pre_steps: first several epsilons does note decay
    :return:
    """
    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


# rl for matching

# rl for repositioning
def s2e(n, total_len=14):
    n = n.astype(int)
    k = (((n[:, None] & (1 << np.arange(total_len))[::-1])) > 0).astype(np.float64)
    return k


# rl for repositioning


# rl for repositioning
def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    obtain exponential decay epsilons
    :param initial_epsilon:
    :param final_epsilon:
    :param steps:
    :param decay: decay rate
    :param pre_steps: first several epsilons does note decay
    :return:
    """
    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


def get_real_coord_given_current_next_coord(coord1, coord2, d):
    '''
    coord1: current GPS coordinate (may not be the real position)
    coord2: next GPS coordinate
    '''
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(coord1[1])
    lng1 = radians(coord1[0])
    lat2 = radians(coord2[1])
    lng2 = radians(coord2[0])

    # Compute the angular distance d/R
    angular_distance = d / R

    # Compute the initial bearing from point A to point B
    bearing = atan2(sin(lng2 - lng1) * cos(lat2),
                    cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lng2 - lng1))

    # Find the latitude of point A'
    lat_prime = asin(sin(lat1) * cos(angular_distance) +
                     cos(lat1) * sin(angular_distance) * cos(bearing))

    # Find the longitude of point A', considering the change across the Prime Meridian or Date Line
    lng_prime = lng1 + atan2(sin(bearing) * sin(angular_distance) * cos(lat1),
                             cos(angular_distance) - sin(lat1) * sin(lat_prime))

    # Normalize the longitude to be within the range [-180, 180]
    lng_prime = (lng_prime + pi) % (2 * pi) - pi

    # Convert the result from radians to degrees
    lat_prime = degrees(lat_prime)
    lng_prime = degrees(lng_prime)

    return (lng_prime, lat_prime)

def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    manhattan_dis = 0
    lon1, lat1 = coord_1
    lon2, lat2 = coord_2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    r = 6371
    lat_dis = r * acos(min(1.0, cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2))
    lon_dis = r * (lat2 - lat1)
    manhattan_dis = (abs(lat_dis) ** 2 + abs(lon_dis) ** 2) ** 0.5

    return manhattan_dis

def manhattan_dist_estimate(coord_1, coord_2):
    lng1, lat1 = coord_1
    lng2, lat2 = coord_2
    # Radius of the Earth in km
    R = 6371.0
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lng1_rad = math.radians(lng1)
    lng2_rad = math.radians(lng2)
    # Calculate the differences in coordinates
    dlat = abs(lat2_rad - lat1_rad)
    dlng = abs(lng2_rad - lng1_rad)
    # Convert latitude difference to km
    lat_dist_km = dlat * R
    # Use the average latitude to approximate the conversion factor for longitude
    avg_lat_rad = (lat1_rad + lat2_rad) / 2
    lng_dist_km = dlng * R * math.cos(avg_lat_rad)
    
    return lat_dist_km + lng_dist_km



def distance_array(coord_1, coord_2):
    """
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
    coord_1 = np.array(coord_1).astype(float)
    coord_2 = np.array(coord_2).astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(coord_1_array[:, 1]) * np.cos(coord_2_array[:, 1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(a ** 0.5)
    r = 6371
    distance = c * r
    return distance

def haversine_dist_array(coord_1, coord_2):
    # Convert coordinates from degrees to radians
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    
    # Differences in coordinates
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    
    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(coord_1_array[:, 1]) * np.cos(coord_2_array[:, 1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Radius of Earth in kilometers
    r = 6371
    distance = c * r
    return distance

def get_distance_array(origin_coord_array, dest_coord_array):
    """
    :param origin_coord_array: list of coordinates
    :type origin_coord_array:  list
    :param dest_coord_array:  list of coordinates
    :type dest_coord_array:  list
    :return: tuple like (
    :rtype: list
    """
    dis_array = []
    for i in range(len(origin_coord_array)):
        dis = distance(origin_coord_array[i], dest_coord_array[i])
        dis_array.append(dis)
    dis_array = np.array(dis_array)
    return dis_array


def route_generation_array(origin_coord_array, dest_coord_array, reposition=False, mode='rg'):
    """

    :param origin_coord_array: the K*2 type list, the first column is lng, the second column
                                is lat.
    :type origin_coord_array: numpy.array
    :param dest_coord_array: the K*2 type list, the first column is lng, the second column
                                is lat.
    :type dest_coord_array: numpy.array
    :param mode: the mode of generation; if the value of mode is complete, return the last node of route;
                 if the value of mode is drop_end, the last node of route will be dropped.
    :type mode: string
    :return: tuple like (itinerary_node_list, itinerary_segment_dis_list, dis_array)
             itinerary_node_list contains the id of nodes, itinerary_segment_dis_list contains
             the distance between two nodes, dis_array contains the distance from origin node to
             destination node
    :rtype: tuple
    """
    # print("route generation start")
    # origin_coord_list为 Kx2 的array，第一列为lng，第二列为lat；dest_coord_array同理
    # itinerary_node_list的每一项为一个list，包含了对应路线中的各个节点编号
    # itinerary_segment_dis_list的每一项为一个array，包含了对应路线中的各节点到相邻下一节点的距离
    # dis_array包含了各行程的总里程
    origin_node_list = get_nodeId_from_coordinate(origin_coord_array[:, 0], origin_coord_array[:, 1])
    dest_node_list = get_nodeId_from_coordinate(dest_coord_array[:, 0], dest_coord_array[:, 1])
    itinerary_node_list = []
    itinerary_segment_dis_list = []
    dis_array = []
    if mode == 'ma':
        for origin, dest in zip(origin_node_list, dest_node_list):
            itinerary_node_list.append([dest])
            dis = distance(node_id_to_coord[origin], node_id_to_coord[dest])
            itinerary_segment_dis_list.append([dis])
            dis_array.append(dis)
        return itinerary_node_list, itinerary_segment_dis_list, np.array(dis_array)
    
    elif mode == 'rg':
        # 返回完整itinerary
        for origin, dest in zip(origin_node_list, dest_node_list):
            origin = int(origin)
            dest = int(dest)
            query = {
                'origin': origin,
                'destination': dest
            }
            re = mycollection.find_one(query)
            if re:
                ite = re['itinerary_node_list']
            else:
                ite = ox.distance.shortest_path(G, origin, dest, weight='length', cpus=16)
                if ite is None:
                    ite = [origin, dest]
                content = {
                    'origin': origin,
                    'destination': dest,
                    'itinerary_node_list': ite
                }
                try:
                    mycollection.insert_one(content)
                except Exception as e:
                    print(f"Error inserting data for origin: {origin}, destination: {dest}: {e}")
            if ite is not None and len(ite) > 1:
                itinerary_node_list.append(ite)
            else:
                itinerary_node_list.append([origin, dest])
      
        for itinerary_node in itinerary_node_list:
            if itinerary_node is not None:
                itinerary_segment_dis = []
                for i in range(len(itinerary_node) - 1):
                    dis = distance(node_id_to_coord[itinerary_node[i]], node_id_to_coord[itinerary_node[i + 1]])
                    itinerary_segment_dis.append(dis)
                dis_array.append(sum(itinerary_segment_dis))
                itinerary_segment_dis_list.append(itinerary_segment_dis)
            if not reposition: 
                itinerary_node.pop()
    
    dis_array = np.array(dis_array)
    return itinerary_node_list, itinerary_segment_dis_list, dis_array


def get_closed_lng_lat(current_lng_lat_array, target_lng_lat_array):
    ret = []
    for cur_lng_cur_lat, tar_lng_list_tar_lat_list in zip(current_lng_lat_array, target_lng_lat_array):
        cur_lng = cur_lng_cur_lat[0]
        cur_lat = cur_lng_cur_lat[1]
        tar_lng_list = [float(i) for i in tar_lng_list_tar_lat_list[0].split("_")]
        tar_lat_list = [float(i) for i in tar_lng_list_tar_lat_list[1].split("_")]
        final_ln = -999
        final_la = -999
        Mindis = 999999
        for ln, la in zip(tar_lng_list, tar_lat_list):
            cur_dis = distance((cur_lat, cur_lng), (la, ln))
            if cur_dis < Mindis:
                Mindis = cur_dis
                final_ln = ln
                final_la = la
        ret.append(np.array([final_ln, final_la]))

    print(1)
    ret = np.array(ret)
    return ret


class road_network:

    def __init__(self, **kwargs):
        self.params = kwargs

    def load_data(self):
        """
        :param data_path: the path of road_network file
        :type data_path:  string
        :param file_name: the filename of road_network file
        :type file_name:  string
        :return: None
        :rtype:  None
        """
        # 路网格式：节点数字编号（从0开始），节点经度，节点纬度，所在grid id
        self.df_road_network = result

    def get_information_for_nodes(self, node_id_array):
        """
        :param node_id_array: the array of node id
        :type node_id_array:  numpy.array
        :return:  (lng_array,lat_array,grid_id_array), lng_array is the array of longitude;
                lat_array is the array of latitude; the array of node id.
        :rtype: tuple
        """
        index_list = [self.df_road_network[self.df_road_network['node_id'] == item].index[0] for item in node_id_array]
        lng_array = self.df_road_network.loc[index_list, 'lng'].values
        lat_array = self.df_road_network.loc[index_list, 'lat'].values
        grid_id_array = self.df_road_network.loc[index_list, 'grid_id'].values
        return lng_array, lat_array, grid_id_array


def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    :param initial_epsilon: initial epsilon
    :type initial_epsilon: float
    :param final_epsilon: final epsilon
    :type final_epsilon: float
    :param steps: the number of iteration
    :type steps: int
    :param decay: decay rate
    :type decay:  float
    :param pre_steps: the number of iteration of pre randomness
    :type pre_steps: int
    :return: the array of epsilon
    :rtype: numpy.array
    """

    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


def sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1, driver_number_dist=''):
    """
    :param driver_info: the information of driver
    :type driver_info:  pandas.DataFrame
    :param t_initial:   time of initial state
    :type t_initial:    int
    :param t_end:       time of terminal state
    :type t_end:        int
    :param driver_sample_ratio:
    :type driver_sample_ratio:
    :param driver_number_dist:
    :type driver_number_dist:
    :return:
    :rtype:
    """
    # 当前并无随机抽样司机；后期若需要，可设置抽样模块生成sampled_driver_info
    new_driver_info = deepcopy(driver_info)
    sampled_driver_info = new_driver_info.sample(frac=driver_sample_ratio)
    sampled_driver_info['status'] = 3
    loc_con = (sampled_driver_info['start_time'] >= t_initial) & (sampled_driver_info['start_time'] <= t_end)
    sampled_driver_info.loc[loc_con, 'status'] = 0
    sampled_driver_info['target_loc_lng'] = sampled_driver_info['lng']
    sampled_driver_info['target_loc_lat'] = sampled_driver_info['lat']
    sampled_driver_info['target_grid_id'] = sampled_driver_info['grid_id']
    sampled_driver_info['remaining_time'] = 0
    sampled_driver_info['matched_order_id'] = 'None'
    sampled_driver_info['total_idle_time'] = 0
    sampled_driver_info['time_to_last_cruising'] = 0
    sampled_driver_info['current_road_node_index'] = 0
    sampled_driver_info['remaining_time_for_current_node'] = 0
    sampled_driver_info['itinerary_node_list'] = [[] for i in range(sampled_driver_info.shape[0])]
    sampled_driver_info['itinerary_segment_time_list'] = [[] for i in range(sampled_driver_info.shape[0])]

    return sampled_driver_info


def sample_request_num(t_mean, std, delta_t):
    """
    sample request num during delta t
    :param t_mean:
    :param std:
    :param delta_t:
    :return:
    """
    random_num = np.random.normal(t_mean, std, 1)[0] * (delta_t / 100)
    random_int = random_num // 1
    random_reminder = random_num % 1

    rn = random.random()
    if rn < random_reminder:
        request_num = random_int + 1
    else:
        request_num = random_int
    return int(request_num)


def skewed_normal_distribution(u, thegma, k, omega, a, input_size):
    return skewnorm.rvs(a, loc=u, scale=thegma, size=input_size)


def order_dispatch(wait_requests, driver_table, maximal_pickup_distance=950, dispatch_method='LD',
                   method='pickup_distance'):
    """
    :param wait_requests: the requests of orders
    :type wait_requests: pandas.DataFrame

    :param driver_table: the information of online drivers
    :type driver_table:  pandas.DataFrame

    :param maximal_pickup_distance: maximum of pickup distance
    :type maximal_pickup_distance: int

    :param dispatch_method: the method of order dispatch
    :type dispatch_method: string

    :return: matched_pair_actual_indexs: order and driver pair, matched_itinerary: the itinerary of matched driver
    :rtype: tuple
    """
    print("matching driver: ", len(driver_table)," matching orders: ",len(wait_requests))
    con_ready_to_dispatch = (driver_table['status'] == 0) | (driver_table['status'] == 4)
    idle_driver_table = driver_table[con_ready_to_dispatch]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]
    matched_pair_actual_indexs = []
    matched_itinerary = []

    if num_wait_request > 0 and num_idle_driver > 0:
        if dispatch_method == 'LD':
            # generate order driver pairs and corresponding itinerary
            request_array_temp = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']]
            request_array = np.repeat(request_array_temp.values, num_idle_driver, axis=0)
            driver_loc_array_temp = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']]
            driver_loc_array = np.tile(driver_loc_array_temp.values, (num_wait_request, 1))
            dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
            if method == "pickup_distance":
                # weight转换为最大pickup distance - 当前pickup distance
                request_array[:, -1] = maximal_pickup_distance - dis_array + 1
            flag = np.where(dis_array <= maximal_pickup_distance)[0]
            print("flag length", len(flag))
            if len(flag) > 0:
            # step 1: generate order driver pairs
                order_driver_pair = np.vstack(
                    [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
                
                #Andrew: 二分图匹配函数入口
                matched_pair_actual_indexs = LD(order_driver_pair.tolist())
            # step 2: generate itinerary
                request_indexs = np.array(matched_pair_actual_indexs)[:, 0]
                driver_indexs = np.array(matched_pair_actual_indexs)[:, 1]
                request_indexs_new = []
                driver_indexs_new = []
                for index in request_indexs:
                    request_indexs_new.append(
                        request_array_temp[request_array_temp['order_id'] == int(index)].index.tolist()[0])
                for index in driver_indexs:
                    driver_indexs_new.append(
                        driver_loc_array_temp[driver_loc_array_temp['driver_id'] == index].index.tolist()[0])
                request_array_new = np.array(request_array_temp.loc[request_indexs_new])[:, :2]
                driver_loc_array_new = np.array(driver_loc_array_temp.loc[driver_indexs_new])[:, :2]
                itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(
                    driver_loc_array_new, request_array_new, mode=env_params['pickup_mode'])

                matched_itinerary = [itinerary_node_list, itinerary_segment_dis_list, dis_array]

        # TODO: ADD NEW dispatch method
    return matched_pair_actual_indexs, np.array(matched_itinerary)

# Andrew: modified cruising function 
def cruising(eligible_driver_table, mode):
    """
    :param eligible_driver_table: information of eligible driver.
    :type eligible_driver_table: pandas.DataFrame
    :param mode: the type of both-rg-cruising, if type is random; it can cruise to every node with equal
                probability; if the type is nearby, it will cruise to the node in adjacent grid or
                just stay at the original region.
    :type mode: string
    :return: itinerary_node_list, itinerary_segment_dis_list, dis_array
    :rtype: tuple
    """
    dest_array = []
    grid_id_list = eligible_driver_table.loc[:, 'grid_id'].values

    for grid_id in grid_id_list:
        if mode == "global-random":
            random_number = random.choice(df_neighbor_centroid['zone_id'].values)
        elif mode == 'random':
            target = [grid_id]
            neighbors = df_neighbor_centroid[df_neighbor_centroid['zone_id'] == grid_id].iloc[0]
            for direction in ['up', 'down', 'left', 'right']:
                neighbor_id = neighbors[direction]
                if neighbor_id != grid_id:
                    target.append(neighbor_id)
            random_number = choice(target)
        
        record = df_neighbor_centroid[df_neighbor_centroid['zone_id'] == random_number]
        if len(record) > 0:
            dest_array.append([record.iloc[0]['centroid_lng'], record.iloc[0]['centroid_lat']])
        else:
            dest_array.append([df_neighbor_centroid.iloc[0]['centroid_lng'], df_neighbor_centroid.iloc[0]['centroid_lat']])
    
    coord_array = eligible_driver_table.loc[:, ['lng', 'lat']].values
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array, np.array(dest_array))
    return itinerary_node_list, itinerary_segment_dis_list, dis_array

def driver_online_offline_decision(driver_table, current_time):
    # 注意pickup和delivery driver不应当下线
    # 车辆状态：0 cruise (park 或正在cruise)， 1 表示delivery，2 pickup, 3 表示下线, 4 reposition
    # This function is aimed to switch the driver states between 0 and 3, based on the 'start_time' and 'end_time' of drivers
    # Notice that we should not change the state of delievery and pickup drivers, since they are occopied. 
    online_driver_table = driver_table.loc[
        (driver_table['start_time'] <= current_time) & (driver_table['end_time'] > current_time)]
    offline_driver_table = driver_table.loc[
        (driver_table['start_time'] > current_time) | (driver_table['end_time'] <= current_time)]

    online_driver_table = online_driver_table.loc[
        (online_driver_table['status'] != 1) | (online_driver_table['status'] != 2)]
    offline_driver_table = offline_driver_table.loc[
        (offline_driver_table['status'] != 1) | (offline_driver_table['status'] != 2)]
    # print(f'online count: {len(online_driver_table)}, offline count: {len(offline_driver_table)}, total count: {len(driver_table)}')
    new_driver_table = driver_table
    new_driver_table.loc[new_driver_table.isin(online_driver_table.to_dict('list')).all(axis=1), 'status'] = 0
    new_driver_table.loc[new_driver_table.isin(offline_driver_table.to_dict('list')).all(axis=1), 'status'] = 3
    # return new_driver_table
    return new_driver_table


# define the function to get zone_id of segment node


def get_nodeId_from_coordinate(lng, lat):
    """

    :param lat: latitude
    :type lat:  float
    :param lng: longitute
    :type lng:  float
    :return:  id of node
    :rtype: string
    """
    node_list = []
    for i in range(len(lat)):
        if lng[i] not in lng_list or lat[i] not in lat_list:
            x = ox.nearest_nodes(G, lng[i], lat[i])
        else:
            x = node_coord_to_id[(lng[i], lat[i])]
        node_list.append(x)
    return node_list


def KM_for_agent():
    # KM used in agent.py for KDD competition
    pass


def random_actions(possible_directions):
    # make random move and generate a one hot vector
    action = random.sample(possible_directions, 1)[0]
    return action


# rl for matching
# state for sarsa
class State:
    def __init__(self, time_slice: int, grid_id: int):
        self.time_slice = time_slice  # time slice
        self.grid_id = grid_id  # the grid where a taxi stays in

    def __hash__(self):
        return hash(str(self.grid_id) + str(self.time_slice))

    def __eq__(self, other):
        if self.grid_id == other.grid_id and self.time_slice == other.time_slice:
            return True
        return False