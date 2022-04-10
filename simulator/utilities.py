# from re import I
# from socket import if_indextoname
import numpy as np
from copy import deepcopy
import random
from random import choice
from dispatch_alg import LD
from math import radians, sin, atan2
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


"""
Here, we load the information of graph network from graphml file.
"""
G = ox.load_graphml('./input/graph.graphml')
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
lat_list = gdf_nodes['y'].tolist()
lng_list = gdf_nodes['x'].tolist()
node_id = gdf_nodes.index.tolist()
node_id_to_lat_lng = {}
node_coord_to_id = {}
for i in range(len(lat_list)):
    node_id_to_lat_lng[node_id[i]] = (lat_list[i], lng_list[i])
    node_coord_to_id[(lat_list[i], lng_list[i])] = node_id[i]

center = (
(env_params['east_lng'] + env_params['west_lng']) / 2, (env_params['north_lat'] + env_params['south_lat']) / 2)
radius = max(abs(env_params['east_lng'] - env_params['west_lng']) / 2,
             abs(env_params['north_lat'] - env_params['south_lat']) / 2)
side = env_params['side']
interval = 2 * radius / side

"""
Here, we build the connection to mongodb, which will be used to speed up access to road network information.
"""
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["route_network"]
   
mycollect = mydb['route_list']

# define the function to get zone_id of segment node
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


result = pd.DataFrame()
nodelist = []
result['lat'] = lat_list
result['lng'] = lng_list
result['node_id'] = gdf_nodes.index.tolist()
for i in range(len(result)):
    nodelist.append(get_zone(lat_list[i], lng_list[i]))
result['grid_id'] = nodelist

# print(time.time()-t)
def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    lat1, lon1, = coord_1
    lat2, lon2 = coord_2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    r = 6371

    alat = sin(dlat / 2) ** 2
    clat = 2 * atan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r

    alon = sin(dlon / 2) ** 2
    clon = 2 * atan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r

    manhattan_dis = abs(lat_dis) + abs(lon_dis)

    return manhattan_dis


def distance_array(coord_1, coord_2):
    """
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
    coord_1 = coord_1.astype(float)
    coord_2 = coord_2.astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = np.abs(coord_2_array[:, 0] - coord_1_array[:, 0])
    dlat = np.abs(coord_2_array[:, 1] - coord_1_array[:, 1])
    r = 6371

    alat = np.sin(dlat / 2) ** 2
    clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r

    alon = np.sin(dlon / 2) ** 2
    clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r

    manhattan_dis = np.abs(lat_dis) + np.abs(lon_dis)

    return manhattan_dis


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



def route_generation_array(origin_coord_array, dest_coord_array, mode='rg'):
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
    origin_node_list = get_nodeId_from_coordinate(origin_coord_array[:, 1], origin_coord_array[:, 0])
    dest_node_list = get_nodeId_from_coordinate(dest_coord_array[:, 1], dest_coord_array[:, 0])
    itinerary_node_list = []
    itinerary_segment_dis_list = []
    dis_array = []
    if mode == 'ma':
        for origin, dest in zip(origin_node_list, dest_node_list):
            itinerary_node_list.append([dest])
            dis = distance(node_id_to_lat_lng[origin], node_id_to_lat_lng[dest])
            itinerary_segment_dis_list.append([dis])
            dis_array.append(dis)
        return itinerary_node_list, itinerary_segment_dis_list, np.array(dis_array)

    if mode == 'rg':
        # 返回完整itinerary
        for origin,dest in zip(origin_node_list,dest_node_list):
            data = {
                'node': str(origin) + str(dest)
            }
            re = mycollect.find_one(data)
            if re:
                ite = [int(item) for item in re['itinerary_node_list'].strip('[').strip(']').split(', ')]
            else:
                ite = ox.distance.shortest_path(G, origin, dest, weight='length', cpus=16)
            if ite is not None and len(ite) > 1:
                itinerary_node_list.append(ite)
            else:
                itinerary_node_list.append([origin,dest])
        # itinerary_node_list = ox.distance.shortest_path(G, origin_node_list, dest_node_list, weight='length', cpus=16)
        for itinerary_node in itinerary_node_list:

            if itinerary_node is not None:
                itinerary_segment_dis = []
                for i in range(len(itinerary_node) - 1):
                    # dis = nx.shortest_path_length(G, node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]], weight='length')

                    dis = distance(node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]])
                    itinerary_segment_dis.append(dis)
                dis_array.append(sum(itinerary_segment_dis))
                itinerary_segment_dis_list.append(itinerary_segment_dis)
            itinerary_node.pop()

        # a toy example
        # for i in range(origin_coord_array.shape[0]):
        #     origin_lng = origin_coord_array[i, 0]
        #     if origin_lng == 0:
        #         itinerary_node = [0,1,2]
        #         itinerary_segment_dis = [1,1,0]
        #         dis = 2
        #     elif origin_lng == 1:
        #         itinerary_node = [2,3,0]
        #         itinerary_segment_dis = [1, 1, 0]
        #         dis = 2
        # itinerary_node_list.append(itinerary_node)
        # itinerary_segment_dis_list.append(itinerary_segment_dis)
        # dis_array.append(dis)
        # dis_array = np.array(dis_array)
    elif mode == 'drop_end':
        # 对itineray node_list和itineray segment_time_list中的各个item，把末尾节点给drop掉

        # a toy example
        # for i in range(origin_coord_array.shape[0]):
        #     origin_lng = origin_coord_array[i, 0]
        #     if origin_lng == 0:
        #         itinerary_node = [0, 1]
        #         itinerary_segment_dis = [1, 1]
        #         dis = 2
        #     elif origin_lng == 1:
        #         itinerary_node = [2, 3]
        #         itinerary_segment_dis = [1, 1]
        #         dis = 2
        # itinerary_node_list.append(itinerary_node)
        # itinerary_segment_dis_list.append(itinerary_segment_dis)
        # dis_array.append(dis)
        # dis_array = np.array(dis_array)
        for origin,dest in zip(origin_node_list, dest_node_list):
            # data = {
            #     'node': str(origin)+str(dest)
            # }
            # re = mycollect.find_one(data)['itinerary_node_list']
            # if re:
            #     ite = re
            # else:
            ite = ox.distance.shortest_path(G, origin, dest, weight='length', cpus=16)
            if ite is not None and len(ite) > 1:
                itinerary_node_list.append(ite)
            else:
                itinerary_node_list.append([origin,dest])

        for itinerary_node in itinerary_node_list:
            itinerary_segment_dis = []
            for i in range(len(itinerary_node) - 1):
                # dis = nx.shortest_path_length(G, node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]], weight='length')
                try:
                    dis = distance(node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]])
                except:
                    print("itinerary exception")
           
                itinerary_segment_dis.append(dis)
            itinerary_node.pop()
            dis_array.append(sum(itinerary_segment_dis))
            # if len(itinerary_node) != itinerary_segment_dis:
            #     print(len(itinerary_node))
            #     print(itinerary_segment_dis)
            itinerary_segment_dis_list.append(itinerary_segment_dis)

    dis_array = np.array(dis_array)
    return itinerary_node_list, itinerary_segment_dis_list, dis_array

class road_network:

    def __init__(self, **kwargs):
        self.params = kwargs


    def load_data(self, data_path, file_name):
        """
        :param data_path: the path of road_network file
        :type data_path:  string
        :param file_name: the filename of road_network file
        :type file_name:  string
        :return: None
        :rtype:  None
        """
        # 路网格式：节点数字编号（从0开始），节点经度，节点纬度，所在grid id
        # columns = ['node_id', 'lng', 'lat', 'grid_id']
        # self.df_road_network = pickle.load(open(data_path + file_name, 'rb'))
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
        lng_array = self.df_road_network.loc[index_list,'lng'].values
        lat_array = self.df_road_network.loc[index_list,'lat'].values
        grid_id_array = self.df_road_network.loc[index_list,'grid_id'].values
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
    sampled_driver_info = new_driver_info
    sampled_driver_info['status'] = 3
    loc_con = sampled_driver_info['start_time'] <= t_initial
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



def reposition(eligible_driver_table, mode):
    """
    :param eligible_driver_table:
    :type eligible_driver_table:
    :param mode:
    :type mode:
    :return:
    :rtype:
    """
    random_number = np.random.randint(0, side * side - 1)
    dest_array = []
    for _ in range(len(eligible_driver_table)):
        record = result[result['grid_id'] == random_number]
        if len(record) > 0:
            dest_array.append([record.iloc[0]['lng'], record.iloc[0]['lat']])
        else:
            dest_array.append([result.iloc[0]['lng'], result.iloc[0]['lat']])
    coord_array = eligible_driver_table.loc[:, ['lng', 'lat']].values
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array, np.array(dest_array))
    return itinerary_node_list, itinerary_segment_dis_list, dis_array



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
    for grid_id in (grid_id_list):
        if mode == "random":
            random_number = random.randint(0,side*side-1)
        elif mode == 'nearby':
            target = []
            if int((grid_id - 1) / side) == int(grid_id / side) and grid_id - 1 > 0:
                target.append(grid_id - 1)
            if int((grid_id + 1) / side) == int(grid_id / side) and grid_id + 1 < side * side:
                target.append(grid_id + 1)
            if grid_id + side < side * side:
                target.append(grid_id + side)
            if grid_id - side > 0:
                target.append(grid_id - side)
            random_number = choice(target)
        record = result[result['grid_id'] == random_number]
        if len(record) > 0:
            dest_array.append([record.iloc[0]['lng'], record.iloc[0]['lat']])
        else:
            dest_array.append([result.iloc[0]['lng'], result.iloc[0]['lat']])
    coord_array = eligible_driver_table.loc[:, ['lng', 'lat']].values
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array,
                                                                                        np.array(dest_array))
    return itinerary_node_list, itinerary_segment_dis_list, dis_array


def order_dispatch(wait_requests, driver_table, maximal_pickup_distance=950, dispatch_method='LD'):
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

            # itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(request_array[:, :2],
            #                                                                                     driver_loc_array[:, :2],
            #                                                                                   mode='drop_end')
            dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])

            flag = np.where(dis_array <= maximal_pickup_distance)[0]
            if len(flag) > 0:
                order_driver_pair = np.vstack(
                    [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
                matched_pair_actual_indexs = LD(order_driver_pair.tolist())
                request_indexs = np.array(matched_pair_actual_indexs)[:, 0]
                driver_indexs = np.array(matched_pair_actual_indexs)[:, 1]
                request_indexs_new = []
                driver_indexs_new = []
                for index in request_indexs:
                    request_indexs_new.append(request_array_temp[request_array_temp['order_id'] == int(index)].index.tolist()[0])
                for index in driver_indexs:
                    driver_indexs_new.append(driver_loc_array_temp[driver_loc_array_temp['driver_id'] == index].index.tolist()[0])
                request_array_new = np.array(request_array_temp.loc[request_indexs_new])[:,:2]
                driver_loc_array_new = np.array(driver_loc_array_temp.loc[driver_indexs_new])[:,:2]
                itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(
                    driver_loc_array_new, request_array_new, mode=env_params['pickup_mode'])
                itinerary_node_list_new = []
                itinerary_segment_dis_list_new = []
                dis_array_new = []
                # for item in matched_pair_actual_indexs:
                #     index = int(item[3])
                #     itinerary_node_list_new.append(itinerary_node_list[index])
                #     itinerary_segment_dis_list_new.append(itinerary_segment_dis_list[index])
                #     dis_array_new.append(dis_array[index])

                matched_itinerary = [itinerary_node_list, itinerary_segment_dis_list, dis_array]
    return matched_pair_actual_indexs, np.array(matched_itinerary)


def driver_online_offline_decision(driver_table, current_time):

    # 注意pickup和delivery driver不应当下线
    new_driver_table = driver_table
    return new_driver_table


# define the function to get zone_id of segment node



def get_nodeId_from_coordinate(lat, lng):
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
        x = node_coord_to_id[(lat[i],lng[i])]
        node_list.append(x)
    return node_list

#############################################################################





