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
side = 4
interval = 2 * radius / side

# database
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["route_network"]
   
mycollect = mydb['route_list']

# define the function to get zone_id of segment node
def get_zone(lat, lng):
    """
    :param lat: the latitude of coordinate
    :type : float
    :param lng:
    :type lng:
    :return:
    :rtype:
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


# given origin and destination, return itenarary

# 在这里加入ra
def get_distance_array(origin_coord_array, dest_coord_array):
    dis_array = []
    for i in range(len(origin_coord_array)):
        dis = distance(origin_coord_array[i], dest_coord_array[i])
        dis_array.append(dis)
    dis_array = np.array(dis_array)
    return dis_array


def route_generation_array(origin_coord_array, dest_coord_array, mode='complete'):
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
    if mode == 'test':
        for origin, dest in zip(origin_node_list, dest_node_list):
            itinerary_node_list.append([origin])
            dis = distance(node_id_to_lat_lng[origin], node_id_to_lat_lng[dest])
            itinerary_segment_dis_list.append([dis])
            dis_array.append(dis)
        return itinerary_node_list, itinerary_segment_dis_list, np.array(dis_array)

    if mode == 'complete':
        # 返回完整itinerary
        for origin,dest in zip(origin_node_list,dest_node_list):
            data = {
                'node': str(origin) + str(dest)
            }
            re = mycollect.find_one(data)['itinerary_node_list']
            if re:
                ite = re
            else:
                ite = ox.distance.shortest_path(G, origin, dest, weight='length', cpus=16)
            if ite is not None and len(ite) > 1:
                itinerary_node_list.append(ite)
            else:
                itinerary_node_list.append([origin,dest])
        # itinerary_node_list = ox.distance.shortest_path(G, origin_node_list, dest_node_list, weight='length', cpus=16)

        for itinerary_node in itinerary_node_list:
            if itinerary_node is not None:
                itinerary_segment_dis = [0]
                for i in range(len(itinerary_node) - 1):
                    # dis = nx.shortest_path_length(G, node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]], weight='length')

                    dis = distance(node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]])
                    itinerary_segment_dis.append(dis)
                dis_array.append(sum(itinerary_segment_dis))
                itinerary_segment_dis_list.append(itinerary_segment_dis)

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


# class GridSystem:
#     def __init__(self, **kwargs):
#         pass

#     def load_data(self, data_path):
#         # self.df_zone_info = pickle.load(open(data_path + 'zone_info.pickle', 'rb'))
#         self.df_zone_info = pickle.load(open(data_path + 'zone_info.pickle', 'rb'))
#         self.num_grid = self.df_zone_info.shape[0]
#         self.adj_mat = pickle.load(open(data_path + 'adj_matrix.pickle', 'rb'))

#     def get_basics(self):
#         # output: basic information about the grid network
#         return self.num_grid


class road_network:
    def __init__(self, **kwargs):
        self.params = kwargs

    def load_data(self, data_path, file_name):
        # 路网格式：节点数字编号（从0开始），节点经度，节点纬度，所在grid id
        # columns = ['node_id', 'lng', 'lat', 'grid_id']
        # self.df_road_network = pickle.load(open(data_path + file_name, 'rb'))
        self.df_road_network = result

    # def generate_road_info(self):
        # data = pd.read_csv(self.params['input_file_path'])
        # lng_max = max(data['origin_lng'].max(), data['dest_lng'].max())
        # lng_min = min(data['origin_lng'].min(), data['dest_lng'].min())
        # lat_max = max(data['origin_lat'].max(), data['dest_lat'].max())
        # lat_min = min(data['origin_lat'].min(), data['dest_lat'].min())
        # center = ((lng_max + lng_min) / 2, (lat_max + lat_min) / 2)
        # interval = max((lng_max - lng_min), (lat_max - lat_min)) / self.params['side']
        # G = ox.graph_from_bbox(lat_max + 0.1, lat_min - 0.1, lng_min - 0.1, lng_max + 0.1, network_type='drive_service')
        # gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
        # nodelist = []
        # lat_list = gdf_nodes['y'].tolist()
        # lng_list = gdf_nodes['x'].tolist()
        # result = pd.DataFrame()
        # result['lat'] = lat_list
        # result['lng'] = lng_list
        # result['node_id'] = gdf_nodes.index.tolist()
        # result['node_id'] = gdf_nodes.index.tolist()[:10]
        # for i in range(len(result)):
        #     nodelist.append(get_zone(lat_list[i], lng_list[i], center, side, interval))
        # result['grid_id'] = nodelist
        # pickle.dump(result, open('./road_network_information' + '.pickle', 'wb'))

    def get_information_for_nodes(self, node_id_array):
        # print(Counter(node_id_array))
        index_list = [self.df_road_network[self.df_road_network['node_id'] == item].index[0] for item in node_id_array]
        lng_array = self.df_road_network.loc[index_list,'lng'].values
        lat_array = self.df_road_network.loc[index_list,'lat'].values
        grid_id_array = self.df_road_network.loc[index_list,'grid_id'].values
        return lng_array, lat_array, grid_id_array


def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
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


# reposition，暂时先不定义
def reposition(eligible_driver_table, mode):
    # 需用到route_generation_array
    itinerary_node_list = []
    itinerary_segment_dis_list = []
    dis_array = np.array([])
    # toy example
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
    print('repositino route end')
    return itinerary_node_list, itinerary_segment_dis_list, dis_array


# cruising，暂时先不定义
def cruising(eligible_driver_table, mode):
    # 需用到route_generation_array
    itinerary_node_list = []
    itinerary_segment_dis_list = []
    dis_array = np.array([])

    # toy example
    dest_array = []
    grid_id_list = eligible_driver_table.loc[:, 'grid_id'].values
    # print("eligible_driver_table",eligible_driver_table)
    # sys.pause()
    for grid_id in (grid_id_list):
        target = []
        if int((grid_id -1)/side) == int(grid_id/side) and grid_id-1 > 0:
            target.append(grid_id-1)
        if int((grid_id +1)/side) == int(grid_id/side) and grid_id+1 < side*side:
            target.append(grid_id+1)
        if grid_id+side < side*side:
            target.append(grid_id+side)
        if grid_id-side > 0:
            target.append(grid_id-side)
        random_number = choice(target)
        if True:
            record = result[result['grid_id'] == random_number]
        elif mode == 'nearby':
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
    con_ready_to_dispatch = (driver_table['status'] == 0) | (driver_table['status'] == 4)
    idle_driver_table = driver_table[con_ready_to_dispatch]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]
    matched_pair_actual_indexs = []
    matched_itinerary = []

    if num_wait_request > 0 and num_idle_driver > 0:
        if dispatch_method == 'LD':
            # generate order driver pairs and corresponding itinerary
            request_array_temp = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']].values
            request_array = np.repeat(request_array_temp, num_idle_driver, axis=0)
            driver_loc_array_temp = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']].values
            driver_loc_array = np.tile(driver_loc_array_temp, (num_wait_request, 1))

            # itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(request_array[:, :2],
            #                                                                                     driver_loc_array[:, :2],
            #                                                                                   mode='drop_end')
            dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])

            flag = np.where(dis_array <= maximal_pickup_distance)[0]
            if len(flag) > 0:
                order_driver_pair = np.vstack(
                    [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
                matched_pair_actual_indexs = LD(order_driver_pair.tolist())
                request_indexs = np.array(matched_pair_actual_indexs,dtype=np.int16)[:, 0]
                driver_indexs = np.array(matched_pair_actual_indexs)[:, 1]
                request_array = np.array(request_array_temp)
                driver_loc_array = np.array(driver_loc_array_temp)
                request_array_new = request_array[np.where(request_indexs==request_array[:,2])][:,:2]
                driver_loc_array_new = driver_loc_array[np.where(driver_indexs==driver_loc_array[:,2])][:,:2]
                itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(
                    driver_loc_array_new, request_array_new, mode='test')
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
    node_list = []
    for i in range(len(lat)):
        x = node_coord_to_id[(lat[i],lng[i])]
        node_list.append(x)
    return node_list

#############################################################################





