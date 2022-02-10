#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: zhangyuhao
@file: order_generation.py
@time: 2022/2/6 下午2:40
@email: yuhaozhang76@gmail.com
@desc: 
"""
from find_closest_point import find_closest_point
from utilities import get_zone
import pandas as pd
from tqdm import tqdm
import osmnx as ox
import pickle
from utilities import G
import warnings

warnings.filterwarnings("ignore")


def csv_to_pickle(input_file, output_file):
    data_num = 1000

    data = pd.read_csv(input_file)
    data = data.head(data_num)
    ori_lng = data['origin_lng'].tolist()
    ori_lat = data['origin_lat'].tolist()
    dest_lng = data['dest_lng'].tolist()
    dest_lat = data['dest_lat'].tolist()
    pickup_time = data['trip_time'].tolist()
    ori_list = []
    origin_lng = []
    origin_lat = []
    dest_list = []
    dest_openstreetmap_lng = []
    dest_openstreetmap_lat = []
    pickup_time_new = []
    for i in tqdm(range(data_num)):
        try:
            x = ox.distance.get_nearest_node(G, (ori_lat[i], ori_lng[i]), method=None, return_dist=False)
            nodes = ox.graph_to_gdfs(G, edges=False)
            point = nodes['geometry'][x]
            ori_id, temp_ori_lat, temp_ori_lng = x, point.y, point.x
            x = ox.distance.get_nearest_node(G, (dest_lat[i], dest_lng[i]), method=None, return_dist=False)
            nodes = ox.graph_to_gdfs(G, edges=False)
            point = nodes['geometry'][x]
            dest_id, temp_dest_lat, temp_dest_lng = x, point.y, point.x
            ori_list.append(ori_id)
            origin_lng.append(temp_ori_lng)
            origin_lat.append(temp_ori_lat)
            dest_list.append(dest_id)
            dest_openstreetmap_lng.append(temp_dest_lng)
            dest_openstreetmap_lat.append(temp_dest_lat)
            pickup_time_new.append(pickup_time[i])
        except:
            data.drop(index=i)
            print('wrong!')
    data_num = len(origin_lng)
    ori_grid_id = []
    dest_grid_id = []
    for i in range(data_num):
        try:
            ori_grid_id.append(get_zone(origin_lat[i], origin_lng[i]))
            dest_grid_id.append(get_zone(dest_openstreetmap_lat[i], dest_openstreetmap_lng[i]))
        except:
            print(ori_list[i])
            print(dest_list[i])
    print('finish!')
    data['origin_id'] = ori_list
    data['dest_id'] = dest_list
    data['origin_grid'] = ori_grid_id
    data['dest_grid'] = dest_grid_id
    itenerary_list = ox.distance.shortest_path(G, ori_list, dest_list, weight='length', cpus=1)
    data['itinerary_segment_dis_list'] = itenerary_list
    data['itinerary_node_list'] = pd.Series([[] for _ in range(len(data))])
    data['trip_time'] = [0] * data_num
    data['designed_reward'] = [1] * data_num
    data['cancel_prob'] = [0] * data_num

    requests = {}
    for i, item in enumerate(pickup_time):
        if item not in requests.keys():
            requests[item] = [data.iloc[i].tolist()]
        else:
            requests[item].append(data.iloc[i].tolist())
    # requests_new = {}
    # print('sorting!')
    # for key in sorted(requests):
    #     requests_new[key] = requests[key]
    pickle.dump(requests, open(output_file, 'wb'))


def nyu_add_node_id_grid():
    raw_data = pickle.load(open('./input/requests_real_data.pickle', 'rb'))
    day = '2015-07-01'
    data = raw_data[day]
    results = {}
    for second in tqdm(data):
        minute = int(int(second) / 60)
        results[minute] = []
        for order in data[second]:
            ori_lng = order[3]
            ori_lat = order[4]
            dest_lng = order[5]
            dest_lat = order[6]
            x = ox.distance.get_nearest_node(G, (ori_lat, ori_lng), method=None, return_dist=False)
            nodes = ox.graph_to_gdfs(G, edges=False)
            point = nodes['geometry'][x]
            ori_id, temp_ori_lat, temp_ori_lng = x, point.y, point.x
            x = ox.distance.get_nearest_node(G, (dest_lat, dest_lng), method=None, return_dist=False)
            nodes = ox.graph_to_gdfs(G, edges=False)
            point = nodes['geometry'][x]
            dest_id, temp_dest_lat, temp_dest_lng = x, point.y, point.x

            ori_grid_id = get_zone(temp_ori_lat, temp_ori_lng)
            dest_grid_id = get_zone(temp_dest_lat, temp_dest_lng)
            itenerary_list = ox.distance.shortest_path(G, ori_id, dest_id, weight='length', cpus=1)
            # order[7] = distance
            records = ['CMT', order[7], temp_ori_lng, temp_ori_lat, temp_dest_lng, temp_dest_lat, 0, ori_id,
                       dest_id, ori_grid_id, dest_grid_id, itenerary_list, [], 1, 0]

            results[minute].append(records)
    pickle.dump(results, open('./output/nyu_07_01.pickle', 'wb'))


if __name__ == '__main__':
    # output_file = './output/requests_test_109_order.pickle'
    # csv_to_pickle('./input/dataset.csv', output_file)
    nyu_add_node_id_grid()