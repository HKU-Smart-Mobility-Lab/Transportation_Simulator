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


def csv_to_pickle(input_file, output_file):
    data_num = 109

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
    for i in tqdm(range(data_num)):
        try:
            ori_id, temp_ori_lat, temp_ori_lng = find_closest_point(ori_lat[i], ori_lng[i])
            dest_id, temp_dest_lat, temp_dest_lng = find_closest_point(dest_lat[i], dest_lng[i])
            ori_list.append(ori_id)
            origin_lng.append(temp_ori_lng)
            origin_lat.append(temp_ori_lat)
            dest_list.append(dest_id)
            dest_openstreetmap_lng.append(temp_dest_lng)
            dest_openstreetmap_lat.append(temp_dest_lat)
        except:
            print('wrong!')
    data_num = len(origin_lng)
    lng_max = max(data['origin_lng'].max(), data['dest_lng'].max())
    lng_min = min(data['origin_lng'].min(), data['dest_lng'].min())
    lat_max = max(data['origin_lat'].max(), data['dest_lat'].max())
    lat_min = min(data['origin_lat'].min(), data['dest_lat'].min())

    center = ((lng_max + lng_min) / 2, (lat_max + lat_min) / 2)
    print("center: ", center)
    radius = max((lng_max - lng_min) / 2, (lat_max - lat_min) / 2)
    side = 2
    interval = 2 * radius / side
    print("interval", interval)
    # G = ox.graph_from_bbox(center[1] + radius, center[1] - radius, center[0] - radius, center[0] + radius,
    #                        network_type='drive_service')
    # gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    # lat_list = gdf_nodes['y'].tolist()
    # lng_list = gdf_nodes['x'].tolist()
    # node_id = gdf_nodes.index.tolist()
    # node_id_to_grid_id = {}
    ori_grid_id = []
    dest_grid_id = []
    for i in range(data_num):
        try:
            ori_grid_id.append(get_zone(origin_lat[i], origin_lng[i], center, side, interval))
            dest_grid_id.append(get_zone(dest_openstreetmap_lat[i], dest_openstreetmap_lng[i], center, side, interval))
        except:
            print(ori_list[i])
            print(dest_list[i])
    print('finish!')
    # for i in tqdm(range(len(node_id))):
    #     node_id_to_grid_id[node_id[i]] = get_zone(lat_list[i], lng_list[i], center, side, interval)
    # for i in range(data_num):
    #     try:
    #         ori_grid_id.append(node_id_to_grid_id[ori_list[i]])
    #         dest_grid_id.append(node_id_to_grid_id[dest_list[i]])
    #     except:
    #         print(ori_list[i])
    #         print(dest_list[i])
    data['origin_id'] = ori_list
    data['dest_id'] = dest_list
    data['origin_grid'] = ori_grid_id
    data['dest_grid'] = dest_grid_id
    G = ox.graph_from_bbox(lat_max + 0.1, lat_min - 0.1, lng_min - 0.1, lng_max + 0.1, network_type='drive_service')
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


if __name__ == '__main__':
    output_file = './output/requests_test_10_order.pickle'
    csv_to_pickle('./input/dataset.csv', output_file)