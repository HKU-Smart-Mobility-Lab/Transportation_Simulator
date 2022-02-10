import pandas as pd
import numpy as np
from path import *
import pickle
from config import *
import osmnx as ox
import time
from tqdm import tqdm

# records = pickle.load(open('toy_driver_info' + '.pickle', 'rb'))
# zone_info = pickle.load(open(data_path + 'zone_info.pickle', 'rb'))
# records = pickle.load(open('toy_records' + '.pickle', 'rb'))
# adj_mat = pickle.load(open(data_path + 'adj_matrix.pickle', 'rb'))
# request = pickle.load(open(data_path + 'driver_info.pickle', 'rb'))
# request = pickle.load(open(data_path + 'requests_test.pickle', 'rb'))

# request = request[:10]
# requests = {}
# for i in range(10):
#     if i % 5 == 0:
#         request_1[0] = str(i//5*2)
#         request_2[0] = str(1 + i // 5*2)
#         requests[str(i)] = [deepcopy(request_1), deepcopy(request_2)]
#     else:
#         requests[str(i)] = []
#
#
# print(records)

# G = ox.graph_from_bbox(env_params['north_lat'], env_params['south_lat'], env_params['east_lng']
                    #    , env_params['west_lng'], network_type='drive_service')
# gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
# lat_list = gdf_nodes['y'].tolist()
# lng_list = gdf_nodes['x'].tolist()
# node_id = gdf_nodes.index.tolist()
# node_id_to_lat_lng = {}
# node_coord_to_id = {}
# for i in range(len(lat_list)):
#     node_id_to_lat_lng[node_id[i]] = (lat_list[i], lng_list[i])
#     node_coord_to_id[(lat_list[i], lng_list[i])] = node_id[i]

# center = (
# (env_params['east_lng'] + env_params['west_lng']) / 2, (env_params['north_lat'] + env_params['south_lat']) / 2)
# radius = max(abs(env_params['east_lng'] - env_params['west_lng']) / 2,
#              abs(env_params['north_lat'] - env_params['south_lat']) / 2)
# side = 4
# interval = 2 * radius / side


def solve( s: str, t: str) -> str:
        # write code here
        s_length,t_length = len(s), len(t)
        max_length = max(s_length,t_length)
        
        result = ''
        flag = 0
        for i in range(max_length):
            
            if i < s_length:
                s_value = int(s[s_length - i - 1])
            else:
                s_value = 0
            if i < t_length:
                t_value = int(t[t_length - i - 1])
            else:
                t_value = 0
            

            result += str(int((s_value + t_value + flag) % 10))
            flag = int((s_value + t_value + flag) / 10)
        if flag:
            result += str(int(flag))
        return result[::-1]

print(solve("1258994789086810959258888307221656691275942378416703768","7007001981958278066472683068554254815482631701142544497"))