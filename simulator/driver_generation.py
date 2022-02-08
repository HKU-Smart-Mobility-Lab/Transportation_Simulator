import pandas as pd
import numpy as np
import pickle
import random
from copy import deepcopy
from path import *
from utilities import *
from find_closest_point import *
import sys


#Driver file
df_driver_info = pd.DataFrame(columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat','node_id' 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list'])
sample_gdf_nodes = gdf_nodes.sample(n=env_params['driver_num'],replace = True)

df_driver_info['driver_id'] = [str(i) for i in range(10)]
df_driver_info['start_time'] = env_params['t_initial']
df_driver_info['end_time'] = env_params['t_end']
df_driver_info['lng'] = sample_gdf_nodes['x'].tolist()
df_driver_info['lat'] = sample_gdf_nodes['y'].tolist()
id_list = sample_gdf_nodes.index.tolist()
df_driver_info['node_id'] = id_list
df_driver_info['grid_id'] = [int(result[result['node_id'] == x].iloc[0]['grid_id']) for x in id_list]
df_driver_info['status'] = 0
df_driver_info['target_loc_lng'] = 0
df_driver_info['target_loc_lat'] = 0
df_driver_info['target_grid_id'] = 0
df_driver_info['remaining_time'] = 0
df_driver_info['matched_order_id'] = 'None'
df_driver_info['total_idle_time'] = 0
df_driver_info['time_to_last_cruising'] = 0
df_driver_info['current_road_node_index'] = 0
df_driver_info['remaining_time_for_current_node'] = 0
df_driver_info['itinerary_node_list'] = [[] for _ in range(len(df_driver_info))]
df_driver_info['itinerary_segment_dis_list'] = [[] for _ in range(len(df_driver_info))]
print(df_driver_info)
pickle.dump(df_driver_info, open(data_path + 'driver_info' + '.pickle', 'wb'))