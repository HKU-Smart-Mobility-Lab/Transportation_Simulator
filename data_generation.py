import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from path import *

#toy example

# road network
df_rn = pd.DataFrame(columns = ['node_id', 'lng', 'lat', 'grid_id'])
node_id = [0,1,2,3]
lng_array = [0,0,1,1]
lat_array = [0,1,1,0]
grid_id = [0,1,2,3]
df_rn['node_id'] = node_id
df_rn['lng'] = lng_array
df_rn['lat'] = lat_array
df_rn['grid_id'] = grid_id
pickle.dump(df_rn, open(data_path + 'road_network_information' + '.pickle', 'wb'))

#adj matrix
adj_matrix = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
pickle.dump(adj_matrix, open(data_path + 'adj_matrix' + '.pickle', 'wb'))

#df zone info
df_zi = pd.DataFrame(data=[0,1,2,3], columns = ['grid_id'])
pickle.dump(df_zi, open(data_path + 'zone_info' + '.pickle', 'wb'))

#Driver file
df_driver_info = pd.DataFrame(columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list'])
df_driver_info['driver_id'] = [str(i) for i in range(10)]
df_driver_info['start_time'] = 0
df_driver_info['end_time'] = 101
df_driver_info['lng'] = 0
df_driver_info['lat'] = 0
df_driver_info['grid_id'] = 0
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
pickle.dump(df_driver_info, open(data_path + 'toy_driver_info' + '.pickle', 'wb'))

# request file
#{'0':[...], '1':[...], ...}
order_id_array = list(range(40))
request_1 = ['0', 0, 0, 1, 1, 2, 2, 2, 2, 2, 0, [0, 1, 2], [1, 1, 0]]
request_2 = ['0', 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, [2, 3, 0], [1, 1, 0]]
requests = {}
for i in range(101):
    if i % 5 == 0:
        request_1[0] = str(i//5*2)
        request_2[0] = str(1 + i // 5*2)
        requests[str(i)] = [deepcopy(request_1), deepcopy(request_2)]
    else:
        requests[str(i)] = []
print(requests)
pickle.dump(requests, open(data_path + 'toy_requests' + '.pickle', 'wb'))