import pandas as pd
from path import *
import pickle

'''
    This util script was under 'simulator', now it is in 'simulator/test'. You may need
    to update path related codes in order to successfully run the code without errors.
'''

filename = "dataset.csv"
data = pd.read_csv(data_path + "input/"+ filename)
print(len(data['ID']))

column_name = ['order_id', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat', 'immediate_reward',
                          'trip_distance', 'trip_time', 'designed_reward', 'dest_grid_id', 'cancel_prob',
                          'itinerary_node_list', 'itinerary_segment_dis_list']

request_info = pd.DataFrame(columns = column_name)
request_info['order_id'] = data['ID']
request_info['origin_lng'] = data['origin_lng']
request_info['origin_lat'] = data['origin_lat']
request_info['dest_lng'] = data['dest_lng']
request_info['dest_lat'] = data['dest_lat']
request_info['trip_distance'] = data['trip_distance']
request_info['immediate_reward'] = 0
request_info['trip_time'] = 0
request_info['designed_reward'] = 0
request_info['dest_grid_id'] = 0
request_info['cancel_prob'] = 0
request_info['itinerary_node_list'] = pd.Series([[] for i in range(len(data))])
# request_info['itinerary_segment_dis_list'] =
print(request_info)
pickle.dump(request_info,open(data_path + "output/" + 'requests' + '.pickle', 'wb'))