import pandas as pd
import numpy as np
from path import *
import pickle
from config import *
import osmnx as ox
import time
from tqdm import tqdm
import sys
from math import radians, sin, atan2
# from utilities import *
# records = pickle.load(open('driver_info' + '.pickle', 'rb'))
# zone_info = pickle.load(open(data_path + 'zone_info.pickle', 'rb'))
# records = pickle.load(open('toy_records' + '.pickle', 'rb'))
# adj_mat = pickle.load(open(data_path + 'adj_matrix.pickle', 'rb'))
# request = pickle.load(open(data_path + './output/multi_thread_order.pickle', 'rb'))
# request = pickle.load(open(data_path + 'requests_test.pickle', 'rb'))

# print(request[0])

# data = pickle.load(open('./statistic/records.pickle','rb'))

# for key in request.keys():
#     for item in request[key]:
#         print(item)
#         if item[11] is None or len(item[11]) <= 1 or item[1] not in node_id_to_lat_lng.keys()\
#             or item[4] not in node_id_to_lat_lng.keys():
#             print(item)
#             request[key].remove(item)
#
# pickle.dump(request,open(data_path + './output/multi_thread_order2.pickle', 'wb'))

# for key in request.keys():
#     for item in request[key]:
#         item.insert(-3,300)
#         item.insert(-3,100)
# pickle.dump(request,open(data_path + './output/multi_thread_order2.pickle', 'wb'))
# for item in request:
#     item.insert(-3,300)
#

# a = pd.DataFrame()
# b = pd.DataFrame()
# a['a'] = [[1],2,3]
# a['b'] = [2,5,4]
# b['a'] = [2,[4],5,6]
# b['b'] = [1,3,4,8]
# remain = [False,True,True,True]
# b = b[remain]
# print(a)
# print(b)
# a['b'] = a['a'].values+b['c'].values
# a['b'] = (a['a']+b['a'])
# print(a['b'])
#
#
# sys.exit()


# data = pickle.load(open('./input/dataset.pickle','rb'))
# test = set()
# for time in data.keys():
#     for order in data[time]:
#         print(order)
#         if order[0] in test:
#             print(order)
#         test.add(order[0])
# print(len(test))
# print(len(data))
from utilities import *
count = 0
orders = pickle.load(open('./input/dataset.pickle','rb'))
for time in range(36000, 37500, 5):
    if time in orders.keys():
        for order in orders[time]:
            if order[0] == 28295:
                itinerary_segment_dis = []
                itinerary_node = order[11]
                for i in range(len(itinerary_node) - 1):
                    dis = distance(node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]])
                    itinerary_segment_dis.append(dis)
                print(np.sum(itinerary_segment_dis))
        count += len(orders[time])
records = pickle.load(open('./toy_records_price.pickle','rb'))
matched = 0
print(count)
for time in records:
    for driver in time:
        if isinstance(time[driver][0], list):
            matched += 1
print(matched)
print(matched/count)

