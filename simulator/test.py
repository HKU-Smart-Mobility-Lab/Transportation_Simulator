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
# a['a'] = [1,2,3]
# a['b'] = [2,5,4]
# print(a[a['a'] * 2 < a['b']])
# a=np.random.normal(5,1,4)
# print(a)
records = pickle.load(open('dataset.pickle','rb'))

#