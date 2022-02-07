import pandas as pd
import numpy as np
from path import *
import pickle

# records = pickle.load(open('toy_driver_info' + '.pickle', 'rb'))
# zone_info = pickle.load(open(data_path + 'zone_info.pickle', 'rb'))
# records = pickle.load(open('road_network_information' + '.pickle', 'rb'))
# adj_mat = pickle.load(open(data_path + 'adj_matrix.pickle', 'rb'))
# request = pickle.load(open(data_path + 'toy_driver_info.pickle', 'rb'))
request = pickle.load(open(data_path + 'requests_test.pickle', 'rb'))

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
print(request)

