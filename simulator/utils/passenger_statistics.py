import pickle
from functools import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from config import *

'''
    This util script was under 'simulator', now it is in 'simulator/test'. You may need
    to update path related codes in order to successfully run the code without errors.
'''

passenger_pd = pickle.load(open("../output3/ma_ma_cruise=False/passenger_records_driver_num_1000.pickle",'rb'))
order = pickle.load(open("../input/order.pickle",'rb'))
matched_pd = passenger_pd[passenger_pd['matching_time'] != 0]


total_num = 0
for time in range(env_params['t_initial'], env_params['t_end']):
    if time in order.keys():
        total_num += len(order[time])
matched_num = len(matched_pd)
print("matching rate",matched_num / total_num)

matching_time_list = matched_pd.matching_time.values.tolist()
pickup_time_list = matched_pd.pickup_end_time.values.tolist()
delivery_time_list = matched_pd.delivery_end_time.values.tolist()

pickup_time = 0
delivery_time = 0

for mt,pt in zip(matching_time_list,pickup_time_list):
    if pt >= mt:
        pickup_time += (pt-mt)
    else:
        print(mt,pt)
    # delivery_time += (dt-pt)

print("pickup time",pickup_time / len(matched_pd))
print("matched time",delivery_time / len(matched_pd))





