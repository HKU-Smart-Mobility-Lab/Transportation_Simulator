import os

from statistic import get_matching_rate
from statistic import get_avg_prematching_waiting_time
from statistic import get_driver_usage_rate
from statistic import get_postmatching_pickup_time,get_driver_pickup_ratio
import sys
import pickle
import pandas as pd
from config import env_params
order = pickle.load(open("input/order.pickle",'rb'))
# column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
#                            'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
#                            'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
# data = pd.DataFrame(order,columns=column_name)

# print(ord)

# print(get_matching_rate(records,order))
# print(get_avg_prematching_waiting_time(records,order))
# print(type(records))
# print(get_postmatching_pickup_time(records))
# print(get_driver_pickup_ratio(records,36000,79200,500))
# print(get_driver_usage_rate(records,36000,79200,500))

path = "./output"
files = os.listdir(path)
files.sort()
files.pop(0)
print(files)
start_time = env_params['t_initial']
end_time = env_params['t_end']
for file in files:
    if os.path.isdir(path + "/" + file):
        print("dir",file)
        sub_files = os.listdir(path + "/" + file)
        for sub_file in sub_files:
            print("sub_file",sub_file)
            if ".pickle" in sub_file and "records" in sub_file:
                records = pickle.load(open(path + "/" + file + "/" + sub_file,'rb'))
                print(records[-1])
                result = []
                driver_num = int(sub_file.split(".")[0].split("_")[-1])
                print(driver_num)
                result.append(get_matching_rate(records,order))
                result.append(get_avg_prematching_waiting_time(records,order))
                result.append(get_postmatching_pickup_time(records))
                result.append(get_driver_pickup_ratio(records,start_time,end_time,driver_num))
                result.append(get_driver_usage_rate(records,start_time,end_time,driver_num))
                print(result)
                sys.exit()



