import os

from statistic import get_matching_rate
from statistic import get_avg_prematching_waiting_time
from statistic import get_driver_delivery_time,get_driver_delivery_ratio
from statistic import get_postmatching_pickup_time,get_driver_pickup_ratio
import sys
import pickle
import pandas as pd
from config import env_params
from tqdm import tqdm
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

path = "./output3"
files = os.listdir(path)
files.sort()
# print(files)
start_time = env_params['t_initial']
end_time = env_params['t_end']
driver_list = []
pickup_list = []
delivery_list = []
cruise_list = []
matching_rate_list = []
waiting_time_list = []
pickup_time_list = []
pickup_ratio_list = []
delivery_time_list = []
delivery_ratio_list = []
result_pd = pd.DataFrame()
# result_pd = pd.read_csv("analysis.csv")
for file in tqdm(files):
    if os.path.isdir(path + "/" + file):
        # print("dir",file)
        sub_files = os.listdir(path + "/" + file)
        sub_files.remove("time_statistic.txt")
        sub_files.sort(key=lambda x:int(x.split(".")[0].split("_")[-1]))
        for sub_file in sub_files:
            # print("sub_file",sub_file)
            if ".pickle" in sub_file and "records" in sub_file and "passenger" not in sub_file:
                records = pickle.load(open(path + "/" + file + "/" + sub_file,'rb'))
                driver_num = int(sub_file.split(".")[0].split("_")[-1])
                pickup_list.append(file.split("_")[0])
                delivery_list.append(file.split("_")[1])
                cruise_list.append(file.split("_")[-1])
                driver_list.append(driver_num)
                matching_rate_list.append(get_matching_rate(records,order))
                waiting_time_list.append(get_avg_prematching_waiting_time(records,order))
                pickup_time_list.append(get_postmatching_pickup_time(records))
                pickup_ratio_list.append(get_driver_pickup_ratio(records,start_time,end_time,driver_num))
                delivery_time_list.append(get_driver_delivery_time(records,start_time,end_time,driver_num))
                delivery_ratio_list.append(get_driver_delivery_ratio(records,start_time,end_time,driver_num))



result_pd['driver_num'] = driver_list
result_pd['pickup'] = pickup_list
result_pd['delivery'] = delivery_list
result_pd['cruise'] = cruise_list
result_pd['matching rate'] = matching_rate_list
result_pd['waiting time'] = waiting_time_list
result_pd['pickup time'] = pickup_time_list
result_pd['pickup ratio'] = pickup_ratio_list
result_pd['delivery time'] = delivery_time_list
result_pd['delivery ratio'] = delivery_ratio_list

result_pd.to_csv("analysis2.csv")

# driver = pickle.load(open("input/driver_info.pickle",'rb'))
# print(driver.columns)
