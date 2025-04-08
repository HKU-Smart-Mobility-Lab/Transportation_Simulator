# import os

# from statistic import get_matching_rate
# from statistic import get_avg_prematching_waiting_time
# from statistic import get_driver_delivery_time,get_driver_delivery_ratio
# from statistic import get_postmatching_pickup_time,get_driver_pickup_ratio
# import sys
# import pickle
# import pandas as pd
# from config import env_params
# from tqdm import tqdm
# order = pickle.load(open("input/order.pickle",'rb'))
# # column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
# #                            'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
# #                            'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
# # data = pd.DataFrame(order,columns=column_name)

# # print(ord)

# # print(get_matching_rate(records,order))
# # print(get_avg_prematching_waiting_time(records,order))
# # print(type(records))
# # print(get_postmatching_pickup_time(records))
# # print(get_driver_pickup_ratio(records,36000,79200,500))
# # print(get_driver_usage_rate(records,36000,79200,500))

# path = "./output3"
# files = os.listdir(path)
# files.sort()
# # print(files)
# start_time = env_params['t_initial']
# end_time = env_params['t_end']
# driver_list = []
# pickup_list = []
# delivery_list = []
# cruise_list = []
# matching_rate_list = []
# waiting_time_list = []
# pickup_time_list = []
# pickup_ratio_list = []
# delivery_time_list = []
# delivery_ratio_list = []
# result_pd = pd.DataFrame()
# # result_pd = pd.read_csv("analysis.csv")
# for file in tqdm(files):
#     if os.path.isdir(path + "/" + file):
#         # print("dir",file)
#         sub_files = os.listdir(path + "/" + file)
#         sub_files.remove("time_statistic.txt")
#         sub_files.sort(key=lambda x:int(x.split(".")[0].split("_")[-1]))
#         for sub_file in sub_files:
#             # print("sub_file",sub_file)
#             if ".pickle" in sub_file and "records" in sub_file and "passenger" not in sub_file:
#                 records = pickle.load(open(path + "/" + file + "/" + sub_file,'rb'))
#                 driver_num = int(sub_file.split(".")[0].split("_")[-1])
#                 pickup_list.append(file.split("_")[0])
#                 delivery_list.append(file.split("_")[1])
#                 cruise_list.append(file.split("_")[-1])
#                 driver_list.append(driver_num)
#                 matching_rate_list.append(get_matching_rate(records,order))
#                 waiting_time_list.append(get_avg_prematching_waiting_time(records,order))
#                 pickup_time_list.append(get_postmatching_pickup_time(records))
#                 pickup_ratio_list.append(get_driver_pickup_ratio(records,start_time,end_time,driver_num))
#                 delivery_time_list.append(get_driver_delivery_time(records,start_time,end_time,driver_num))
#                 delivery_ratio_list.append(get_driver_delivery_ratio(records,start_time,end_time,driver_num))



# result_pd['driver_num'] = driver_list
# result_pd['pickup'] = pickup_list
# result_pd['delivery'] = delivery_list
# result_pd['cruise'] = cruise_list
# result_pd['matching rate'] = matching_rate_list
# result_pd['waiting time'] = waiting_time_list
# result_pd['pickup time'] = pickup_time_list
# result_pd['pickup ratio'] = pickup_ratio_list
# result_pd['delivery time'] = delivery_time_list
# result_pd['delivery ratio'] = delivery_ratio_list

# result_pd.to_csv("analysis2.csv")

# # driver = pickle.load(open("input/driver_info.pickle",'rb'))
# # print(driver.columns)
# import pickle

# driver_pd = pickle.load(open("input/driver_distribution.pickle","rb"))
# print(driver_pd.head(5))


# generate new_orders

# import pandas as pd
# import numpy as np
# import os
# import pickle
# from tqdm import tqdm
# def transfer(s):
#     s = s.strip("[").strip("]").strip(" ")
#     new_list = []
#     for i in s.split(","):
#         if s != '':
#             new_list.append(int(i))
#         return new_list

# def transfer2(s):
#     s = s.strip("[").strip("]").strip(" ")
#     new_list = []
#     for i in s.split(","):
#         if s != '':
#             new_list.append(float(i))
#         return new_list


# north_lat = 40.8845,
# south_lat = 40.6968,
# east_lng = -74.0831,
# west_lng = -73.8414,

# date_res ={}
# path = "input/"
# files = os.listdir(path)
# for file in files:
#     if file.endswith("csv") and file.startswith("NYU"):
#         tmp_pd = pd.read_csv(path + file).sample(frac=0.1,replace=False)
#         tmp_pd = tmp_pd[(tmp_pd['origin_lat'] >= south_lat) & (tmp_pd['dest_lat'] >= south_lat) & (tmp_pd['origin_lat'] <= north_lat) & (tmp_pd['dest_lat'] <= north_lat)\
#            & (tmp_pd['origin_lng'] >= east_lng) & (tmp_pd['dest_lng'] >= east_lng) & (tmp_pd['origin_lng'] <= west_lng) & (tmp_pd['dest_lng'] <= west_lng)]
#         tmp_pd['itinerary_node_list'] = tmp_pd['itinerary_node_list'].apply(transfer)
#         tmp_pd['itinerary_segment_dis_list'] = tmp_pd['itinerary_segment_dis_list'].apply(transfer2)
#         date = file.split(".")[0].split("_")[1]

#         day_res = {}

        
#         time = tmp_pd.start_time.unique()
#         for t in tqdm(time):
#             day_res[t] = tmp_pd[tmp_pd['start_time']==t][['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
#                                 'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
#                                 'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']].values.tolist()
#         date_res[date] = day_res

# pickle.dump(date_res,open("input1/new_order_frac=0.1.pickle","wb"))

from matplotlib import pyplot as plt
import pickle
def get_train_curve(file):
    with open(file,"r") as f:
        lines = f.readlines()
        ret = []
        
        tmp = []
        for line in lines:
            if "reward" in line:
                tmp.append(float(line.split(" ")[-1]))
                if len(tmp) % 10 == 0:
                    print(sum(tmp) / 10)
                    ret.append(sum(tmp) / 10)
                    tmp = []
        
        x = [i for i in range(len(ret))]
        print(len(x),len(ret))
        plt.plot(x,ret,'ro-')
        plt.show()

def get_test_curve(file):
    with open(file,"r") as f:
        lines = f.readlines()
        ret = []

        for line in lines:
            if "total reward" in line:
                ret.append(float(line.split(" ")[-1]))

        
        x = [i for i in range(len(ret))]
        print(len(x),len(ret))
        plt.plot(x,ret,'ro-')
        plt.show()

def get_test_value(file):
    with open(file,"r") as f:
        lines = f.readlines()[-18:-1]
        for line in lines:
            print(line.split(" ")[-1].strip())

    

if __name__ == "__main__":
    # get_train_curve("1201rl_train.txt")
    # get_test_curve("1124rl_test.out")
    get_test_value("12-13-random.txt")
    # data = pickle.load(open("output3/order_record-1103.pickle","rb"))
    # print(data.columns)
    # print(data.loc[:,["trip_time","trip_distance","designed_reward"]])
