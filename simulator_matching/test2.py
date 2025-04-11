# # from utilities import *
# import sys
#
# import numpy as np
# import pandas as pd
#
import os
import pickle
#
# records = pickle.load(open('dataset.pickle','rb'))
#
# for time in records.keys():
#     for order in records[time]:
#             print(order)
#             sys.exit()
#
#
# pickle.dump(records,open('dataset.pickle','wb'))

# from azureml.opendatasets import NycTlcYellow
#
# from datetime import datetime
# from dateutil import parser
# import json
# #
# end_date = parser.parse('2015-05-05')
# start_date = parser.parse('2015-05-04')
# nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
# #
# nyc_tlc_df = nyc_tlc.to_pandas_dataframe()
# json_file = open("./order.json",mode='w')
# json.dump(nyc_tlc_df[nyc_tlc_df.columns[7:9]].values.tolist(),json_file,indent=4)

#
# nyc_tlc_df.info()
# print(nyc_tlc_df)
# import sys

# from matplotlib import pyplot as plt


# reward = []
# total_orders = []
# matched_orders = []
# matching_rate = []
# with open("./8-10_100driver.out",'r') as f:
#     for line in f.readlines():
#         if line.strip().startswith("epoch total reward:"):
#             reward.append(float(line.strip().split(":")[-1].strip()))
#         if line.strip().startswith("total orders"):
#             total_orders.append(line.strip().split(" ")[-1])
#         if line.strip().startswith("matched orders"):
#             matched_orders.append(line.strip().split(" ")[-1])

# x_axis = [i for i in range(len(matched_orders))]
# for x,y in zip(matched_orders,total_orders):
#     matching_rate.append(int(x) / int(y))
# print(matching_rate)
# print(len(matching_rate))
# print(len(x_axis))
# plt.plot(x_axis,reward)
# plt.xlabel("epoch")
# plt.ylabel("reward")
# plt.show()


import pandas as pd
import os
from tqdm import tqdm
dir = "input1"
files = os.listdir(dir)
ret  = dict()
for file in files:
    if file.endswith(".csv"):

        column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                    'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                    'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
        data = pd.read_csv(dir + "/" + file)
        date = file.split("_")[1].split(".")[0]
        print(date)
        data.drop(['fare','timestamp','date'],axis=1,inplace=True)
        data['trip_time'] = data['trip_distance'] / 22.788 * 3600
        data['itinerary_node_list'] = data['itinerary_node_list'].apply(lambda x:eval(x))
        data['itinerary_segment_dis_list'] = data['itinerary_segment_dis_list'].apply(lambda x:eval(x))
        data = data.sample(frac=0.1,replace=False)
        day_dic = dict()
        for i in tqdm(range(86401)):
            day_dic[i] = data[data['start_time']==i].values.tolist()
        ret[date] = day_dic
pickle.dump(ret,open("input1/order-11-13-frac=0.1.pickle","wb"))
        # order_id = []
        # origin_id = []
        # origin_lat = []
        # origin_lng = []
        # dest_id = []
        # dest_lat = []
        # dest_lng = []
        # trip_distance = []
        # start_time = []
        # origin_grid_id = []
        # dest_grid_id = []
        # itinerary_node_list = []
        # itinerary_segment_dis_list = []
        # trip_time = []
        # designed_reward = []
        # cancel_prob = []
        # day_dic = dict()
        # for t in tqdm(data.groupby('start_time')):
        #     order_id.append(t[1]['order_id'].values)
        #     origin_id.append(t[1]['origin_id'].values)
        #     origin_lat.append(t[1]['origin_lat'].values)
        #     origin_lng.append(t[1]['origin_lng'].values)
        #     dest_id.append(t[1]['dest_id'].values)
        #     dest_lat.append(t[1]['dest_lat'].values)
        #     dest_lng.append(t[1]['dest_lng'].values)
        #     trip_distance.append(t[1]['trip_distance'].values)
        #     origin_grid_id.append(t[1]['origin_grid_id'].values)
        #     dest_grid_id.append(t[1]['dest_grid_id'].values)
        #     itinerary_node_list.append(t[1]['itinerary_node_list'].values)
        #     itinerary_segment_dis_list.append(t[1]['itinerary_segment_dis_list'].values)
        #     trip_time.append(t[1]['trip_time'].values)
        #     designed_reward.append(t[1]['designed_reward'].values)
        #     cancel_prob.append(t[1]['cancel_prob'].values)
        #     interval_df = pd.DataFrame({'order_id':order_id, 'origin_id':origin_id,'origin_lat':origin_lat,
        #                           'origin_lng':origin_lng,'dest_id':dest_id,'dest_lat':dest_lat,'dest_lng':dest_lng,
        #                           'trip_distance':trip_distance,'origin_grid_id':origin_grid_id,'dest_grid_id':dest_grid_id,
        #                           'itinerary_node_list':itinerary_node_list,'itinerary_segment_dis_list':itinerary_segment_dis_list,
        #                           'trip_time':trip_time,'designed_reward':designed_reward,'cancel_prob':cancel_prob})
        #     day_dic[t[0]] = interval_df
        # print(day_dic)
        # sys.exit()
        # # print(data.head(2)[['itinerary_node_list','start_time','itinerary_segment_dis_list']])
        # sys.exit()
