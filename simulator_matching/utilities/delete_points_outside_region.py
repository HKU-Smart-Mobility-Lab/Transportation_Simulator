from copyreg import pickle
import pandas as pd
import numpy as np
import os
from config import env_params



north_lat = env_params['north_lat']
south_lat = env_params['south_lat']
east_lng = env_params['east_lng']
west_lng = env_params['west_lng']

path = "./input/"
sub_files = os.listdir(path)

# for sub_file in sub_files:
#     if sub_file.endswith(".csv") and not sub_file.startswith("driver"):
#         data = pd.read_csv(path + sub_file)
#         print(data.columns)
#         index = data.index[(data['origin_lat'] >= south_lat) & (data['origin_lat'] <= north_lat)
#             & (data['dest_lat'] >= south_lat) & (data['dest_lat'] <= north_lat)
#             & (data['dest_lng'] >= west_lng) & (data['dest_lng']<= east_lng)
#             & (data['origin_lng'] >= west_lng) & (data['origin_lng'] <= east_lng) ]
        
#         data.drop(index=index,inplace=True)
#         data.to_csv(path +"1" + sub_file,index=False)

days = dict{}
for sub_file in sub_files:
    if sub_file.endswith(".pickle") and not sub_file.startswith("driver"):
        data = pickle.load(open(path + sub_file,"rb"))
        days[sub_file.split(".")[0].split("_")[1]] = data

