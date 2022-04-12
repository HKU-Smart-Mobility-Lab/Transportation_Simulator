from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [500,1000,1500,2000,2500,3000]
    max_distance_num = [5]

    cruise_flag = [True,False]
    pickup_flag = ['rg','ma']
    delivery_flag = ['rg','ma']
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:
                for single_driver_num in driver_num:
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['maximal_pickup_distance'] = single_max_distance_num
                        simulator = Simulator(**env_params)
                        simulator.reset()
                        track_record = []

                        t = time.time()
                        for step in tqdm(range(simulator.finish_run_step)):
                            new_tracks = simulator.step()
                            track_record.append(new_tracks)

                        match_and_cancel_track_list = simulator.match_and_cancel_track
                        file_path = './output2/' + pc_flag + "_" + dl_flag + "_" + "cruise="+str(cr_flag)
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                        pickle.dump(track_record, open(file_path + '/records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))
                        pickle.dump(match_and_cancel_track_list,open(file_path+'/match_and_cacel_'+str(single_driver_num)+'.pickle','wb'))
                        file = open(file_path + '/time_statistic.txt', 'a')
                        file.write(str(time.time()-t)+'\n')




