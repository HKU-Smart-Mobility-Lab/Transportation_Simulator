from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    driver_num = [500,1000,1500,2000,2500]
    max_distance_num = [0.5,1,2,3]
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for single_driver_num in driver_num:
        for single_max_distance_num in max_distance_num:
            env_params['driver_num'] = single_driver_num
            env_params['maximal_pickup_distance'] = single_max_distance_num
            simulator = Simulator(**env_params)
            simulator.reset()
            track_record = []
            t = time.time()
            for step in tqdm(range(simulator.finish_run_step)):
                new_tracks = simulator.step()
                track_record.append(new_tracks)
            pickle.dump(track_record, open('./output/records_driver_num_'+str(single_driver_num)+'_distance_' +
                                           str(single_max_distance_num) + '.pickle', 'wb'))
            file = open('./output/time_statistic.txt', 'a')
            file.write(str(time.time()-t)+'\n')




