from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time

# python D:\Feng\drl_subway_comp\main.py

if __name__ == "__main__":
    simulator = Simulator(**env_params)
    test_num = 1
    track_record = []
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for num in range(test_num):
        simulator.reset()
        for step in range(simulator.finish_run_step):
            print(step)
            new_tracks = simulator.step()
            track_record.append(new_tracks)
    pickle.dump(track_record, open(data_path + 'toy_records' + '.pickle', 'wb'))



