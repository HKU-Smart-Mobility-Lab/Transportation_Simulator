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
    simulator = Simulator(**env_params)
    test_num = 1
    track_record = []
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for num in tqdm(range(test_num)):
        simulator.reset()
        for step in range(simulator.finish_run_step):
            new_tracks = simulator.step()
            track_record.append(new_tracks)

    # pickle.dump(track_record, open(data_path + 'toy_records_no_price_500' + '.pickle', 'wb'))

    pickle.dump(track_record, open(data_path + 'toy_records_price' + '.pickle', 'wb'))




