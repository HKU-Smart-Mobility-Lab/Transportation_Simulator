env_params = {
't_initial' :36000,
't_end' : 36500,
'delta_t' : 5,  # s
'vehicle_speed' : 22.68,   # km / h
'repo_speed' : 1, #目前的设定需要与vehicl speed保持一致
'order_sample_ratio' : 1,
'order_generation_mode' : 'sample_from_base',
'driver_sample_ratio' : 1,
'maximum_wait_time_mean' : 10,
'maximum_wait_time_std' : 0,
"maximum_pickup_time_passenger_can_tolerate_mean":600,  # s
"maximum_pickup_time_passenger_can_tolerate_std":200, # s
"maximum_price_passenger_can_tolerate_mean":150, # ￥
"maximum_price_passenger_can_tolerate_std":50,  # ￥
'maximal_pickup_distance' : 1,  # km
'request_interval': 5,  # s
'cruise_flag' : False,
'max_idle_time' : 1,
'cruise_mode': 'random',
'reposition_flag': False,
'eligible_time_for_reposition' : 10, # s
'reposition_mode': '',
'track_recording_flag' : True,
'driver_far_matching_cancel_prob_file' : 'driver_far_matching_cancel_prob',
'input_file_path':'input/dataset.csv',
'request_file_name' : 'input/order', #'toy_requests',
'driver_file_name' : 'input/driver_info',
'road_network_file_name' : 'road_network_information.pickle',
'dispatch_method': 'LD', #LD: lagarange decomposition method designed by Peibo Duan
'method': 'instant_reward_no_subway',
'simulator_mode' : 'toy_mode',
'experiment_mode' : 'test',
'driver_num':500,
'side':4,
'price_per_km':5,  # ￥ / km
'road_information_mode':'load',
'north_lat': 40.8845,
'south_lat': 40.6968,
'east_lng': -74.0831,
'west_lng': -73.8414
}
