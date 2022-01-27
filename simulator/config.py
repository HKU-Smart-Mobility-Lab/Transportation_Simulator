env_params = {
't_initial' :0,
't_end' : 100,
'delta_t' : 1,
'vehicle_speed' : 1,
'repo_speed' : 1, #目前的设定需要与vehicl speed保持一致
'order_sample_ratio' : 1,
'order_generation_mode' : 'sample_from_base',
'driver_sample_ratio' : 1,
'maximum_wait_time_mean' : 10,
'maximum_wait_time_std' : 0,
'maximal_pickup_distance' : 4,
'request_interval': 1,
'cruise_flag' : True,
'max_idle_time' : 5,
'cruise_mode': 'random',
'reposition_flag': True,
'eligible_time_for_reposition' : 10,
'reposition_mode': '',
'track_recording_flag' : True,
'driver_far_matching_cancel_prob_file' : 'driver_far_matching_cancel_prob',
'input_file_path':'input/dataset.csv',
'request_file_name' : 'output/requests', #'toy_requests',
'driver_file_name' : 'toy_driver_info',
'road_network_file_name' : 'road_network_information.pickle',
'dispatch_method': 'LD', #LD: lagarange decomposition method designed by Peibo Duan
'method': 'instant_reward_no_subway',
'simulator_mode' : 'toy_mode',
'experiment_mode' : 'test',
'side':4,
'road_information_mode':'load'
}

