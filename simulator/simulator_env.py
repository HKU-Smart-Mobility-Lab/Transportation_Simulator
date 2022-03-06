import numpy as np
import pandas as pd
from copy import deepcopy
import random
from config import *
from simulator_pattern import *
from numpy.random import choice
import timeit
import time
import math
import  sys
from utilities import *

class Simulator:
    def __init__(self, **kwargs):

        # basic parameters: time & sample
        self.t_initial = kwargs['t_initial']
        self.t_end = kwargs['t_end']
        self.delta_t = kwargs['delta_t']
        self.vehicle_speed = kwargs['vehicle_speed']
        self.repo_speed = kwargs.pop('repo_speed', 3)

        # order generation
        self.order_sample_ratio = kwargs['order_sample_ratio']
        self.order_generation_mode = kwargs['order_generation_mode']
        self.request_interval = kwargs['request_interval']

        # wait cancel
        self.maximum_wait_time_mean = kwargs.pop('maximum_wait_time_mean', 120)
        self.maximum_wait_time_std = kwargs.pop('maximum_wait_time_std', 0)

        # driver cancel far matching
        #self.driver_cancel_prob_array = pickle.load(open(data_path + kwargs['driver_far_matching_cancel_prob_file'] + '.pickle', 'rb'))

        # maximal pickup distance
        self.maximal_pickup_distance = kwargs['maximal_pickup_distance']

        # track recording
        self.track_recording_flag = kwargs['track_recording_flag']
        self.new_tracks = {}

        # pattern
        self.simulator_mode = kwargs.pop('simulator_mode', 'simulator_mode')
        self.experiment_mode = kwargs.pop('experiment_mode', 'test')
        self.request_file_name = kwargs['request_file_name']
        self.driver_file_name = kwargs['driver_file_name']
        pattern_params = {'simulator_mode': self.simulator_mode, 'request_file_name': self.request_file_name,
                          'driver_file_name': self.driver_file_name}
        pattern = SimulatorPattern(**pattern_params)

        # grid system initialization
        # self.GS = GridSystem()
        # self.GS.load_data(data_path)
        # self.num_zone = self.GS.get_basics()

        # road network
        road_network_file_name = kwargs['road_network_file_name']
        self.RN = road_network()
        self.RN.load_data(data_path, road_network_file_name)

        # dispatch method
        self.dispatch_method = kwargs['dispatch_method']
        self.method = kwargs['method']

        # cruise and reposition related parameters
        self.cruise_flag = kwargs['cruise_flag']
        self.cruise_mode = kwargs['cruise_mode']
        self.max_idle_time = kwargs['max_idle_time']

        self.reposition_flag = kwargs['reposition_flag']
        self.reposition_mode = kwargs['reposition_mode']
        self.eligible_time_for_reposition = kwargs['eligible_time_for_reposition']

        # get steps
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)

        # request tables
        self.request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id','dest_lat', 'dest_lng','trip_distance','start_time','origin_grid_id','dest_grid_id',
                                'itinerary_node_list', 'itinerary_segment_dis_list','maximum_pickup_time_passenger_can_tolerate','maximum_price_passenger_can_tolerate','trip_time', 'cancel_prob', 't_matched',
                                'pickup_time', 'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time', 
                                'pickup_distance']
                      
        self.wait_requests = None
        self.matched_requests = None

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']
        self.driver_table = None
        self.driver_sample_ratio = kwargs['driver_sample_ratio']

        # order and driver databases
        self.driver_info = pattern.driver_info
        self.driver_info['grid_id'] = self.driver_info['grid_id'].values.astype(int)
        self.request_all = pattern.request_all
      

    def initial_base_tables(self):
        # time/day
        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        # construct driver table
        self.driver_table = sample_all_drivers(self.driver_info, self.t_initial, self.t_end, self.driver_sample_ratio)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)

        # construct order table
        self.request_databases = deepcopy(self.request_all)
        self.wait_requests = pd.DataFrame(columns=self.request_columns)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)

    def reset(self):
        self.initial_base_tables()


    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes, matched_itinerary):
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)
        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes, columns=['order_id', 'driver_id', 'weight', 'flag'])
        matched_pair_index_df = matched_pair_index_df.drop(columns=['flag'])
        matched_itinerary_df = pd.DataFrame(columns=['itinerary_node_list', 'itinerary_segment_dis_list', 'pickup_distance'])
        if len(matched_itinerary) > 0:
            matched_itinerary_df['itinerary_node_list'] = matched_itinerary[0]
            matched_itinerary_df['itinerary_segment_dis_list'] = matched_itinerary[1]
            matched_itinerary_df['pickup_distance'] = matched_itinerary[2]

        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']

        # 'maximum_pickup_time_passenger_can_tolerate', 'maximum_price_passenger_can_tolerate'
        matched_itinerary_df['pickup_time'] = matched_itinerary_df['pickup_distance'].values / env_params['vehicle_speed'] * 3600
        matched_itinerary_df['delivery_time'] = matched_itinerary_df['pickup_distance'].values * env_params['price_per_km']

        # when the order is matched
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)

        if df_matched.shape[0] > 0:
            idle_driver_table = self.driver_table[(self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)]
            order_array = df_matched['order_id'].values
            cor_order = []
            cor_driver = []
            for i in range(len(matched_pair_index_df)):
                cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
                cor_driver.append(idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])
            cor_driver = np.array(cor_driver)
            df_matched = df_matched.iloc[cor_order, :]

            # driver decide whether cancelled
            # 现在暂时不让其取消。需考虑时可用self.driver_cancel_prob_array来计算
            driver_cancel_prob = np.zeros(len(matched_pair_index_df))
            prob_array = np.random.rand(len(driver_cancel_prob))
            con_driver_remain = prob_array >= driver_cancel_prob

            # passenger decide whether cancelled
            # 现在暂时不让其取消。需考虑时可用各订单的'cancel_prob'属性来计算
            con_passenge_keep_wait = df_matched['maximum_pickup_time_passenger_can_tolerate'].values > \
                                     matched_itinerary_df[
                                         'pickup_time'].values
            con_passenge_accept_price = df_matched[
                                            'maximum_price_passenger_can_tolerate'].values >= matched_itinerary_df[
                                            'delivery_time'].values
            passenger_cancel_prob = np.zeros(len(matched_pair_index_df))
            # prob_array = np.random.rand(len(passenger_cancel_prob))
            # con_passenger_remain = con_passenge_keep_wait & con_passenge_accept_price
            con_remain = con_driver_remain #& con_passenger_remain

            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            # 若匹配上后又被取消，目前假定司机按原计划继续cruising or repositioning
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_itinerary_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + new_matched_requests['trip_time'].values
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values

            # driver not cancelled
            self.driver_table.loc[cor_driver[con_remain], 'status'] = 2
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
            self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests['dest_grid_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['pickup_time'].values
            self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
            self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'time_to_last_cruising'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'current_road_node_index'] = 0
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests['itinerary_node_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_segment_dis_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_segment_dis_list'] + new_matched_requests['itinerary_segment_dis_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time_for_current_node'] = \
                matched_itinerary_df[con_remain]['itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed
            # # update matched tracks for one time
            if self.track_recording_flag == True:
                for i, index in enumerate(cor_driver[con_remain]):
                    try:
                        driver_id = self.driver_table.loc[index, 'driver_id']
                        node_id_list = self.driver_table.loc[index, 'itinerary_node_list']
                        lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(node_id_list)
                        time_array = np.cumsum(self.driver_table.loc[index, 'itinerary_segment_dis_list']) / self.vehicle_speed
                        time_array = np.concatenate([np.array([self.time]), self.time + time_array[:-1]])
                        delivery_time = len(new_matched_requests['itinerary_node_list'][i])
                        pickup_time = len(time_array) - delivery_time
                        task_type_array = np.concatenate([2 + np.zeros(pickup_time), 1 + np.zeros(delivery_time)])
                        self.new_tracks[driver_id] = np.vstack([lat_array, lng_array,np.array(node_id_list),grid_id_array,task_type_array, time_array]).T.tolist()
                    except:
                        print("time " + str(i) + " loss")
        # when the order is not matched
        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]],axis=0)

        return new_matched_requests, update_wait_requests


    def order_generation(self):
       # 原始订单信息假定构成如下：'order_id'，'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat',
       #                      'immediate_reward', 'trip_distance','trip_time', 'designed_reward',
       #                      'dest_grid_id'，'cancel_prob'， 'itinerary_node_list', 'itinerary_segment_dis_list'

       #generate new orders
       if self.order_generation_mode == 'sample_from_base':
           # directly sample orders from the historical order database
           sampled_requests = []
           count_interval = int(math.floor(self.time / self.request_interval))
        #    print("test",str(count_interval * self.request_interval))
           if count_interval * self.request_interval not in self.request_databases.keys():
               return
           self.request_database = self.request_databases[count_interval * self.request_interval]
           database_size = len(self.request_database)


           # sample a portion of historical orders
           num_request = int(np.rint(self.order_sample_ratio * database_size))
           if num_request <= database_size:
               sampled_request_index = np.random.choice(database_size, num_request).tolist()
               sampled_requests = [self.request_database[index] for index in sampled_request_index]

           


           #generate complete information for new orders
           np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
            
           column_name = ['order_id', 'origin_id','origin_lat', 'origin_lng','dest_id',
       'dest_lat', 'dest_lng', 'trip_distance', 'start_time',
       'origin_grid_id', 'dest_grid_id', 'itinerary_node_list','itinerary_segment_dis_list',
        'maximum_pickup_time_passenger_can_tolerate','maximum_price_passenger_can_tolerate','trip_time','designed_reward', 'cancel_prob']
                         

        #                         'wait_time','status','maximum_wait_time','pickup_distance','pickup_time', 'driver_id',  't_matched'
           if len(sampled_requests) > 0:
               len1 = []
               len2 = []
               itinerary_segment_dis_list = []
               itinerary_node_list = np.array(sampled_requests)[:, 11]
               for itinerary_node in itinerary_node_list:
                   if itinerary_node is not None:
                       itinerary_segment_dis = []
                       for i in range(len(itinerary_node) - 1):
                           # dis = nx.shortest_path_length(G, node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]], weight='length')
                           dis = distance(node_id_to_lat_lng[itinerary_node[i]],
                                          node_id_to_lat_lng[itinerary_node[i + 1]])
                           itinerary_segment_dis.append(dis)
                       len1.append(len(itinerary_node))
                       len2.append(len(itinerary_segment_dis))
                       itinerary_segment_dis_list.append(itinerary_segment_dis)
               for i in range(len(itinerary_node_list)):
                   if len(itinerary_node_list[i]) == len(itinerary_segment_dis_list[i]):
                       continue
                   # temp = itinerary_node_list[i][:-1]
                   # itinerary_node_list[i] = temp
                   itinerary_node_list[i].pop()


               wait_info = pd.DataFrame(sampled_requests, columns=column_name)
               wait_info['itinerary_node_list'] = itinerary_node_list
               wait_info['start_time'] = self.time
               wait_info['wait_time'] = 0
               wait_info['status'] = 0
               wait_info['maximum_wait_time'] = self.maximum_wait_time_mean
               wait_info['itinerary_segment_dis_list'] = itinerary_segment_dis_list
               wait_info['weight'] = wait_info['trip_distance'] * 3.2
               wait_info = wait_info.drop(columns=['trip_distance'])
               wait_info = wait_info.drop(columns=['designed_reward'])

               self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

       return

    def cruise_and_reposition(self):
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']

        # reposition decision
        # total_idle_time 为reposition间的间隔， time to last cruising 为cruising间的间隔。
        if self.reposition_flag == True:
            # print("total_idle_time",self.driver_table['total_idle_time'])
            con_eligibe = (self.driver_table['total_idle_time'] > self.eligible_time_for_reposition) & \
                          (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            # print("eligible_driver_table",eligible_driver_table)
            eligible_driver_index = np.array(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    reposition(eligible_driver_table, self.reposition_mode)
                self.driver_table.loc[eligible_driver_index, 'status'] = 4
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed
                self.driver_table.loc[eligible_driver_index, 'total_idle_time'] = 0
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(lambda x: x[-1]).values
                lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(target_node_array)
                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = grid_id_array


        if self.cruise_flag == True:
            
            con_eligibe = (self.driver_table['time_to_last_cruising'] > self.max_idle_time) & \
                          (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = list(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    cruising(eligible_driver_table,self.cruise_mode)
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                    lambda x: x[-1]).values
                lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(target_node_array)
                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = grid_id_array


    def real_time_track_recording(self):
        con_real_time = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 3) | \
                        (self.driver_table['status'] == 4)
        real_time_driver_table = self.driver_table.loc[con_real_time, ['driver_id', 'lng', 'lat', 'status']]
        real_time_driver_table['time'] = self.time
        lat_array = real_time_driver_table['lat'].values.tolist()
        lng_array = real_time_driver_table['lng'].values.tolist()
        node_list = []
        grid_list = []
        for i in range(len(lng_array)):
                        id = node_coord_to_id[(lat_array[i],lng_array[i])]
                        node_list.append(id)
                        grid_list.append(result[result['node_id'] == id ]['grid_id'].tolist()[0])
        real_time_driver_table['node_id'] = node_list
        real_time_driver_table['grid_id'] = grid_list
        real_time_driver_table = real_time_driver_table[['driver_id','lat','lng','node_id','grid_id','status','time']]
        # print("columns",real_time_driver_table)
        real_time_tracks = real_time_driver_table.set_index('driver_id').T.to_dict('list')
        self.new_tracks = {**self.new_tracks, **real_time_tracks}


    def update_state(self):
        # update next state
        # 车辆状态：0 cruise (park 或正在cruise)， 1 表示delivery，2 pickup, 3 表示下线, 4 reposition
        # 先更新未完成任务的，再更新已完成任务的
        self.driver_table['current_road_node_index'] = self.driver_table['current_road_node_index'].values.astype(int)

        loc_cruise = self.driver_table['status'] == 0
        loc_parking = loc_cruise & (self.driver_table['remaining_time'] == 0)
        loc_actually_cruising = loc_cruise & (self.driver_table['remaining_time'] > 0)
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].values - self.delta_t
        loc_finished = self.driver_table['remaining_time'] <= 0
        loc_unfinished = ~loc_finished
        loc_delivery = self.driver_table['status'] == 1
        loc_pickup = self.driver_table['status'] == 2
        loc_reposition = self.driver_table['status'] == 4
        loc_road_node_transfer = self.driver_table['remaining_time_for_current_node'].values - self.delta_t <= 0

        # for unfinished tasks
        self.driver_table.loc[loc_cruise, 'total_idle_time'] += self.delta_t #这里针对所有cruising车辆，不论其任务是否完成
          #下面这部分目前用了for 循环，之后可考虑如何进一步简化
        con_real_time_ongoing = loc_unfinished & (loc_cruise | loc_pickup | loc_reposition)
        self.driver_table.loc[~loc_road_node_transfer & con_real_time_ongoing, 'remaining_time_for_current_node'] -= self.delta_t

        road_node_transfer_list = list(self.driver_table[loc_road_node_transfer & con_real_time_ongoing].index)
        current_road_node_index_array = self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'].values
        current_remaining_time_for_node_array = self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'].values
        transfer_itinerary_node_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_node_list'].values
        transfer_itinerary_segment_dis_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_segment_dis_list'].values
        new_road_node_index_array = np.zeros(len(road_node_transfer_list))
        new_road_node_array = np.zeros(new_road_node_index_array.shape[0])
        new_remaining_time_for_node_array = np.zeros(new_road_node_index_array.shape[0])

        for i in range(len(road_node_transfer_list)):
            current_node_index = current_road_node_index_array[i]
            itinerary_segment_time = np.array(transfer_itinerary_segment_dis_list[i][current_node_index:]) / self.vehicle_speed
            itinerary_segment_time[0] = current_remaining_time_for_node_array[i]
            itinerary_segment_cumsum_time = itinerary_segment_time.cumsum()
            new_road_node_index = (itinerary_segment_cumsum_time > self.delta_t).argmax()
            new_remaining_time = itinerary_segment_cumsum_time[new_road_node_index] - self.delta_t
            new_road_node_index = new_road_node_index - 1 + current_node_index
            new_road_node = transfer_itinerary_node_list[i][new_road_node_index]

            new_road_node_index_array[i] = new_road_node_index
            new_road_node_array[i] = new_road_node
            new_remaining_time_for_node_array[i] = new_remaining_time

        self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'] = new_road_node_index_array.astype(int)
        self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'] = new_remaining_time_for_node_array
        lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(new_road_node_array)
        self.driver_table.loc[road_node_transfer_list, 'lng'] = lng_array
        self.driver_table.loc[road_node_transfer_list, 'lat'] = lat_array
        self.driver_table.loc[road_node_transfer_list, 'grid_id'] = grid_id_array

        # for finished tasks

        # for all the finished tasks
        self.driver_table.loc[loc_finished, 'remaining_time'] = 0
        con_not_pickup = loc_finished & (loc_actually_cruising | loc_delivery | loc_reposition)
        con_not_pickup_actually_cruising = loc_finished & (loc_delivery | loc_reposition)
        self.driver_table.loc[con_not_pickup, 'lng'] = self.driver_table.loc[con_not_pickup, 'target_loc_lng'].values
        self.driver_table.loc[con_not_pickup, 'lat'] = self.driver_table.loc[con_not_pickup, 'target_loc_lat'].values
        self.driver_table.loc[con_not_pickup, 'grid_id'] = self.driver_table.loc[con_not_pickup, 'target_grid_id'].values
        self.driver_table.loc[con_not_pickup, ['status', 'time_to_last_cruising',
                                           'current_road_node_index', 'remaining_time_for_current_node']] = 0
        self.driver_table.loc[con_not_pickup_actually_cruising, 'total_idle_time'] = 0
        shape = self.driver_table[con_not_pickup].shape[0]
        empty_list = [[] for i in range(shape)]
        self.driver_table.loc[con_not_pickup, 'itinerary_node_list'] =  np.array(empty_list + [[-1]], dtype=object)[:-1]
        self.driver_table.loc[con_not_pickup, 'itinerary_segment_dis_list'] =  np.array(empty_list + [[-1]], dtype=object)[:-1]

        # for parking finished
        self.driver_table.loc[loc_parking, 'time_to_last_cruising'] += self.delta_t

        # for actually cruising finished

        # for actually cruising finished

        # for delivery finished
        self.driver_table.loc[loc_finished & loc_delivery, 'matched_order_id'] = 'None'

        #for pickup    delivery是载客  pickup是接客
        #分两种情况，一种是下一时刻pickup 和 delivery都完成，另一种是下一时刻pickup 完成，delivery没完成
        #当前版本delivery直接跳转，因此不需要做更新其中间路线的处理。车辆在pickup完成后，delivery完成前都实际处在pickup location。完成任务后直接跳转到destination
        #如果需要考虑delivery的中间路线，可以把pickup和delivery状态进行融合
        finished_pickup_driver_index_array = np.array(self.driver_table[loc_finished & loc_pickup].index)
        current_road_node_index_array = self.driver_table.loc[finished_pickup_driver_index_array, 'current_road_node_index'].values
        itinerary_segment_dis_list = self.driver_table.loc[finished_pickup_driver_index_array, 'itinerary_segment_dis_list'].values
        remaining_time_array = np.zeros(len(finished_pickup_driver_index_array))
        for i in range(remaining_time_array.shape[0]):
            current_node_index = current_road_node_index_array[i]
            remaining_time_array[i] = np.sum(itinerary_segment_dis_list[i][current_node_index:]) / self.vehicle_speed - self.delta_t
        delivery_not_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array > 0]
        delivery_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array <= 0]
        self.driver_table.loc[delivery_not_finished_driver_index, 'status'] = 1
        self.driver_table.loc[delivery_not_finished_driver_index, 'remaining_time'] = remaining_time_array[remaining_time_array > 0]
        if len(delivery_finished_driver_index > 0):
            self.driver_table.loc[delivery_finished_driver_index, 'lng'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_loc_lng'].values
            self.driver_table.loc[delivery_finished_driver_index, 'lat'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_loc_lat'].values
            self.driver_table.loc[delivery_finished_driver_index, 'grid_id'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_grid_id'].values
            self.driver_table.loc[delivery_finished_driver_index, ['status', 'time_to_last_cruising',
                                               'current_road_node_index', 'remaining_time_for_current_node']] = 0
            self.driver_table.loc[delivery_finished_driver_index, 'total_idle_time'] = 0
            # shape = self.driver_table[delivery_finished_driver_index].shape[0]
            shape = self.driver_table.loc[delivery_finished_driver_index].values.shape[0]
            empty_list = [[] for i in range(shape)]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'matched_order_id'] = 'None'
        self.wait_requests['wait_time'] += self.delta_t
        return

    def driver_online_offline_update(self):
        # update driver online/offline status
        # currently, only offline con need to be considered.
        # offline driver will be deleted from the table
        next_time = self.time + self.delta_t
        self.driver_table = driver_online_offline_decision(self.driver_table, next_time)
        return

    def update_time(self):
        # time counter
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        return


    def step(self):
        self.new_tracks = {}

        # Step 1: order dispatching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table, self.maximal_pickup_distance, self.dispatch_method)
        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(matched_pair_actual_indexes, matched_itinerary)
        self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
        self.matched_requests = self.matched_requests.reset_index(drop=True)
        self.wait_requests = df_update_wait_requests.reset_index(drop=True)

        # Step 3: bootstrap new orders
        self.order_generation()

        # Step 4: cruising and/or repositioning decision
        self.cruise_and_reposition()

        # Step 4.1: track recording
        if self.track_recording_flag == True:
            self.real_time_track_recording()


        # Step 5: update next state for drivers
        self.update_state()

        # Step 6： online/offline update()
        self.driver_online_offline_update()

        # Step 7: update time
        self.update_time()

        return self.new_tracks
