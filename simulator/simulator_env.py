from simulator_pattern import *
from utilities import *
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import sys

class Simulator:
    def __init__(self, **kwargs):

        # basic parameters: time & sample
        self.t_initial = kwargs['t_initial']
        self.t_end = kwargs['t_end']
        self.delta_t = kwargs['delta_t']
        self.vehicle_speed = kwargs['vehicle_speed']
        self.repo_speed = kwargs.pop('repo_speed', 3)
        self.time = None
        self.current_step = None

        # order generation
        self.order_sample_ratio = kwargs['order_sample_ratio']
        self.order_generation_mode = kwargs['order_generation_mode']
        self.request_interval = kwargs['request_interval']

        # wait cancel
        self.maximum_wait_time_mean = kwargs.pop('maximum_wait_time_mean', 120)
        self.maximum_wait_time_std = kwargs.pop('maximum_wait_time_std', 0)

        # driver cancel after matching based on maximal pickup distance
        self.maximal_pickup_distance = kwargs['maximal_pickup_distance']

        # track recording
        self.track_recording_flag = kwargs['track_recording_flag']
        self.new_tracks = {}
        self.match_and_cancel_track = {}
        # pattern
        self.simulator_mode = kwargs.pop('simulator_mode', 'simulator_mode')
        self.experiment_mode = kwargs.pop('experiment_mode', 'test')
        self.request_file_name = kwargs['request_file_name']
        self.driver_file_name = kwargs['driver_file_name']
        pattern_params = {'simulator_mode': self.simulator_mode, 'request_file_name': self.request_file_name,
                          'driver_file_name': self.driver_file_name}
        pattern = SimulatorPattern(**pattern_params)

        # road network
        road_network_file_name = kwargs['road_network_file_name']
        '''
        plan to delete
        '''
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
        self.request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                                'trip_distance', 'start_time', 'origin_grid_id','dest_grid_id', 'itinerary_node_list',
                                'itinerary_segment_dis_list', 'trip_time', 'cancel_prob', 't_matched',
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
        self.request_databases = None
        self.request_database = None

    def initial_base_tables(self):
        """
        This function used to initial the driver table and order table
        :return: None
        """
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
        """
        This function used to update driver table and wait requests after matching
        :param matched_pair_actual_indexes: matched pair including driver id and order id
        :param matched_itinerary: including driver pick up route info
        :return: matched requests and wait requests
        """
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)
        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes, columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])
        # matched_pair_index_df = matched_pair_index_df.drop(columns=['flag'])
        matched_itinerary_df = pd.DataFrame(columns=['itinerary_node_list', 'itinerary_segment_dis_list', 'pickup_distance'])
        if len(matched_itinerary) > 0:
            matched_itinerary_df['itinerary_node_list'] = matched_itinerary[0]
            matched_itinerary_df['itinerary_segment_dis_list'] = matched_itinerary[1]
            matched_itinerary_df['pickup_distance'] = matched_itinerary[2]

        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']

        # price and pickup time info which used to judge whether cancel the order-driver pair
        matched_itinerary_df['pickup_time'] = matched_itinerary_df['pickup_distance'].values / self.vehicle_speed * 3600

        # extract the order is matched
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

            # price and pickup time moudle which used to judge whether cancel the order-driver pair
            # matched_itinerary_df['pickup_time'].values
            con_passenge_keep_wait = df_matched['maximum_pickup_time_passenger_can_tolerate'].values > \
                                                        matched_itinerary_df['pickup_time'].values


            con_passenger_remain = con_passenge_keep_wait
            con_remain = con_driver_remain & con_passenger_remain
            # order after cancelled
            update_wait_requests = df_matched[~con_remain]

            # driver after cancelled
            # 若匹配上后又被取消，目前假定司机按原计划继续cruising or repositioning
            self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0

            # order not cancelled
            new_matched_requests = df_matched[con_remain]
            new_matched_requests['t_matched'] = self.time
            new_matched_requests['pickup_distance'] = matched_itinerary_df[con_remain]['pickup_distance'].values
            new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed * 3600
            new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + new_matched_requests['trip_time'].values
            new_matched_requests['status'] = 1
            new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values
            itinerary_node_list = new_matched_requests['itinerary_node_list']

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
            try:
                self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests['itinerary_node_list']).values
            except:
                print(self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'])
                print(matched_itinerary_df[con_remain]['itinerary_node_list'])
                print(new_matched_requests['itinerary_node_list'])
            self.driver_table.loc[cor_driver[con_remain], 'itinerary_segment_dis_list'] = \
                (matched_itinerary_df[con_remain]['itinerary_segment_dis_list'] + new_matched_requests['itinerary_segment_dis_list']).values
            self.driver_table.loc[cor_driver[con_remain], 'remaining_time_for_current_node'] = \
                matched_itinerary_df[con_remain]['itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600
            # update matched tracks for one time
            if self.track_recording_flag:
                for j, index in enumerate(cor_driver[con_remain]):
                    driver_id = self.driver_table.loc[index, 'driver_id']
                    node_id_list = self.driver_table.loc[index, 'itinerary_node_list']
                    lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(node_id_list)
                    time_array = np.cumsum(self.driver_table.loc[index, 'itinerary_segment_dis_list']) / self.vehicle_speed * 3600
                    time_array = np.concatenate([np.array([self.time]), self.time + time_array])
                    delivery_time = len(new_matched_requests['itinerary_node_list'].values.tolist()[j])
                    pickup_time = len(time_array) - delivery_time
                    task_type_array = np.concatenate([2 + np.zeros(pickup_time), 1 + np.zeros(delivery_time)])
                    order_id = self.driver_table.loc[index, 'matched_order_id']

                    self.new_tracks[driver_id] = np.vstack(
                        [lat_array, lng_array, np.array([order_id] * len(lat_array)), np.array(node_id_list), grid_id_array, task_type_array,
                         time_array]).T.tolist()

                self.match_and_cancel_track[self.time] = [len(df_matched),len(new_matched_requests)]
                    


        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]],axis=0)

        return new_matched_requests, update_wait_requests

    def order_generation(self):
        """
        This function used to generate initial order by different time
        :return:
        """
        if self.order_generation_mode == 'sample_from_base':
            # directly sample orders from the historical order database
            sampled_requests = []
            temp_request = []
            min_time = max(env_params['t_initial'], self.time - self.request_interval)
            for time in range(min_time, self.time):
                if time in self.request_databases.keys():
                    temp_request.extend(self.request_databases[time])
            # if self.time in self.request_databases.keys():
            #     temp_request = self.request_databases[self.time]
            if temp_request == []:
                return
            database_size = len(temp_request)
            # sample a portion of historical orders
            num_request = int(np.rint(self.order_sample_ratio * database_size))
            if num_request <= database_size:
                sampled_request_index = np.random.choice(database_size, num_request, replace=False).tolist()
                sampled_requests = [temp_request[index] for index in sampled_request_index]

            # generate complete information for new orders

            column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                           'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                           'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']

            if len(sampled_requests) > 0:
                itinerary_segment_dis_list = []
                itinerary_node_list = np.array(sampled_requests)[:, 11]
                trip_distance = []
                # trip_distance = npSS.array(sampled_requests)[:, 7]
                for k, itinerary_node in enumerate(itinerary_node_list):
                    try:
                        itinerary_segment_dis = []
                        # route generation
                        if env_params['delivery_mode'] == 'rg':
                            for i in range(len(itinerary_node) - 1):
                                dis = distance(node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]])
                                itinerary_segment_dis.append(dis)
                        # start - end manhadun distance
                        elif env_params['delivery_mode'] == 'ma':
                            dis = distance(node_id_to_lat_lng[itinerary_node[0]], node_id_to_lat_lng[itinerary_node[-1]])
                            itinerary_node_list[k] = [itinerary_node[0],itinerary_node[-1]]
                            itinerary_segment_dis.append(dis)
                        itinerary_segment_dis_list.append(itinerary_segment_dis)
                        trip_distance.append(sum(itinerary_segment_dis))
                    except Exception as e:
                        print(e)
                        print(itinerary_node)
                # for j in range(len(itinerary_node_list)):
                #     if len(itinerary_node_list[j]) == len(itinerary_segment_dis_list[j]):
                #         continue
                #     itinerary_node_list[j].pop()

                wait_info = pd.DataFrame(sampled_requests, columns=column_name)
                wait_info['itinerary_node_list'] = itinerary_node_list
                wait_info['start_time'] = self.time
                wait_info['trip_distance'] = trip_distance
                wait_info['wait_time'] = 0
                wait_info['status'] = 0
                # wait_info['maximum_wait_time'] = np.random.normal(self.maximum_wait_time_mean,
                #                                                   self.maximum_wait_time_std, len(wait_info))
                wait_info['maximum_wait_time'] = self.maximum_wait_time_mean
                wait_info['itinerary_segment_dis_list'] = itinerary_segment_dis_list
                wait_info['weight'] = wait_info['trip_distance'] * 5
                # add extra info of orders
                # 添加分布  价格高的删除
                wait_info['maximum_price_passenger_can_tolerate'] = np.random.normal(
                    env_params['maximum_price_passenger_can_tolerate_mean'],
                    env_params['maximum_price_passenger_can_tolerate_std'],
                    len(wait_info))
                wait_info = wait_info[
                    wait_info['maximum_price_passenger_can_tolerate'] >= wait_info['trip_distance'] * env_params[
                        'price_per_km']]
                wait_info['maximum_pickup_time_passenger_can_tolerate'] = np.random.normal(
                    env_params['maximum_pickup_time_passenger_can_tolerate_mean'],
                    env_params['maximum_pickup_time_passenger_can_tolerate_std'],
                    len(wait_info))

                # wait_info = wait_info.drop(columns=['trip_distance'])
                # wait_info = wait_info.drop(columns=['designed_reward'])
                self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

        return

    def cruise_and_reposition(self):
        """
        This function used to judge the drivers' status and update drivers' table
        :return: None
        """
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']

        # reposition decision
        # total_idle_time 为reposition间的间隔， time to last both-rg-cruising 为cruising间的间隔。
        if self.reposition_flag:
            con_eligibe = (self.driver_table['total_idle_time'] > self.eligible_time_for_reposition) & \
                          (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = np.array(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    reposition(eligible_driver_table, self.reposition_mode)
                self.driver_table.loc[eligible_driver_index, 'status'] = 4
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
                self.driver_table.loc[eligible_driver_index, 'total_idle_time'] = 0
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(lambda x: x[-1]).values
                lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(target_node_array)
                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = grid_id_array
        if self.cruise_flag:
            con_eligibe = (self.driver_table['time_to_last_cruising'] > self.max_idle_time) & \
                          (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = list(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    cruising(eligible_driver_table,self.cruise_mode)
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                    lambda x: x[-1]).values
                lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(target_node_array)
                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = grid_id_array

    def real_time_track_recording(self):
        """
        This function used to record the drivers' info which doesn't delivery passengers
        :return: None
        """
        con_real_time = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 3) | \
                        (self.driver_table['status'] == 4)
        real_time_driver_table = self.driver_table.loc[con_real_time, ['driver_id', 'lng', 'lat', 'status']]
        real_time_driver_table['time'] = self.time
        lat_array = real_time_driver_table['lat'].values.tolist()
        lng_array = real_time_driver_table['lng'].values.tolist()
        node_list = []
        grid_list = []
        for i in range(len(lng_array)):
            id = node_coord_to_id[(lat_array[i], lng_array[i])]
            node_list.append(id)
            grid_list.append(result[result['node_id'] == id ]['grid_id'].tolist()[0])
        real_time_driver_table['node_id'] = node_list
        real_time_driver_table['grid_id'] = grid_list
        real_time_driver_table = real_time_driver_table[['driver_id','lat','lng','node_id','grid_id','status','time']]
        real_time_tracks = real_time_driver_table.set_index('driver_id').T.to_dict('list')
        self.new_tracks = {**self.new_tracks, **real_time_tracks}

    def update_state(self):
        """
        This function used to update the drivers' status and info
        :return: None
        """
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
        self.driver_table.loc[loc_cruise, 'total_idle_time'] += self.delta_t
        con_real_time_ongoing = loc_unfinished & (loc_cruise | loc_reposition) | loc_pickup
        self.driver_table.loc[~loc_road_node_transfer & con_real_time_ongoing, 'remaining_time_for_current_node'] -= self.delta_t

        road_node_transfer_list = list(self.driver_table[loc_road_node_transfer & con_real_time_ongoing].index)
        current_road_node_index_array = self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'].values
        current_remaining_time_for_node_array = self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'].values
        transfer_itinerary_node_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_node_list'].values
        transfer_itinerary_segment_dis_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_segment_dis_list'].values
        new_road_node_index_array = np.zeros(len(road_node_transfer_list))
        new_road_node_array = np.zeros(new_road_node_index_array.shape[0])
        new_remaining_time_for_node_array = np.zeros(new_road_node_index_array.shape[0])

        # update the driver itinerary list
        for i in range(len(road_node_transfer_list)):
            current_node_index = current_road_node_index_array[i]
            itinerary_segment_time = np.array(transfer_itinerary_segment_dis_list[i][current_node_index:]) / self.vehicle_speed * 3600
            itinerary_segment_time[0] = current_remaining_time_for_node_array[i]
            itinerary_segment_cumsum_time = itinerary_segment_time.cumsum()
            new_road_node_index = (itinerary_segment_cumsum_time > self.delta_t).argmax()
            new_remaining_time = itinerary_segment_cumsum_time[new_road_node_index] - self.delta_t
            if itinerary_segment_cumsum_time[-1] <= self.delta_t:
                new_road_node_index = len(transfer_itinerary_segment_dis_list[i])-1
            else:
                new_road_node_index = new_road_node_index + current_node_index
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

        # for all the finished tasks
        self.driver_table.loc[loc_finished & (~ loc_pickup), 'remaining_time'] = 0
        con_not_pickup = loc_finished & (loc_actually_cruising | loc_delivery | loc_reposition)
        con_not_pickup_actually_cruising = loc_finished & (loc_delivery | loc_reposition)
        self.driver_table.loc[con_not_pickup, 'lng'] = self.driver_table.loc[con_not_pickup, 'target_loc_lng'].values
        self.driver_table.loc[con_not_pickup, 'lat'] = self.driver_table.loc[con_not_pickup, 'target_loc_lat'].values
        self.driver_table.loc[con_not_pickup, 'grid_id'] = self.driver_table.loc[con_not_pickup, 'target_grid_id'].values
        self.driver_table.loc[con_not_pickup, ['status', 'time_to_last_cruising', 'current_road_node_index',
                                               'remaining_time_for_current_node']] = 0
        self.driver_table.loc[con_not_pickup_actually_cruising, 'total_idle_time'] = 0
        shape = self.driver_table[con_not_pickup].shape[0]
        empty_list = [[] for _ in range(shape)]
        self.driver_table.loc[con_not_pickup, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
        self.driver_table.loc[con_not_pickup, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]

        # for parking finished
        self.driver_table.loc[loc_parking, 'time_to_last_cruising'] += self.delta_t

        # for delivery finished
        self.driver_table.loc[loc_finished & loc_delivery, 'matched_order_id'] = 'None'
        """
        for pickup    delivery是载客  pickup是接客
        分两种情况，一种是下一时刻pickup 和 delivery都完成，另一种是下一时刻pickup 完成，delivery没完成
        当前版本delivery直接跳转，因此不需要做更新其中间路线的处理。车辆在pickup完成后，delivery完成前都实际处在pickup location。完成任务后直接跳转到destination
        如果需要考虑delivery的中间路线，可以把pickup和delivery状态进行融合
        """
        finished_pickup_driver_index_array = np.array(self.driver_table[loc_finished & loc_pickup].index)
        current_road_node_index_array = self.driver_table.loc[finished_pickup_driver_index_array,
                                                              'current_road_node_index'].values
        itinerary_segment_dis_list = self.driver_table.loc[finished_pickup_driver_index_array,
                                                           'itinerary_segment_dis_list'].values
        remaining_time_current_node_temp = self.driver_table.loc[finished_pickup_driver_index_array,
                                                           'remaining_time_for_current_node'].values
        remaining_time_array = np.zeros(len(finished_pickup_driver_index_array))
        for i in range(remaining_time_array.shape[0]):
            current_node_index = current_road_node_index_array[i]
            remaining_time_array[i] = np.sum(itinerary_segment_dis_list[i][current_node_index+1:]) / self.vehicle_speed * 3600 + remaining_time_current_node_temp[i]
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
                                                                   'current_road_node_index',
                                                                   'remaining_time_for_current_node']] = 0
            self.driver_table.loc[delivery_finished_driver_index, 'total_idle_time'] = 0
            shape = self.driver_table.loc[delivery_finished_driver_index].values.shape[0]
            empty_list = [[] for _ in range(shape)]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'matched_order_id'] = 'None'
        self.wait_requests['wait_time'] += self.delta_t
        return

    def driver_online_offline_update(self):
        """
        update driver online/offline status
        currently, only offline con need to be considered.
        offline driver will be deleted from the table
        :return: None
        """
        next_time = self.time + self.delta_t
        self.driver_table = driver_online_offline_decision(self.driver_table, next_time)
        return

    def update_time(self):
        """
        This function used to count time
        :return:
        """
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        return

    def step(self):
        """
        This function used to run the simulator step by step
        :return:
        """
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

        # Step 4: both-rg-cruising and/or repositioning decision
        self.cruise_and_reposition()

        # Step 4.1: track recording
        if self.track_recording_flag:
            self.real_time_track_recording()

        # Step 5: update next state for drivers
        self.update_state()

        # Step 6： online/offline update()
        self.driver_online_offline_update()

        # Step 7: update time
        self.update_time()

        return self.new_tracks
