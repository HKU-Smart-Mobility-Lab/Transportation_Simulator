# main.py
This file executes the entire progress of the simulator and saves the result in pickle files.  
[detail](#1.0)

<h2 id="0"> </h2>

# simulator_eny.py
This file defines the class simulator which includes most of the necessary parameters and hole steps of the simulator from order dispatch to updating drivers' state and time.   
[detail](#2.0)

# utilities.py
This file first loads the information of the graph network from graphml file and then builds the connection to mongodb, which will be used to speed up access to road network information. There are also some core and frequently used functions defined here.   
[detail](#3.0)

# dispatch_alg.py
This file implements the function LD, which utilizes a binary map matching algorithm to dispatch orders.

[detail](#4.0)

# config.py
This file includes all adjustable parameters to make the model more realistic.  
[detail](#5.0)

***

> ## main.py

<h2 id="1.0">

output:    

|                name                    |   type  |  describe      |
|                 :----:                 |  :----: | :----:         |
|records_driver_num_(single_driver_num)  | pickle  | track record   | 
|passenger_records_driver_num_(single_driver_num)  | pickle  | record requests| 
|match_and_cacel_(single_driver_num)     | pickle  | track record   | 
</h2>

[back](#0) 

***

> ## simulator_eny.py

<h2 id="2.0">  

[initial_base_tabels()](#1.1) : Initial the driver table and order table.

[update_info_after_matching_multi_process(matched_pair_actual_indexes, matched_itinerary)](#1.2) : Update driver table and wait requests after matching.

[order_generation()](#1.3) : Generate initial order by different time.

[cruise_and_reposition()](#1.4) : Judge the drivers' status and update drivers' table.

[real_time_track_recording()](#1.5) : Record the drivers' info which doesn't delivery passengers.

[update_state()](#1.6) : Update the drivers' status and info.  
  
[driver_online_offline_update()](#1.7) : Update driver online/offline status
currently, only offline con need to be considered. Offline drivers will be deleted from the table.  

[update_time()](#1.8) : Count time consumed by program.  

[step()](#1.9) : Run the simulator step by step.  
Step 1: order dispatching;  
Step 2: driver/passenger reaction after dispatching;  
Step 3: bootstrap new orders;  
Step 4: both cruising and/or repositioning decisions;  
Step 5: track recording;  
Step 5: update the next state for drivers;  
Step 6: online/offline update();  
Step 7: update time.

---

### ```initial_base_tables()```  
<h2 id="1.1">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```update_info_after_matching_multi_process()```
<h2 id="1.2">

input:

|     name     |type  | describe|
|:----:    |  :----: | :----:|
|matched_pair_actual_indexes|pandas.Dataframe| matched pair including driver id and order id|  
|matched_itinerary|pandas.Dataframe|including driver pick up route info|

return:
|     name     |type  | describe|
|:----:    |  :----: | :----:|
|new_matched_requests|pandas.Dataframe| matched requests|  
|update_wait_requests|pandas.Dataframe| wait requests|
</h2>

### ```order_generation()```  
<h2 id="1.3">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```cruise_and_reposition()```  
<h2 id="1.4">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>  

### ```real_time_track_recording()```  
<h2 id="1.5">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```update_state()```  
<h2 id="1.6">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```driver_online_offline_update()```  
<h2 id="1.7">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```update_time()```  
<h2 id="1.8">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```step()```  
<h2 id="1.9">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|none      | -       | -     | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|self.new_tracks| -       | -     |  
</h2>
</h2>

[back](#0)

***

> ## utilities.py

<h2 id="3.0">

[get_zone(lat, lng)](#3.1) : Get zone_id of segment node.  
[distance(coord_1, coord_2)](#3.2) : Calculate manhattan distance between these two points.  
[get_distance_array(origin_coord_array, dest_coord_array)](#3.3) : Calculate manhattan distance between two lists.  
[route_generation_array(origin_coord_array, dest_coord_array, mode='rg')](#3.4) : Generate the route between array1 and array2, include the id of nodes, the distance between nodes and distance from origin node to destination node.  
[load_data(data_path, file_name)](#3.5) : Get data from ```graph.graphml```.  
[get_information_for_nodes(node_id_array)](#3.6) : Get the array of longitude, latitute and node id.  
[get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10)](#3.7) : Get the array of epsilon.  
[sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1, driver_number_dist='')](#3.8) : Get information of sampled drivers.  
[sample_request_num(t_mean, std, delta_t)](#3.9) : Get sample request num during delta t.  
[reposition(eligible_driver_table, mode)](#3.10) : Simulate repositioning drivers.  
[cruising(eligible_driver_table, mode)](#3.11) : Simulate cruising drivers.  
[order_dispatch(wait_requests, driver_table, maximal_pickup_distance=950, dispatch_method='LD')](#3.12) : Generate order and driver pair.  
[driver_online_offline_decision(driver_table, current_time)](#3.13) : empty.
[get_nodeId_from_coordinate(lat, lng)](#3.14) : Get the id of node according to its coordinate. 
---

### ```get_zone(lat, lng)```
<h2 id="3.1">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|lat      | float       | the latitude of coordinate| 
|lng      | float       | the id of the zone that the point belongs to|

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none      | -       | -     |  
</h2>

### ```distance(coord_1, coord_2)```
<h2 id="3.2">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|coord_1   | tuple(latitude,longitude) | the coordinate of one point    | 
|coord_2   | tuple(latitude,longitude) | the coordinate of another point|

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|manhattan_dis|float | manhattan distance between these two points |  
</h2>

### ```get_distance_array(origin_coord_array, dest_coord_array)```
<h2 id="3.3">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|origin_coord_array   | list | one list of coordinates      | 
|dest_coord_array     | list | another list of coordinates  |

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|manhattan_dis|float | manhattan distance between these two points |  
</h2>

### ```route_generation_array(origin_coord_array, dest_coord_array, mode='rg')```
<h2 id="3.4">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|origin_coord_array   | numpy.array | the K*2 type list, the first column is lng, the second columns lat | 
|dest_coord_array     | numpy.array | the K*2 type list, the first column is lng, the second columns lat |

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|itinerary_node_list| tuple | the id of nodes |
|itinerary_segment_dis_list| tuple |  distance between two nodes | 
|dis_array| tuple | the distance from the origin node to the destination node |   
</h2>

### ```load_data(self, data_path, file_name)```
<h2 id="3.5">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|data_path   | string | the path of road_network file    | 
|file_name   | string | the filename of road_network file|

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|none|- | - |  
</h2>

### ```get_information_for_nodes(node_id_array)```
<h2 id="3.6">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|node_id_array   | numpy.array | the array of node_id    | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|lng_array|tuple |the array of longitude| 
|lat_array|tuple | the array of latitude | 
|grid_id_array|tuple | the array of node id |  
</h2>

### ```get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10)```
<h2 id="3.7">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|initial_epsilon   | float | the initial epsilon  | 
|final_epsilon   | float | the final epsilon  | 
|steps   | int | the number of iterations | 
|decay   | float | the number of iterations of pre-randomness | 
|pre_steps   | int | the array of epsilon  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|epsilons|numpy.array |the array of epsilon|   

</h2>

### ```sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1, driver_number_dist='')```
<h2 id="3.8">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|driver_info   | pandas.DataFrame | the information of drivers  | 
|t_initial   | int | time of initial state  | 
|t_end   | int | time of terminal state  | 
|driver_sample_ratio   | float | the ratio of drivers sampled  | 
 
return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|sampled_driver_info|pandas.DataFrame|the information of sampled drivers|   

</h2>

### ```sample_request_num(t-mean, std, delta_t)```
<h2 id="3.9">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|t_mean   | float | mean value of the data  | 
|std   | float | Standard deviation of the data  | 
|delta_t   | int | -  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|request_num|int|the sample request num during delta t|   

</h2>

### ```reposition(eligible_driver_table, mode)```
<h2 id="3.10">
input:    
    
|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|eligible_driver_table   | pandas.DataFrame | information of eligible driver  | 
|mode   | string | the type of both-rg-cruising, if the type is random; it can cruise to every node with equal probability; if the type is nearby, it will cruise to the node in the adjacent grid or just stay at the original region.  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|itinerary_node_list|tuple| id of nodes |   
|itinerary_segment_dis_list|tuple| distance between two nodes |   
|dis_array|tuple| distance from origin node to destination node |   

</h2>

### ```cruising(eligible_driver_table, mode)```
<h2 id="3.11">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|eligible_driver_table   | pandas.DataFrame | information of eligible driver  | 
|mode   | string | the type of both-rg-cruising, if the type is random; it can cruise to every node with equal probability; if the type is nearby, it will cruise to the node in the adjacent grid or just stay at the original region.  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|itinerary_node_list|tuple| - |   
|itinerary_segment_dis_list|tuple| - |   
|dis_array|tuple| - |   

</h2>

### ```order_dispatch(wait_requests, driver_table, maximal_pickup_distance=950, dispatch_method='LD')```
<h2 id="3.12">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|wait_requests   | pandas.DataFrame | the requests of orders  | 
|driver_table   | pandas.DataFrame |  the information of online drivers  | 
|maximal_pickup_distance=950   | int | maximum of pickup distance  | 
|dispatch_method   | string | the method of order dispatch  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|matched_pair_actual_indexs|tuple|order and driver pair|   
|matched_itinerary|tuple|the itinerary of matched driver|   

</h2>

### ```driver_online_offline_decision(driver_table, current_time)```
<h2 id="3.13">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|driver_table   | pandas.DataFrame | info of drivers information  | 
|current_time   | int | time in 24h  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|new_driver_table|pandas.DataFrame|same to driver_table|   

</h2>

### ```get_nodeId_from_coordinate(lat, lng)```
<h2 id="3.14">
input:    

|   name   |   type  |describe|
|:----:    |  :----: | :----:|
|lat  | float | mean value of the data  | 
|lng   | float | Standard deviation of the data  | 

return:

|   name   | type | describe|
|:----:    |  :----: | :----:|
|node_list|string|the id of node|   

</h2>

</h2> 

[back](#0)

***

>## dispatch_alg.py

<h2 id="4.0">

### 
```LD(dispatch_observ)```  

input:

|                name                    |   type  |  describe      |
|                 :----:                 |  :----: | :----:         |
|dispatch_observ | pandas.DataFrame  | info about drivers and orders   | 

output:    

|                name                    |   type  |  describe      |
|                 :----:                 |  :----: | :----:         |
|dispatch_action | list  | result of dispatch   | 

</h2>



</h2>

***

> ## config.py

<h2 id="5.0">

```python
't_initial' # start time of the simulation (s)
't_end'  # end time of the simulation (s)
'delta_t' # interval of the simulation (s) 
'vehicle_speed' # speed of vehicle (km / h)
'repo_speed'  # speed of repositioning
'order_sample_ratio' # ratio of order sampling
'order_generation_mode'  # the mode of order generation
'driver_sample_ratio' : 1, # ratio of driver sampling
'maximum_wait_time_mean' : 300, # mean value of maximum waiting time
'maximum_wait_time_std' : 0, # variance of maximum waiting time
"maximum_pickup_time_passenger_can_tolerate_mean":float('inf'),  # s
"maximum_pickup_time_passenger_can_tolerate_std"
"maximum_price_passenger_can_tolerate_mean"
"maximum_price_passenger_can_tolerate_std"
'maximal_pickup_distance'  # km
'request_interval': 5,  #
'cruise_flag' :False, # 
'delivery_mode':'rg',
'pickup_mode':'rg',
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
'dispatch_method': 'LD', #LD: Lagrange decomposition method designed by Peibo Duan
# 'method': 'instant_reward_no_subway',
'simulator_mode' : 'toy_mode',
'experiment_mode' : 'train',
'driver_num':500,
'side':4, # grid side length
'price_per_km':5,  # ￥ / km
'road_information_mode':'load',
'north_lat': 40.8845,
'south_lat': 40.6968,
'east_lng': -74.0831,
'west_lng': -73.8414,
'rl_mode': 'reposition',  # reposition and matching
'method': 'sarsa_no_subway',  #  'sarsa_no_subway' / 'pickup_distance' / 'instant_reward_no_subway'   
'reposition_method' #2C_global_aware',  # A2C, A2C_global_aware, random_cruise, stay  
```
</h2> 

[back](#0) 