### An Open-Sourced Network-Based Large-Scale Simulation Platform for Shared Mobility Operations

![效果图](https://img.shields.io/static/v1?label=build&message=passing&color=green) ![](https://img.shields.io/static/v1?label=python&message=3&color=blue) ![](https://img.shields.io/static/v1?label=release&message=2.0&color=green) ![](https://img.shields.io/static/v1?label=license&message=MIT&color=blue)

### Background

Establish an open-sourced network-based simulation platform for shared mobility operations. The simulation explicitly characterizes drivers’ movements on road networks for trip delivery, idle cruising, and en-route pick-up. 



### Contributions

• We develop a comprehensive, multi-functional and open-sourced simulator for ride-sourcing service, which can be used by both researchers and industrial practitioners on a variety of operational tasks. The proposed simulation platform overcomes a few challenges faced by previous simulators, including the closeness to reality, representation of customers’ and drivers’ heterogeneous behaviors, generalization for different tasks. 

• The simulator provide interfaces for the training and testing of different tasks, such as testing of optimization algorithms, training/testing of reinforcement learning based approaches for matching, repositioning and pricing, evaluations of economic models for equilibrium analysis and operational strategy designs. Each module is organized as an independent package, making the system easier to maintain, extend, and experiment with reinforcement learning (RL) or rule-based strategies.

• Based on a vehicle utilization based validation task, some RL based experiments, and one task for theoretical model evaluation, the simulator is validated to be effective and efficient for ride-sourcing related researches. In the future, the simulator can be easily modified for completing other tasks, such as dynamic pricing, ride-pooling service operations, control of shared autonomous vehicles, etc.

### Install

1. Download the code

  `git clone git@github.com:HKU-Smart-Mobility-Lab/Transportation_Simulator.git`

2. Pull the docker image

  `docker pull jingyunliu663/simulator`

- after running the code, you can use `docker images` to check whether the image is available
- the docker image comes with the conda environment `new_simulator` and the mongoDB service running in background within the container

3. Run the docker image & get a docker container
```bash
docker run -d -e CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 -v /path/to/the/Transportation_Simulator:/simulator/scripts --name simulator jingyunliu663/simulator
```
- Arguments:
  - `-d`: detach, run in background
  - `-v path/to/your/local/file/system:/path/to/your/target/directory/inside/container`: This will mount the local directory into the container. Any changes made to the files in this directory within the container will also be reflected on the local host system, and vice versa.
  - `--name`: name the container as *simulator*
  - the last argument is the image name (after all, container is a running instance of an image)

- you can use `docker ps` to check the running containers

4. Enter the interactive shell of the conatiner `simulator`
```bash
docker exec -it simulator bash -c "cd / && exec bash"

```

- After enter the interactive shell , you will be in the working directory `/simulator`(the directory you choose to mount to), you can navigate yourself to  `/simulator/scripts` directory to run the main function


### Download Data
Here, we provide you data for Manhattan that we used in our experiments. You can download it from Onedrive.(https://connecthkuhk-my.sharepoint.com/:f:/g/personal/ctj21_connect_hku_hk/EuHy9dMKNy1CkZf5K87GMlYBAI54ZPeGzjanbrpBTZO5CQ?e=95BBYF). And due to the data privacy, we can not provide you the data for Hong Kong and Chengdu.

### Survey Data
We conducted a survey in NYC, and one hundred people participated in this survey. The content and results of the questionnaire can be found in the Survey_E-hailing_Taxi_Service_Studies.xlsx file. Hope that this data can help you for your research. And if you use these data in your research, please cite our paper: 
Feng S., Chen T., Zhang Y., Ke J.* & H. Yang, 2023. A multi-functional simulation platform for on-demand ride service operations. Preprint at ArXiv:2303.12336. https://doi.org/10.48550/arXiv.2303.12336.

### File Structure

```
- simulator_matching/
  - main.py
  - simulator_env.py
  - matching_agent.py
  - config.py
  - input/
  - output/
  - utils/
  
- simulator_pricing/
  - main.py
  - simulator_env.py
  - pricing_agent.py
  - config.py
  - input/
  - output/
  - utils/

- simulator_reposition/
  - main.py
  - simulator_env.py
  - reposition_agent.py
  - config.py
  - input/
  - output/
  - utils/

- readme.md
```

##### Data preparing

There are three files in 'input' directory. You can use the driver data and order data provided by us. Also, you can run `python handle_raw_data.py`  to generate orders' information, run `python driver_generation.py`  to generate drviers' information.  

In [config.py](https://github.com/HKU-Smart-Mobility-Lab/Transpotation_Simulator/blob/main/simulator/config.py), you can set the parameters of the simulator.

```python
't_initial' # start time of the simulation (s)
't_end'  # end time of the simulation (s)
'delta_t' # interval of the simulation (s) 
'vehicle_speed' # speed of vehicle (km / h)
'repo_speed'  # speed of reposition
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
'dispatch_method': 'LD', #LD: lagarange decomposition method designed by Peibo Duan
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

##### Real Road Network Module

We use [osmnx](https://pypi.org/project/osmnx/) to acquire the shortest path from the real world. You can set 'north_lat', 'south_lat', 'east_lng' and 'west_lng' in [config.py](https://github.com/HKU-Smart-Mobility-Lab/Transpotation_Simulator/blob/main/simulator/config.py) , and you can get road network for the specified region.

##### Mongodb Module

If your road network is huge, yu can use mongodb to store the road network information and add index on node id which uniquely identifies a node on the road network.
You can use the following code to connect your local database route_network and collection route_list which stores the information. After that, you can use find_one interface to achieve the shortest path list easily. 
```python
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["route_network"]
mycollect = mydb['route_list']

re_data = {
            'node': str(ori_id) + str(dest_id)
        }
re = mycollect.find_one(re_data)
```


##### Price Module

This module implements dynamic pricing based on trip distance, supply-demand conditions, and passenger cancellation behavior. The pricing agent can be configured as a static rule-based system or trained using tabular Q-learning to adaptively maximize platform revenue while retaining customer acceptance.

##### Cruising and Repositioning Module

This module determines idle driver movements after order completion. Reinforcement learning strategies such as A2C or rule-based methods (random cruise, stay) are supported. It helps optimize system-wide occupancy rate and improves driver utilization across the grid.


##### Dispatching Module

In dispatch_alg.py, we implement the function LD, we use binary map matching algorithm to dispatch orders. It supports various matching algorithms, including instant reward-based heuristics, Q-learning, and SARSA-based reinforcement learning. Matching weights can be dynamically adjusted based on state-action values learned during training.

##### Experiment

You can modify the parameters in [config.py](https://github.com/HKU-Smart-Mobility-Lab/Transpotation_Simulator/blob/main/simulator/config.py), and then excute `python main.py`. The records will be recorded in the directory named output.

### Tutorials







### Technical questions

We welcome your contributions.

- Please report bugs and improvements by submitting [GitHub issue](https://github.com/HKU-Smart-Mobility-Lab/Transpotation_Simulator/issues).
- Submit your contributions using [pull requests](https://github.com/HKU-Smart-Mobility-Lab/Transpotation_Simulator/pulls). 



### Contributors

This simulator is supported by the [Smart Mobility Lab](	https://github.com/HKU-Smart-Mobility-Lab) at The Univerisity of Hong Kong and [Intelligent Transportation Systems (ITS) Laboratory](https://sites.google.com/view/hkustits/home) at The Hong Kong University of Science and Technology.



### Ownership

The ownership of this repository is Prof. Hai Yang, Dr. Siyuan Feng from ITS Lab at The Hong Kong University of Science and Technology and Dr. Jintao Ke from SML at The Univerisity of Hong Kong.

###  Cite

@article{feng2024multi,
  title={A multi-functional simulation platform for on-demand ride service operations},
  author={Feng, Siyuan and Chen, Taijie and Zhang, Yuhao and Ke, Jintao and Zheng, Zhengfei and Yang, Hai},
  journal={Communications in Transportation Research},
  volume={4},
  pages={100141},
  year={2024},
  publisher={Elsevier}
}













##### 
