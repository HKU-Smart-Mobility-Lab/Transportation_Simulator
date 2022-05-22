import warnings
import osmnx as ox
import pandas as pd
import math
import queue
import threading
import pickle
import time
from tqdm import tqdm
import pymongo
from math import radians, sin, atan2
warnings.filterwarnings("ignore")
exitFlag = 0

env_params = {
    'north_lat': 40.8845,
    'south_lat': 40.6968,
    'east_lng': -74.0831,
    'west_lng': -73.8414
}



G = ox.load_graphml('./input/graph.graphml')
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
lat_list = gdf_nodes['y'].tolist()
lng_list = gdf_nodes['x'].tolist()
node_id = gdf_nodes.index.tolist()
node_id_to_lat_lng = {}
node_coord_to_id = {}
for i in range(len(lat_list)):
    node_id_to_lat_lng[node_id[i]] = (lat_list[i], lng_list[i])
    node_coord_to_id[(lat_list[i], lng_list[i])] = node_id[i]

center = (
    (env_params['east_lng'] + env_params['west_lng']) / 2, (env_params['north_lat'] + env_params['south_lat']) / 2)
radius = max(abs(env_params['east_lng'] - env_params['west_lng']) / 2,
             abs(env_params['north_lat'] - env_params['south_lat']) / 2)
side = 10
interval = 2 * radius / side

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["route_network"]

mycollect = mydb['route_list']

def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    lat1, lon1, = coord_1
    lat2, lon2 = coord_2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)
    r = 6371

    alat = sin(dlat / 2) ** 2
    clat = 2 * atan2(alat ** 0.5, (1 - alat) ** 0.5)
    lat_dis = clat * r

    alon = sin(dlon / 2) ** 2
    clon = 2 * atan2(alon ** 0.5, (1 - alon) ** 0.5)
    lon_dis = clon * r

    manhattan_dis = abs(lat_dis) + abs(lon_dis)

    return manhattan_dis

def get_zone(lat, lng):
    """
    :param lat: the latitude of coordinate
    :type : float
    :param lng: the longitude of coordinate
    :type lng: float
    :return: the id of zone that the point belongs to
    :rtype: float
    """
    if lat < center[1]:
        i = math.floor(side / 2) - math.ceil((center[1] - lat) / interval) + side % 2
    else:
        i = math.floor(side / 2) + math.ceil((lat - center[1]) / interval) - 1

    if lng < center[0]:
        j = math.floor(side / 2) - math.ceil((center[0] - lng) / interval) + side % 2
    else:
        j = math.floor(side / 2) + math.ceil((lng - center[0]) / interval) - 1
    return i * side + j


data = pd.read_csv('C:\\Users\\kejintao\\Downloads\\yellow_tripdata_2015-05.csv')
data_num = len(data)
pbar = tqdm(total=data_num)
queueLock = threading.Lock()
workQueue = queue.Queue(data_num+1)
threads = []

ori_id_list = []
origin_lng = []
origin_lat = []
ori_grid_id_list = []

dest_id_list = []
dest_lng = []
dest_lat = []
dest_grid_id_list = []
itinerary_node_list = []
itinerary_segment_dis_list = []
dis_array = []
pickup_time = []
pickup_distance = []


class myThread (threading.Thread):
    def __init__(self, threadID, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.q = q

    def run(self):
        process_data(self.q)


def process_data(q):
    while not exitFlag:
        if not workQueue.empty():
            data = q.get()
            # try:
            # temp_record = {}
            x = ox.distance.get_nearest_node(G, (data[6], data[5]), method=None, return_dist=False)
            point = gdf_nodes['geometry'][x]
            ori_id, temp_ori_lat, temp_ori_lng = x, point.y, point.x
            x = ox.distance.get_nearest_node(G, (data[10], data[9]), method=None, return_dist=False)
            point = gdf_nodes['geometry'][x]
            dest_id, temp_dest_lat, temp_dest_lng = x, point.y, point.x
            ori_id_list.append(ori_id)
            origin_lng.append(temp_ori_lng)
            origin_lat.append(temp_ori_lat)
            ori_grid_id_list.append(get_zone(temp_ori_lat,temp_ori_lng))
            dest_id_list.append(dest_id)
            dest_lat.append(temp_dest_lat)
            dest_lng.append(temp_dest_lng)
            dest_grid_id_list.append(get_zone(temp_dest_lat,temp_dest_lng))
            # pickup_distance.append(data[4])
            pickup_time.append(data[1])
            data = {
                'node': str(ori_id) + str(dest_id)
            }
            re = mycollect.find_one(data)
            if re:
                ite = [int(item) for item in re['itinerary_node_list'].strip('[').strip(']').split(', ')]
            else:
                ite = ox.distance.shortest_path(G, ori_id, dest_id, weight='length', cpus=16)
            if ite is not None and len(ite) > 1:
                itinerary_node_list.append(ite)
                itinerary_segment_dis = []
                for i in range(len(ite) - 1):
                    # dis = nx.shortest_path_length(G, node_id_to_lat_lng[itinerary_node[i]], node_id_to_lat_lng[itinerary_node[i + 1]], weight='length')

                    dis = distance(node_id_to_lat_lng[ite[i]],
                                   node_id_to_lat_lng[ite[i + 1]])
                    itinerary_segment_dis.append(dis)
                pickup_distance.append(sum(itinerary_segment_dis))
                itinerary_segment_dis_list.append(itinerary_segment_dis)
            else:
                itinerary_node_list.append([ori_id, dest_id])
                dis = distance(node_id_to_lat_lng[ori_id],
                                   node_id_to_lat_lng[dest_id])
                pickup_distance.append(dis)
                itinerary_segment_dis_list.append(dis)

            pbar.update(1)
            # except Exception as err:
            #     print(err)
        # time.sleep(1)


t1 = time.time()
for i in range(10):
    thread = myThread(i, workQueue)
    thread.start()
    threads.append(thread)


# 填充队列
queueLock.acquire()
for i in range(len(data)):
    workQueue.put(data.iloc[i].values.tolist())
queueLock.release()
# 等待队列清空
while not workQueue.empty():
    pass

exitFlag = 1
for t in threads:
    t.join()

print(time.time()-t1)
pd_data = pd.DataFrame()
pd_data['order_id'] = [i for i in range(len(origin_lat))]
pd_data['origin_id'] = ori_id_list
pd_data['origin_lat'] = origin_lat
pd_data['origin_lng'] = origin_lng
pd_data['dest_id'] = dest_id_list
pd_data['dest_lat'] = dest_lat
pd_data['dest_lng'] = dest_lng
pd_data['trip_distance'] = pickup_distance
pd_data['start_time'] = pickup_time
pd_data['origin_grid_id'] = ori_grid_id_list
pd_data['dest_Grid_id'] = dest_grid_id_list
pd_data['itinerary_node_list'] = itinerary_node_list
pd_data['itinerary_segment_dis_list'] = itinerary_segment_dis_list
pd_data['trip_time'] = 0
pd_data['cancel_prob'] = 0
pd_data.to_csv('dataset.csv',index=False)
# pickle.dump(result, open('./simulator/output/multi_thread_2015.pickle', 'wb'))
print("退出主线程")