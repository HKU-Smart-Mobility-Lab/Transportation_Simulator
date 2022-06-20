
import osmnx as ox
from config import *
# this function is to find the cloeset point in openstreetmap to real point.
# return a tuple(latitude,longitude)
def find_closest_point_by_big_map(lat,lng, mode = 'id'):
    print("params",env_params['north_lat'], env_params['south_lat'], env_params['east_lng']
         , env_params['west_lng'])
    G = ox.graph_from_bbox(env_params['north_lat'], env_params['south_lat'], env_params['east_lng']
         , env_params['west_lng'], network_type='drive_service')
    x = ox.distance.get_nearest_node(G, (lat, lng), method=None, return_dist=False)
    print("nodeid",x)
    if mode == 'id':
        return x

    nodes = ox.graph_to_gdfs(G, edges=False)
    point = nodes['geometry'][str(x)]
    return point.y,point.x


def find_closest_point(lat=40.736828, lng=-73.99477, mode = 'id'):
    G = ox.graph_from_bbox(lat + 0.01, lat - 0.01, lng - 0.01, lng + 0.01, network_type='drive_service')
    x = ox.distance.get_nearest_node(G, (lat, lng), method=None, return_dist=False)
    nodes = ox.graph_to_gdfs(G, edges=False)
    point = nodes['geometry'][x]
    if mode == 'id':
        return x, point.y, point.x
    nodes = ox.graph_to_gdfs(G, edges=False)
    point = nodes['geometry'][x]
    return point.y, point.x


if __name__ == "__main__":
    print(find_closest_point())