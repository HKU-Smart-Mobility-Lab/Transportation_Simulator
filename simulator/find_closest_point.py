

import osmnx as ox

# this function is to find the cloeset point in openstreetmap to real point.
# return a tuple(latitude,longitude)
def find_closest_point(lat = 40.736828,lng = -73.99477):
    G = ox.graph_from_bbox(lat + 0.1, lat - 0.1, lng - 0.1, lng + 0.1, network_type='drive_service')
    x = ox.distance.get_nearest_node(G, (lat, lng), method=None, return_dist=False)
    return x
    # nodes = ox.graph_to_gdfs(G, edges=False)
    # point = nodes['geometry'][42532899]
    # return point.y,point.x


if __name__ == "__main__":
    print(find_closest_point())