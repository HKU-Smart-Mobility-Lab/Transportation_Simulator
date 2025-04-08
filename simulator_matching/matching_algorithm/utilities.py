import numpy as np
import osmnx as ox
import networkx as nx
import h3 # type: ignore
import googlemaps

# 初始化 Google Maps 客户端
API_KEY = ""  # 替换为你的 API 密钥
# gmaps = googlemaps.Client(key=API_KEY)

def calculate_google_maps_distance(user_lat, user_lng, stand_lat, stand_lng, mode="driving"):
    """
    使用 Google Maps API 计算两点之间的路网距离。
    
    Parameters:
    - user_lat, user_lng: 用户的纬度和经度。
    - stand_lat, stand_lng: 的士站点的纬度和经度。
    - mode: 交通方式 ("driving", "walking", "bicycling", "transit")。
    
    Returns:
    - distance: 路网距离（以公里为单位）。
    - duration: 预计行驶时间（以分钟为单位）。
    """
    try:
        # 调用 Google Maps Distance Matrix API
        result = gmaps.distance_matrix(
            origins=(user_lat, user_lng),
            destinations=(stand_lat, stand_lng),
            mode=mode
        )
        
        # 提取距离和时间
        distance = result["rows"][0]["elements"][0]["distance"]["value"] / 1000  # 转换为公里
        print(f"Google Maps Distance: {distance} km")
        #duration = result["rows"][0]["elements"][0]["duration"]["value"] / 60  # 转换为分钟
        
        return distance
    except Exception as e:
        print(f"Google Maps API Error: {e}")
        return None, None




road_network = None  # 设置全局的道路网络

def load_road_network(lat, lon, dist=3000):
    global road_network
    if road_network is None:
        road_network = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
    return road_network



def calculate_h3_distance(user_lat, user_lng, stand_lat, stand_lng, h3_resolution=9, fallback_distance=99999):
    """
    使用 H3 网格系统计算两点之间的网格距离。
    
    Parameters:
    - user_lat, user_lng: 用户的纬度和经度。
    - stand_lat, stand_lng: 的士站点的纬度和经度。
    - h3_resolution: H3 网格分辨率（默认 9）。
    
    Returns:
    - 网格之间的距离（单位：跳数）。
    """
    try:
        user_h3 = h3.latlng_to_cell(user_lat, user_lng, h3_resolution)
        stand_h3 = h3.latlng_to_cell(stand_lat, stand_lng, h3_resolution)
        # 计算网格间的跳数
        h3_distance = h3.grid_distance(user_h3, stand_h3)
        print(f"H3 Distance: {h3_distance/1000} km")
        return h3_distance / 1000
    except Exception as e:
        print(f"H3 Error: {e}")
        print(f"Returning fallback distance of {fallback_distance/1000} km")
        return fallback_distance / 1000   


def calculate_osmnx_distance(lat1, lon1, lat2, lon2, fallback_distance=99999):
    global total_cal_dist_time  # Declare the use of the global variable
    try:
        G = load_road_network(lat1, lon1)

        
        orig_node = ox.distance.nearest_nodes(G, lon1, lat1)
        dest_node = ox.distance.nearest_nodes(G, lon2, lat2)
        
        # 检查路径是否存在
        if not nx.has_path(G, orig_node, dest_node):
            print("No path exists between the nodes. Returning fallback distance.")
            return fallback_distance
        
        length = nx.shortest_path_length(G, orig_node, dest_node, weight='length')
        print(f"OSMnx Distance: {length/1000} km")
        return length / 1000  # 转换为公里

    except ValueError:
        print(f"Could not find a valid path, returning fallback distance of {fallback_distance} km")
        print(f"Returning fallback distance of {fallback_distance/1000} km")
        return fallback_distance / 1000
    
def calculate_distances_batch(driver_lats, driver_lngs, order_lats, order_lngs, method="google", batch_size=10, h3_resolution=9, fallback_distance=999999):
    if method == "google":
        distances = []
        origins = list(zip(driver_lats, driver_lngs))
        destinations = list(zip(order_lats, order_lngs))
        for i in range(0, len(origins), batch_size):
            batch_origins = origins[i:i+batch_size]
            for j in range(0, len(destinations), batch_size):
                batch_destinations = destinations[j:j+batch_size]
                try:
                    results = gmaps.distance_matrix(batch_origins, batch_destinations, mode="driving")
                    for row in results["rows"]:
                        batch_distances = []
                        for elem in row["elements"]:
                            if "distance" in elem and elem["status"] == "OK":
                                distance_km = elem["distance"]["value"] / 1000
                                batch_distances.append(distance_km)
                            else:
                                batch_distances.append(fallback_distance / 1000)  # 返回默认值
                        distances.append(batch_distances)
                        # print(f"Google Maps distances (in km): {batch_distances}")
                except Exception as e:
                    print(f"Google Maps API Error in batch {i // batch_size + 1}, {j // batch_size + 1}: {e}")
                    distances.extend([[fallback_distance / 1000] * len(batch_destinations)] * len(batch_origins))  # 填充默认值
        return np.array(distances).reshape(len(driver_lats), len(order_lats))
    elif method == "h3":
        try:
            driver_h3 = [h3.latlng_to_cell(lat, lng, h3_resolution) for lat, lng in zip(driver_lats, driver_lngs)]
            order_h3 = [h3.latlng_to_cell(lat, lng, h3_resolution) for lat, lng in zip(order_lats, order_lngs)]
            distances = []
            for d_h3 in driver_h3:
                distances.append([h3.grid_distance(d_h3, o_h3) / 1000 for o_h3 in order_h3])
            return np.array(distances)
        except Exception as e:
            print(f"H3 Error: {e}")
            return np.full((len(driver_lats), len(order_lats)), fallback_distance / 1000)
    elif method == "osmnx":
        try:
            distances = []
            for driver_lat, driver_lng in zip(driver_lats, driver_lngs):
                G = load_road_network(driver_lat, driver_lng)
                orig_node = ox.distance.nearest_nodes(G, driver_lng, driver_lat)
                dest_nodes = [ox.distance.nearest_nodes(G, lng, lat) for lat, lng in zip(order_lats, order_lngs)]
                batch_distances = [
                    nx.shortest_path_length(G, orig_node, dest_node, weight="length") / 1000
                    if nx.has_path(G, orig_node, dest_node)
                    else fallback_distance / 1000  # 返回默认值
                    for dest_node in dest_nodes
                ]
                distances.append(batch_distances)
            return np.array(distances)
        except Exception as e:
            print(f"OSMnx Error: {e}")
            return np.full((len(driver_lats), len(order_lats)), fallback_distance / 1000)  # 返回默认值
    else:
        raise ValueError("Invalid distance calculation method.")