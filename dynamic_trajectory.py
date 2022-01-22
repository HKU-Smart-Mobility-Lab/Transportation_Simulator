import folium
import os
from tqdm import tqdm
from folium import plugins
import pickle
import json
import csv


def draw_gps(locations_true, output_path, file_name):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    m = folium.Map(locations_true[0][0], zoom_start=30, attr='default')  # 中心区域的确定
    for single_route in locations_true:

        lc = folium.PolyLine(  # polyline方法为将坐标用实线形式连接起来
            single_route,  # 将坐标点连接起来
            weight=4,  # 线的大小为4
            color='red',  # 线的颜色为红色
            opacity=0.8,  # 线的透明度
        ).add_to(m)  # 将这条线添加到刚才的区域m内

        # attr = {"fill": "#007DEF", "font-weight": "bold", "font-size": "20"}
        # plugins.PolyLineTextPath(
        #     lc, "--> **", repeat=True, offset=5, attributes=attr
        # ).add_to(m)
        plugins.AntPath(
            locations=single_route, reverse=False, dash_array=[20, 30]
        ).add_to(m)

        folium.Marker(single_route[0], popup='<b>Starting Point</b>').add_to(m)
        folium.Marker(single_route[-1], popup='<b>End Point</b>').add_to(m)

    m.save(os.path.join(output_path, file_name))  # 将结果以HTML形式保存到指定路径


def generate_route_lat_lng(route_file, node_file):
    route_info = pickle.load(open(route_file, 'rb'))
    with open('./data/node_json.json', 'r', encoding='utf-8') as file:
        id_lng_lat = json.load(file)
    route = []
    for i in range(len(route_info)):
        temp_route = []
        for node in route_info.iloc[i]['itinerary_segment_dis_list']:
            if str(node) in list(id_lng_lat.keys()):
                temp_route.append(id_lng_lat[str(node)])
            else:
                print(node)
        route.append(temp_route)
    return route


def generate_car_csv(all_route):
    route_file = open('./data/route.csv', 'w')
    route_csv = csv.writer(route_file)
    route_csv.writerow(['geometry'])
    for s_route in all_route:
        temp_lat_lng = []
        for item in s_route:
            temp_lat_lng.append([item[1],item[0]])
        temp_route = {
            "type": "LineString",
            "coordinates": temp_lat_lng
        }
        route_csv.writerow([json.dumps(temp_route)])


if __name__ == '__main__':
    node_file = './data/road_network_information.pickle'
    route_file = './data/requests.pickle'
    all_route = generate_route_lat_lng(route_file, node_file)
    generate_car_csv(all_route)
    # draw_gps(all_route, './data', 'plan_A.html')
