#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: zhangyuhao
@file: taxi_data_module.py
@time: 2022/2/6 上午1:39
@email: yuhaozhang76@gmail.com
@desc: 
"""
import pandas as pd
from tqdm import tqdm
from find_closest_point import find_closest_point
from datetime import datetime


def time_string_to_minutes(time):
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time_now = datetime.utcnow()
    minutes = int((time_now - time).total_seconds() / 60)
    return minutes


def t2s(t): #t format is hh:mm:ss
    if t != '0':
        h,m,s = t.split(" ")[-1].split(":")
#         print(h , m, s)
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
         return 0


def transform_data_from_nyu_to_one_day(input_file, output_file, day):
    driver_columns = ['ID', 'trip_distance', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat', 'trip_time']
    data = pd.read_csv(input_file)
    data = data[(data['data'] == day)]
    output_data = pd.DataFrame(driver_columns)
    for index, row in tqdm(data.iterrows()):
        origin_lat, origin_lng = find_closest_point(row[' pickup_latitude'], row[' pickup_longitude'])
        dest_lat, dest_lng = find_closest_point(row[' dropoff_latitude'], row[' dropoff_longitude'])
        row[' pickup_datetime'] = row[' pickup_datetime'].apply(t2s)
        record = [index, row[' trip_distance'], origin_lng, origin_lat, dest_lng, dest_lat, row[' pickup_datetime']]
        output_data = output_data.append(record, ignore_index=True)
    output_data.to_csv(output_file)


def transform_data_from_nyu_to_one_month(input_file, output_file):
    driver_columns = ['ID', 'trip_distance', 'origin_lng', 'origin_lat', 'dest_lng', 'dest_lat', 'trip_time', 'day']
    data = pd.read_csv(input_file)
    output_data = pd.DataFrame(driver_columns)
    for index, row in tqdm(data.iterrows()):
        time = row[' pickup_datetime']
        day = time.split(' ')[0].split('-')[2]
        # minutes = time_string_to_minutes(time)
        origin_lat, origin_lng = find_closest_point(row[' pickup_latitude'], row[' pickup_longitude'])
        dest_lat, dest_lng = find_closest_point(row[' dropoff_latitude'], row[' dropoff_longitude'])
        row[' pickup_datetime'] = row[' pickup_datetime'].apply(t2s)
        record = [index, row[' trip_distance'], origin_lng, origin_lat, dest_lng, dest_lat, row[' pickup_datetime'], day]
        output_data = output_data.append(record, ignore_index=True)
    output_data.to_csv(output_file)


if __name__ == '__main__':
    # input_path = './input/yellow_tripdata_2014-01.csv'
    input_path = './input/test.csv'
    output_path = './output/dataset.csv'
    transform_data_from_nyu_to_one_month(input_path, output_path)