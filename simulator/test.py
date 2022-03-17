import pickle
import numpy as np
import pandas as pd


def calculate_match_rate():
    driver_num = [500, 1000, 1500, 2000, 2500]
    max_distance_num = [0.5, 1, 2, 3]
    count = 0
    orders = pickle.load(open('./input/order.pickle', 'rb'))
    for time in range(36000, 79200):
        if time in orders.keys():
            for order in orders[time]:
                if order[0] == 49213:
                    print(order)
            count += len(orders[time])
    file = open('./output/match_rate.txt', 'w')
    for item in driver_num:
        for item2 in max_distance_num:
            tet = open('./input/temp.txt')
            x = set()
            for line in tet:
                x.add(line.strip())
            records = pickle.load(
                open('./output/records_driver_num_' + str(item) + '_distance_' + str(item2) + '.pickle', 'rb'))
            matched = 0
            print(count)
            for i,time in enumerate(records):
                for driver in time:
                    if isinstance(time[driver][0], list):
                        # if str(int(time[driver][0][2])) not in x:
                        print(i,driver,time[driver][0])
                        matched += 1
            print(matched)
            print(matched / count)
            file.write('driver_num: ' + str(item) + '\t' + 'distance_threshold: ' + str(
                item2) + '\t' + 'match_rate:' + str(matched / count) + '\n')

#
# test = (40.71561,-73.994272)
# print(test in node_coord_to_id.keys())


def delete_none_order():
    data = pickle.load(open('./input/order.pickle', 'rb'))
    # lng = data['lat'].values.tolist()
    # for item in lng:
    #     if item == 40.71561:
    #         print('test')
    # print(data[data['lng']== -73.994272])
    for time in data.keys():
        for order in data[time]:
            test = (order[2], order[3])

    pickle.dump(data, open('./input/order.pickle','wb'))


if __name__ == '__main__':
    # delete_none_order()
    calculate_match_rate()
    # t = np.array([1,2,3,4])
    # x = (t.cumsum()>10).argmax()
    # print(x)
    # print(np.random.choice([1,2,3,4,5,6,7], 7,replace=False).tolist())
    # test = pd.DataFrame([[1,2,3],[2,3,4]])
    # indexs = [3,2]
    # new_indexs = []
    # for item in indexs:
    #     index = test[test[1] == item].index.tolist()[0]
    #     new_indexs.append(index)
    # print(new_indexs)
    # print(test.iloc[new_indexs])









