import pickle
import numpy as np
import pandas as pd


def calculate_match_rate():
    count = 0
    orders = pickle.load(open('./input/order.pickle','rb'))
    for time in range(36000, 36500, 5):
        if time in orders.keys():
            count += len(orders[time])
    records = pickle.load(open('./output/records_driver_num_500.pickle','rb'))
    matched = 0
    print(count)
    for time in records:
        for driver in time:
            if isinstance(time[driver][0], list):
                matched += 1
    print(matched)
    print(matched/count)


def delete_none_order():
    data = pickle.load(open('./input/order.pickle', 'rb'))
    for time in data.keys():
        for order in data[time]:
            if order[0] == 32752:
                print(order)
    # pickle.dump(data,open('./input/order.pickle','wb'))


if __name__ == '__main__':
    delete_none_order()
    # calculate_match_rate()
    # print(np.random.choice([1,2,3,4,5,6,7], 7,replace=False).tolist())
    # test = pd.DataFrame([[1,2,3],[2,3,4]])
    # indexs = [3,2]
    # new_indexs = []
    # for item in indexs:
    #     index = test[test[1] == item].index.tolist()[0]
    #     new_indexs.append(index)
    # print(new_indexs)
    # print(test.iloc[new_indexs])









