import pickle


def calculate_match_rate():
    count = 0
    orders = pickle.load(open('./input/dataset.pickle','rb'))
    for time in range(36000, 37500, 5):
        if time in orders.keys():
            count += len(orders[time])
    records = pickle.load(open('./toy_records_price.pickle','rb'))
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
    data2 = pickle.load(open('./input/dataset.pickle', 'rb'))
    print(data[1][1])
    print(data2[1][0])
    for time in data.keys():
        for order in data[time]:
            if order[11] is None or len(order[11]) == 1:
                data[time].remove(order)

    for time in data.keys():
        for order in data[time]:
            if order[11] is None or len(order[11]) == 1:
                print(order)

    pickle.dump(data, open('./input/order.pickle', 'wb'))


if __name__ == '__main__':
    delete_none_order()

