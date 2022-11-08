import numpy as np
import pandas as pd
import pickle
from copy import deepcopy

def LD(dispatch_observ):
    columns_name = ['order_id', 'driver_id', 'order_driver_flag','reward_units']
    #dic_dispatch_observ = dispatch_observ.copy()
    dispatch_observ = pd.DataFrame(dispatch_observ, columns=columns_name)
    dispatch_observ['reward_units'] = 0. + dispatch_observ['reward_units'].values
    dic_dispatch_observ = dispatch_observ.to_dict(orient='records')

    dispatch_action = []

    # get orders and drivers
    l_orders = dispatch_observ['order_id'].unique()  # df: order id
    l_drivers = dispatch_observ['driver_id'].unique() # df: driver id

    M = len(l_orders)  # the number of orders
    N = len(l_drivers)  # the number of drivers

    # coefficients and parameters, formulated as M * N matrix
    non_exist_link_value = 0.
    matrix_reward = non_exist_link_value + np.zeros([M, N])  # reward     # this value should be smaller than any possible weights
    matrix_flag = np.zeros([M, N])  # pick up distance
    matrix_x_variables = np.zeros([M, N])  # 1 means there is potential match. otherwise, 0

    index_order = np.where(dispatch_observ['order_id'].values.reshape(dispatch_observ.shape[0], 1) == l_orders)[1]
    index_driver = np.where(dispatch_observ['driver_id'].values.reshape(dispatch_observ.shape[0], 1) == l_drivers)[1]

    matrix_reward[index_order, index_driver] = dispatch_observ['reward_units'].values
    matrix_flag[index_order, index_driver] = dispatch_observ['order_driver_flag'].values
    matrix_x_variables[index_order, index_driver] = 1

    # algorithm ----------------------------------------------------------------------------------------------------
    # initialize lower bound of the solution
    initial_best_reward = 0
    initial_best_solution = np.zeros([M, N])
    dic_dispatch_observ.sort(key=lambda od_info: od_info['reward_units'], reverse=True)
    assigned_order = set()
    assigned_driver = set()
    initial_dispatch_action = []
    for od in dic_dispatch_observ:
        # make sure each order is assigned to one driver, and each driver is assigned with one order
        if (od["order_id"] in assigned_order) or (od["driver_id"] in assigned_driver):
            continue
        assigned_order.add(od["order_id"])
        assigned_driver.add(od["driver_id"])
        initial_dispatch_action.append(dict(order_id=od["order_id"], driver_id=od["driver_id"]))
    df_init_dis = pd.DataFrame(initial_dispatch_action)
    index_order_init = np.where(df_init_dis['order_id'].values.reshape(df_init_dis.shape[0], 1) == l_orders)[1]
    index_driver_init = np.where(df_init_dis['driver_id'].values.reshape(df_init_dis.shape[0], 1) == l_drivers)[1]
    initial_best_reward += np.sum(matrix_reward[index_order_init, index_driver_init])
    initial_best_solution[index_order_init, index_driver_init] = 1

    max_iterations = 30  # 25
    u = np.zeros(N)  # initialization
    Z_LB = initial_best_reward  # the lower bound of original problem that is initialized with the naive algorithm
    Z_UP = float('inf')  # infinity
    theta = 1.0
    gap = 0.0001

    # ---------------------------------------------Start iteration--------------------------------------------------
    for t in range(1, max_iterations + 1):
        matrix_x = np.zeros([M,N])
        QI = matrix_reward - u
        QI_masked = np.ma.masked_where(matrix_x_variables != 1, QI)
        idx_col_array = np.argmax(QI_masked, axis=1)
        idx_row_array = np.array(range(M))
        matrix_x[idx_row_array, idx_col_array] = 1

        # calculate Z_UP and Z_D
        Z_D = np.sum(u) +  np.sum(matrix_reward * matrix_x)
        Z_UP = Z_D if Z_D < Z_UP else Z_UP

        #stage 1
        copy_matrix_reward = non_exist_link_value + np.zeros([M,N])
        copy_matrix_reward[idx_row_array, idx_col_array] = matrix_reward[idx_row_array, idx_col_array]
        copy_matrix_x = np.zeros([M,N])
        idx_col_array = np.array(range(N))
        idx_row_array = np.argmax(copy_matrix_reward, axis=0)
        con = copy_matrix_reward[idx_row_array, idx_col_array] > non_exist_link_value
        idx_col_array = idx_col_array[con]
        idx_row_array = idx_row_array[con]
        if len(idx_row_array) > 0:
            copy_matrix_x[idx_row_array, idx_col_array] = 1

        #stage 2
        index_existed_pair = np.where(copy_matrix_x == 1)
        index_drivers_with_order = np.unique(index_existed_pair[1])
        index_drivers_without_order = np.setdiff1d(np.array(range(N)), index_drivers_with_order)
        index_orders_with_driver = np.unique(index_existed_pair[0])
        index_orders_without_driver = np.setdiff1d(np.array(range(M)), index_orders_with_driver)

        if len(index_orders_without_driver) != 0:
            second_allocated_driver = []
            for m in index_orders_without_driver.tolist():
                con_second = np.isin(index_drivers_without_order, second_allocated_driver)
                if np.all(con_second):
                    break
                else:
                    reward_array = matrix_reward[m][index_drivers_without_order]
                    masked_reward_array = np.ma.masked_where(con_second, reward_array)
                    index_reward = np.argmax(masked_reward_array)
                    if masked_reward_array[index_reward] > 0:
                        index_driver = index_drivers_without_order[index_reward]
                        second_allocated_driver.append(index_driver)
                        copy_matrix_x[m][index_driver] = 1

        #stage 3
        new_Z_LB = np.sum(copy_matrix_x * matrix_reward)
        if new_Z_LB > Z_LB:
            Z_LB = new_Z_LB
            initial_best_solution = np.zeros([M, N])
            initial_best_solution[copy_matrix_x == 1] = 1

        # update u
        sum = 0
        sum_m = np.sum(matrix_x, axis=0)
        sum = np.sum((1 - sum_m) ** 2)
        if sum == 0:
            sum = 0.00001  # given a small value
        k_t = theta * (Z_D - Z_LB) / sum

        u = u + k_t * (sum_m - 1) / t
        u[u<0] = 0

        if (Z_UP == 0) or ((Z_UP - Z_LB) / Z_UP <= gap):
            matrix_x = initial_best_solution
            break
        if t == max_iterations:
            matrix_x = initial_best_solution
            break

    # solution
    index_existed = np.where(matrix_x == 1)
    for m, index_driver in zip(index_existed[0].tolist(), index_existed[1].tolist()):
        dispatch_action.append([l_orders[m], l_drivers[index_driver], matrix_reward[m][index_driver],
                                matrix_flag[m][index_driver]])

    return dispatch_action
