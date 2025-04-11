import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from algorithm.utilities import calculate_distances_batch

def calculate_cost_matrix(drivers, orders, price_weight=0.2, rating_weight=0.05, power_weight=0.2, time_weight=0.5, padding_value=1e10):
    """
    Calculate the cost matrix using vectorized operations and DataFrame apply.
    Fill non-matching entries with padding_value instead of np.inf.
    """
    # 仅选择空闲司机
    idle_drivers = drivers[drivers["status"] == "idle"]

    if idle_drivers.empty or orders.empty:
        return np.array([])

    # 初始化成本矩阵
    num_drivers = len(idle_drivers)
    num_orders = len(orders)

    # 获取司机数据
    driver_positions = idle_drivers[["lat", "lon"]].values
    driver_speeds = np.maximum(idle_drivers["speed"].values, 1)
    driver_ratings = idle_drivers["rating"].values
    driver_powers = idle_drivers["remaining_power"].values / 100
    driver_ids = idle_drivers["driver_id"].values
    driver_rejections = idle_drivers["rejection_list"].values

    # 获取订单数据
    order_positions = orders[["origin_lat", "origin_lon"]].values
    order_prices = orders["price"].values
    order_ids = orders["order_id"].values
    order_preferences = orders["preference_list"].values

    # 计算距离矩阵
    distances = calculate_distances_batch(
        driver_positions[:, 0], driver_positions[:, 1],
        order_positions[:, 0], order_positions[:, 1],
        method="osmnx"
    )

    # 计算时间矩阵
    estimated_times = distances / driver_speeds[:, np.newaxis]

    # 归一化处理
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    estimated_times = (estimated_times - estimated_times.min()) / (estimated_times.max() - estimated_times.min())
    driver_powers = (driver_powers - driver_powers.min()) / (driver_powers.max() - driver_powers.min())
    order_prices = (order_prices - order_prices.min()) / (order_prices.max() - order_prices.min())
    driver_ratings = (driver_ratings - driver_ratings.min()) / (driver_ratings.max() - driver_ratings.min())

    # 计算成本矩阵
    costs = (
        distances +
        time_weight * estimated_times +
        power_weight * (1 - driver_powers[:, np.newaxis]) -
        price_weight * order_prices -
        rating_weight * driver_ratings[:, np.newaxis]
    )

    # 处理偏好列表和拒绝列表
    for i, order_id in enumerate(order_ids):
        preference_list = set(order_preferences[i])
        if preference_list:
            valid_drivers = np.isin(driver_ids, list(preference_list))
            costs[~valid_drivers, i] = padding_value

        for j, driver_id in enumerate(driver_ids):
            if order_id in driver_rejections[j]:
                costs[j, i] = padding_value

    # 确保是方阵
    max_dim = max(num_drivers, num_orders)
    padded_cost_matrix = np.full((max_dim, max_dim), padding_value)
    padded_cost_matrix[:num_drivers, :num_orders] = costs

    return padded_cost_matrix

def match_subset(drivers_subset, orders_subset, original_drivers, padding_value=1e10):
    """
    使用成本矩阵和匈牙利算法为子集匹配订单和司机。
    """
    # 仅选择空闲司机
    idle_drivers_subset = drivers_subset[drivers_subset["status"] == "idle"]

    cost_matrix = calculate_cost_matrix(idle_drivers_subset, orders_subset, padding_value=padding_value)
    if cost_matrix.size == 0 or np.all(np.isinf(cost_matrix)):
        print("Subset cost matrix is infeasible")
        return []

    try:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        print(f"Cost matrix is infeasible: {e}")
        return []

    assignments = []
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] == padding_value:
            continue

        driver_id = idle_drivers_subset.iloc[row]["driver_id"]
        order_id = orders_subset.iloc[col]["order_id"]
        preference_list = set(orders_subset.iloc[col]["preference_list"])
        rejection_list = set(idle_drivers_subset.iloc[row]["rejection_list"])

        # 拒绝列表检查
        if rejection_list and order_id in rejection_list:
            continue
        
        # 偏好列表检查
        if preference_list and driver_id not in preference_list:
            continue

        assignments.append({"driver_id": driver_id, "order_id": order_id})
        original_drivers.loc[idle_drivers_subset.index[row], "status"] = "matched"

    return assignments

def match_orders_to_drivers_sequential(drivers, orders, debug=False, padding_value=1e10):
    """
    Sequentially match orders to drivers in two steps:
    1. Match preferred orders with preferred drivers.
    2. Match all remaining orders with all remaining idle drivers.
    """
    # **Step 1: 被偏好司机匹配被偏好订单**
    print("Step 1: Matching preference orders with preference drivers...")
    
    preference_orders = orders[orders["preference_list"].apply(lambda x: len(x) > 0)]
    preference_drivers = set([driver for assignment in preference_orders["preference_list"] for driver in assignment])
    preference_drivers_df = drivers[(drivers["driver_id"].isin(preference_drivers)) & (drivers["status"] == "idle")]

    preference_assignments = match_subset(preference_drivers_df, preference_orders, drivers, padding_value=padding_value)

    # **Step 2: 剩余所有订单与所有空闲司机匹配**
    print("Step 2: Matching all remaining orders with all remaining idle drivers...")
    
    # 更新已匹配订单
    matched_order_ids = {a["order_id"] for a in preference_assignments}
    remaining_orders = orders[~orders["order_id"].isin(matched_order_ids)]

    # 更新已匹配司机
    remaining_drivers = drivers[drivers["status"] == "idle"]

    # 进行全局匹配
    remaining_assignments = match_subset(remaining_drivers, remaining_orders, drivers, padding_value=padding_value)

    # **组合所有匹配结果**
    assignments = preference_assignments + remaining_assignments

    if debug:
        print(f"Final Assignments: {assignments}")

    return assignments
