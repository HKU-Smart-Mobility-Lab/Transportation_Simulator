import numpy as np

class PricingAgent:
    def __init__(self, strategy="static", learning_rate=0.01, discount_rate=0.95):
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.price_table = {}  # 静态定价使用

        # 动态定价参数
        self.price_options = [2.5, 3.0, 3.5, 4.0, 4.5]  # 离散价格动作
        self.q_table = {}  # {(state_tuple): [Q-values for each price option]}

    def _discretize_state(self, pricing_state):
        d = np.mean(pricing_state["trip_distances"])  # 平均距离（km）
        s = pricing_state["supply"]
        dem = pricing_state["demand"]
        return (int(d * 10), s // 10, dem // 10)  # 简单离散化

    # pricing agent的行为:对订单返回一个价格pd.series
    def get_action(self, pricing_state, epsilon=0.1):
        trip_distances = pricing_state["trip_distances"]  # 应该是 Series 类型

        if self.strategy == "static":
            # 与旧代码一致：统一向量操作
            return 2.5 + 0.5 * ((1000 * trip_distances - 322).clip(lower=0) / 322)

        elif self.strategy == "dynamic":
            state_key = self._discretize_state(pricing_state)
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * len(self.price_options)

            if np.random.rand() < epsilon:
                price_per_km = np.random.choice(self.price_options)
            else:
                best_idx = np.argmax(self.q_table[state_key])
                price_per_km = self.price_options[best_idx]

            # 动态定价按距离乘以价格
            return price_per_km * trip_distances  # 向量运算

        else:
            raise ValueError("Unsupported pricing strategy")


    def update(self, pricing_state, action_prices, reward):
        if self.strategy != "dynamic":
            return

        state_key = self._discretize_state(pricing_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(self.price_options)

        avg_price = np.mean(action_prices)
        action_idx = min(range(len(self.price_options)), key=lambda i: abs(self.price_options[i] - avg_price))

        old_q = self.q_table[state_key][action_idx]
        self.q_table[state_key][action_idx] = (1 - self.learning_rate) * old_q + self.learning_rate * reward

    @staticmethod
    def observe(simulator):
        return {
            "supply": simulator.get_driver_count(),
            "demand": simulator.get_request_count(),
            "base_price": 2.5,
            "location": "global",
        }