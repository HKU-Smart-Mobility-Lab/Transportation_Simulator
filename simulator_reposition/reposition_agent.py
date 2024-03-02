import numpy as np
from Transportation_Simulator.simulator_reposition.reposition_strategy_base.A2C import *  # 确保 A2C 正确导入

class RepositionAgent:
    def __init__(self, agent_params):
        """ 只接受 `agent_params`，不引入 `simulator` """
        self.agent = A2C(**agent_params)

    def get_action(self, state_array, index_grid, epsilon):
        """ 通过强化学习代理选择动作 """
        if state_array is None:
            return np.array([])  # 无可用状态时，不执行任何动作

        action_array = np.zeros(state_array.shape[0])
        for i in range(len(action_array)):
            action_array[i] = self.agent.egreedy_actions(state_array[i], epsilon, index_grid[i])

        return action_array

    def update(self, transitions):
        """ 训练智能体：收集经验并更新策略 """
        if all(t.shape[0] > 0 for t in transitions):
            self.agent.perceive(transitions)

    def save_model(self, epoch, model_timestamp):
        """ 保存模型 """
        self.agent.save_model(epoch, model_timestamp)
