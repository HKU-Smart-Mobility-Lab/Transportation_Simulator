import numpy as np
import time
import pickle
import os
from tqdm import tqdm
from simulator_env import Simulator
from reposition_agent import RepositionAgent
from utilities import *
import wandb

class RepositionTrainer:
    def __init__(self, simulator: Simulator, reposition_agent: RepositionAgent, train_config):
        """
        训练 `RepositionAgent` 的 `Trainer` 类
        :param simulator: `Simulator` 实例
        :param reposition_agent: `RepositionAgent` 实例
        :param train_config: 训练配置
        """
        self.simulator = simulator
        self.reposition_agent = reposition_agent
        self.train_config = train_config
        self.epsilons = get_exponential_epsilons(
            train_config['init_epsilon'], train_config['final_epsilon'], train_config['num_epochs'],
            decay=train_config['decay'], pre_steps=train_config['pre_steps']
        )
        self.total_reward_record = np.zeros(train_config['num_epochs'])
        # 初始化 Weights & Biases
        wandb.login()
        self.reposition_refactor = wandb.init(project="simulator_reposition_refactor",
                                       config={"repo_method": "A2C",
                                               "driver_num": 200,
                                               "EPOCH":601},)        

    def log_epoch_metrics(self, epoch, duration):
        """
        记录每个 Epoch 的指标
        """
        print(f"Epoch: {epoch}")
        print(f"Epoch running time: {duration:.2f}s")
        print(f"Total reward: {self.simulator.total_reward}")

        self.reposition_refactor.log({"Total reward": self.simulator.total_reward})

    def run_training_epoch(self, epoch):
        """
        执行一个完整的 `Reposition` 训练 Epoch
        """
        metrics = {'total_reward': 0, 'epoch_duration': 0}

        # 设置仿真环境
        self.simulator.reset()
        start_time = time.time()

        for step in tqdm(range(self.simulator.finish_run_step)):
            # 1️⃣ 获取状态
            state_array, index_grid = self.simulator.get_reposition_state()

            # 2️⃣ 选择动作
            action_array = self.reposition_agent.get_action(state_array, index_grid, self.epsilons[epoch])

            # 3️⃣ 执行动作
            self.simulator.execute_action(action_array)

        # 4️⃣ 训练智能体
        transitions = self.simulator.get_transitions()
        if all(t.shape[0] > 0 for t in transitions):
            self.reposition_agent.update(transitions)

        # 5️⃣ 记录训练数据
        end_time = time.time()
        metrics['total_reward'] = self.simulator.total_reward
        metrics['epoch_duration'] = end_time - start_time
        self.log_epoch_metrics(epoch, metrics['epoch_duration'])
        return metrics

    def save_training_results(self, epoch):
        """
        存储训练结果
        """
        output_path = self.train_config['output_path']
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if epoch % self.train_config['save_interval'] == 0:
            file_path = os.path.join(output_path, "reposition_training_results.pickle")
            with open(file_path, "wb") as f:
                pickle.dump(self.total_reward_record, f)
            print(f"Training reward record saved to {file_path}")

    def train(self):
        """
        执行完整的 `RepositionAgent` 训练
        """
        for epoch in range(self.train_config['num_epochs']):
            metrics = self.run_training_epoch(epoch)
            self.total_reward_record[epoch] = metrics['total_reward']
            self.save_training_results(epoch)

            if epoch == self.train_config['stop_epoch']:
                self.reposition_agent.save_model(epoch, self.train_config['model_timestamp'])
                self.reposition_refactor.finish()
                break
