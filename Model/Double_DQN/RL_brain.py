# # 基于策略的学习方法，用于数值连续的问题
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
#
#
# # ----------------------------------------------------- #
# # （1）构建训练网络
# # ----------------------------------------------------- #
# class Net(nn.Module):
#     def __init__(self, n_states, n_hiddens, n_actions):
#         super(Net, self).__init__()
#         # 只有一层隐含层的网络
#         self.fc1 = nn.Linear(n_states, n_hiddens)
#         self.fc2 = nn.Linear(n_hiddens, n_actions)
#
#     # 前向传播
#     def forward(self, x):
#         x = self.fc1(x)  # [b, states]==>[b, n_hiddens]
#         x = F.relu(x)
#         x = self.fc2(x)  # [b, n_hiddens]==>[b, n_actions]
#         # 对batch中的每一行样本计算softmax，q值越大，概率越大
#         x = F.softmax(x, dim=1)  # [b, n_actions]==>[b, n_actions]
#         return x
#
#
# # ----------------------------------------------------- #
# # （2）强化学习模型
# # ----------------------------------------------------- #
# class PolicyGradient:
#     def __init__(self, n_states, n_hiddens, n_actions,
#                  learning_rate, gamma):
#         # 属性分配
#         self.n_states = n_states  # 状态数
#         self.n_hiddens = n_hiddens
#         self.n_actions = n_actions  # 动作数
#         self.learning_rate = learning_rate  # 衰减
#         self.gamma = gamma  # 折扣因子
#         self._build_net()  # 构建网络模型
#
#     # 网络构建
#     def _build_net(self):
#         # 网络实例化
#         self.policy_net = Net(self.n_states, self.n_hiddens, self.n_actions)
#         # 优化器
#         self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
#
#     # 动作选择，根据概率分布随机采样
#     def take_action(self, state):  # 传入某个人的状态
#         # numpy[n_states]-->[1,n_states]-->tensor
#         state = torch.Tensor(state[np.newaxis, :])
#         # 获取每个人的各动作对应的概率[1,n_states]-->[1,n_actions]
#         probs = self.policy_net(state)
#         # 创建以probs为标准类型的数据分布
#         action_dist = torch.distributions.Categorical(probs)
#         # 以该概率分布随机抽样 [1,n_actions]-->[1] 每个状态取一组动作
#         action = action_dist.sample()
#         # 将tensor数据变成一个数 int
#         action = action.item()
#         return action
#
#     # 获取每个状态最大的state_value
#     def max_q_value(self, state):
#         # 维度变换[n_states]-->[1,n_states]
#         state = torch.tensor(state, dtype=torch.float).view(1, -1)
#         # 获取状态对应的每个动作的reward的最大值 [1,n_states]-->[1,n_actions]-->[1]-->float
#         max_q = self.policy_net(state).max().item()
#         return max_q
#
#     # 训练模型
#     def learn(self, transitions_dict):  # 输入batch组状态[b,n_states]
#         # 取出该回合中所有的链信息
#         state_list = transitions_dict['states']
#         action_list = transitions_dict['actions']
#         reward_list = transitions_dict['rewards']
#
#         G = 0  # 记录该条链的return
#         self.optimizer.zero_grad()  # 优化器清0
#         # 梯度上升最大化目标函数
#         for i in reversed(range(len(reward_list))):
#             # 获取每一步的reward, float
#             reward = reward_list[i]
#             # 获取每一步的状态 [n_states]-->[1,n_states]
#             state = torch.tensor(state_list[i], dtype=torch.float).view(1, -1)
#             # 获取每一步的动作 [1]-->[1,1]
#             action = torch.tensor(action_list[i]).view(1, -1)
#             # 当前状态下的各个动作价值函数 [1,2]
#             q_value = self.policy_net(state)
#             # 获取已action对应的概率 [1,1]
#             log_prob = torch.log(q_value.gather(1, action))
#             # 计算当前状态的state_value = 及时奖励 + 下一时刻的state_value
#             G = reward + self.gamma * G
#             # 计算每一步的损失函数
#             loss = -log_prob * G
#             # 反向传播
#             loss.backward()
#         # 梯度下降
#         self.optimizer.step()

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random


# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)

    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # 目前队列长度
    def size(self):
        return len(self.buffer)

    # -------------------------------------- #
    # 构造深度学习网络，输入状态s，得到各个动作的reward
    # -------------------------------------- #

    # class Net(nn.Module):
    #     # 构造只有一个隐含层的网络
    #     def __init__(self, n_states, n_hidden, n_actions):
    #         super(Net, self).__init__()
    #         # [b,n_states]-->[b,n_hidden]
    #         self.fc1 = nn.Linear(n_states, n_hidden)
    #         # [b,n_hidden]-->[b,n_actions]
    #         self.fc2 = nn.Linear(n_hidden, n_actions)
    #
    #     # 前传
    #     def forward(self, x):  # [b,n_states]
    #         x = self.fc1(x)
    #         x = self.fc2(x)
    #         return x


class Net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(Net, self).__init__()
        # 只有一层隐含层的网络
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b, states]==>[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b, n_hiddens]==>[b, n_actions]
        # 对batch中的每一行样本计算softmax，q值越大，概率越大
        x = F.softmax(x, dim=1)  # [b, n_actions]==>[b, n_actions]
        return x


# -------------------------------------- #
# 构造深度强化学习模型 (Double DQN)
# -------------------------------------- #

class DoubleDQN:
    # （1）初始化
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # （2）动作选择
    def take_action(self, state):
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
        state = torch.Tensor(state[np.newaxis, :])
        # 如果小于该值就取最大的值对应的索引
        if np.random.random() < self.epsilon:  # 0-1
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()  # int
        # 如果大于该值就随机探索
        else:
            # 随机选择一个动作
            action = np.random.randint(self.n_actions)
        return action

    # （3）网络训练 (Double DQN 关键修改)
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态 array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        # 下一时刻的状态 array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 获取当前Q值
        q_values = self.q_net(states).gather(1, actions)  # [b,1]

        # =============== Double DQN 关键修改 =============== #
        # 1. 使用在线网络选择下一状态的最优动作
        next_q_values = self.q_net(next_states)  # [b, n_actions]
        max_actions = next_q_values.argmax(dim=1).unsqueeze(1)  # [b, 1] 动作索引

        # 2. 使用目标网络评估该动作的价值
        next_target_values = self.target_q_net(next_states)  # [b, n_actions]
        max_next_q_values = next_target_values.gather(1, max_actions)  # [b, 1]
        # ================================================= #

        # 计算目标Q值：即时奖励+折扣因子*下个时刻的最大回报
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播参数更新
        dqn_loss.backward()
        # 对训练网络更新
        self.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        return self.q_net(state).max().item()
