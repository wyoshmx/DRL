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
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


# -------------------------------------- #
# Dueling DQN 网络结构
# -------------------------------------- #

class DuelingNet(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(DuelingNet, self).__init__()
        # 共享的特征提取层
        self.feature = nn.Sequential(
            nn.Linear(n_states, n_hidden),
            nn.ReLU()
        )

        # 优势函数分支
        self.advantage = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )

        # 状态价值函数分支
        self.value = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        # 组合价值函数和优势函数: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# -------------------------------------- #
# Dueling DQN 算法
# -------------------------------------- #

class DuelingDQN:
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        self.n_states = n_states
        self.n_hidden = n_hidden
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0

        # 实例化训练网络和目标网络
        self.q_net = DuelingNet(self.n_states, self.n_hidden, self.n_actions)
        self.target_q_net = DuelingNet(self.n_states, self.n_hidden, self.n_actions)

        # 目标网络参数初始化
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            state = torch.Tensor(state[np.newaxis, :])
            actions_value = self.q_net(state)
            action = actions_value.argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 当前Q值
        q_values = self.q_net(states).gather(1, actions)

        # 使用Double DQN策略
        next_q_values = self.q_net(next_states)
        next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
        next_target_values = self.target_q_net(next_states)
        max_next_q_values = next_target_values.gather(1, next_actions)

        # 目标Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        max_q = self.q_net(state).max().item()
        return max_q