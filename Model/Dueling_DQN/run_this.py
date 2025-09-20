# ------------------------------- #
# 模型参数设置
# ------------------------------- #
import gym
import torch
import numpy as np
from Model.Dueling_DQN.RL_brain import ReplayBuffer, DuelingDQN
from Model.Dueling_DQN.parsers import args
import matplotlib
matplotlib.use('Agg')  # 或 'TkAgg'
import matplotlib.pyplot as plt

n_hiddens = args.n_hiddens
learning_rate = args.lr
gamma = args.gamma
epsilon = args.epsilon
target_update = args.target_update
batch_size = args.batch_size
min_size = args.min_size
capacity = args.capacity

return_list = []  # 保存每回合的reward
max_q_value = 0  # 初始的动作价值函数
max_q_value_list = []  # 保存每一step的动作价值函数

# ------------------------------- #
# （1）加载环境
# ------------------------------- #

env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 4
n_actions = env.action_space.n  # 动作数 2

# ------------------------------- #
# （2）经验回放池和模型实例化
# ------------------------------- #

replay_buffer = ReplayBuffer(capacity)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

agent = DuelingDQN(n_states=n_states,
                   n_hidden=n_hiddens,
                   n_actions=n_actions,
                   learning_rate=learning_rate,
                   gamma=gamma,
                   epsilon=epsilon,
                   target_update=target_update,
                   device=device)

# ------------------------------- #
# （3）训练
# ------------------------------- #

for i in range(100):  # 训练100回合
    # 记录每个回合的return
    episode_return = 0
    # 获取初始状态
    state = env.reset()[0]
    # 结束的标记
    done = False

    # 开始迭代
    while not done:
        # 动作选择
        action = agent.take_action(state)
        # 动作价值函数，曲线平滑
        max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
        # 保存每一step的动作价值函数
        max_q_value_list.append(max_q_value)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        # 将经验存入经验回放池
        replay_buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 记录每个回合的return
        episode_return += reward

        # 当经验回放池容量超过一定值后，开始训练
        if replay_buffer.size() > min_size:
            # 随机采样
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            # 转换为字典
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'rewards': b_r,
                'next_states': b_ns,
                'dones': b_d
            }
            # 训练模型
            agent.update(transition_dict)

    # 保存每个回合的return
    return_list.append(episode_return)

    # 打印回合信息
    print(f'iter:{i}, return:{episode_return}, avg_return:{np.mean(return_list[-10:])}')

# 关闭动画
env.close()

# -------------------------------------- #
# 绘图
# -------------------------------------- #

plt.subplot(121)
plt.plot(return_list)
plt.title('return')
plt.subplot(122)
plt.plot(max_q_value_list)
plt.title('max_q_value')
plt.show()