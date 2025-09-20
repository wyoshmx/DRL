import gym
import torch
import numpy as np
from Model.Double_DQN.RL_brain import ReplayBuffer, DoubleDQN
from Model.Double_DQN.parsers import args
import matplotlib

matplotlib.use('Agg')  # 或 'TkAgg'
import matplotlib.pyplot as plt

# ------------------------------- #
# 模型参数设置
# ------------------------------- #

n_hiddens = args.n_hiddens
learning_rate = args.lr
gamma = args.gamma
epsilon = args.epsilon
target_update = args.target_update
batch_size = args.batch_size
min_size = args.min_size
capacity = args.capacity

return_list = []
max_q_value = 0
max_q_value_list = []

env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

replay_buffer = ReplayBuffer(capacity)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

agent = DoubleDQN(n_states=n_states,
                  n_hidden=n_hiddens,
                  n_actions=n_actions,
                  learning_rate=learning_rate,
                  gamma=gamma,
                  epsilon=epsilon,
                  target_update=target_update,
                  device=device)

for i in range(100):
    episode_return = 0
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.take_action(state)
        max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
        max_q_value_list.append(max_q_value)
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward

        if replay_buffer.size() > min_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'rewards': b_r,
                'next_states': b_ns,
                'dones': b_d
            }
            agent.update(transition_dict)

    return_list.append(episode_return)
    print(f'iter:{i}, return:{episode_return}, avg_return:{np.mean(return_list[-10:])}')

env.close()

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(return_list)
plt.title('Episode Return')
plt.xlabel('Episode')
plt.ylabel('Return')

plt.subplot(122)
plt.plot(max_q_value_list)
plt.title('Max Q Value')
plt.xlabel('Step')
plt.ylabel('Q Value')

plt.tight_layout()
plt.savefig('double_dqn_results.png')
plt.show()
