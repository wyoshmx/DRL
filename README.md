### 强化学习算法总结

------

### 类别 1: Value-Based 方法 (DQN 家族)

核心是学习一个动作价值函数 $Q(s,a)$，然后通过选择具有最高 Q 值的动作（例如 $arg max_aQ(s,a)$）来决策，属于 **Off-Policy** 算法

#### 1.1 DQN (Deep Q-Network)

**核心思想:** 将神经网络与 $Q-Learning$ 结合，引入**经验回放 (Experience Replay)** 和**目标网络 (Target Network)** 以稳定训练

- **经验回放**: 存储经历 $(s,a,r,s^′,done)$，训练时随机采样，打破数据相关性
- **目标网络**: 使用独立的目标网络计算 TD 目标 $y_{target}$，提供稳定学习目标

**网络结构:**

```python
class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)  # [b, n_states] -> [b, n_hidden]
        self.fc2 = nn.Linear(n_hidden, n_actions) # [b, n_hidden] -> [b, n_actions]
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
# 实例化
self.q_net = Net(n_states, n_hidden, n_actions)         # 训练网络
self.target_q_net = Net(n_states, n_hidden, n_actions)  # 目标网络
```

**损失函数与更新:**

- **目标**：让当前 Q 网络的预测值尽可能接近目标 Q 值

- **方法**： 最小化**时序差分误差 (Temporal Difference Error)** 的**均方误差 (MSE)**

- **目标Q值**: $y_{target}=r+γ⋅max_{a′}Q_{target}(s^′,a^′;θ_{target})⋅(1−done)$

- **损失函数**: $L(θ)=MSE[Q(s,a;θ),y_{target}]$

- **更新过程**:

  * **步骤 1**: 从经验回放池中随机采样一个批量的转移数据 `(s, a, r, s', done)`

  * **步骤 2**: 使用**目标网络**计算所有 `s'` 的 `max Q` 值，得到 `y_target`

  ```python
  # 输入当前状态，得到采取各运动得到的概率
  # 根据actions索引在训练网络的输出的第1维度上获取对应索引的Q值
  q_values = self.q_net(states).gather(1, actions)  # [b,1]
  
  # # # 使用目标网络选择下一状态的最优动作，再使用目标网络评估该动作的价值
  max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
  
  # 计算目标Q值：即时奖励+折扣因子*下个时刻的最大回报
  q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
  
  # 目标网络和训练网络之间的均方误差损失
  dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
  self.optimizer.zero_grad() # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
  dqn_loss.backward() # 反向传播参数更新
  self.optimizer.step() # 对训练网络更新
  ```

  * **步骤 3 (硬更新)**: 定期（如每200步）将目标网络的参数直接更新为当前网络的参数

  ```python
    if self.count % self.target_update == 0:
        self.target_q_net.load_state_dict(self.q_net.state_dict())
  ```

**动作选择 (ϵ-greedy):**

```python
def take_action(self, state):
    # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
    state = torch.Tensor(state[np.newaxis, :])
    # 如果小于该值就取最大的值对应的索引
    if np.random.random() < self.epsilon:  # 0-1
        # 前向传播获取该状态对应的动作的reward
        actions_value = self.q_net(state)
        action = actions_value.argmax().item()   # 获取reward最大值对应的动作索引
    else:
        action = np.random.randint(self.n_actions)
    return action
```

#### 1.2 Double DQN

**核心思想:**标准 DQN 在计算 $y_{target}$ 时，**选择和评估**下一个状态 $s'$ 的最优动作都依赖于目标网络 $Q_{target}$。这会导致 Q 值被**显著高估 (overestimation)**，因为 `max` 操作会偏向于带有正误差的估计值，Double DQN 解决了这个问题。它将**动作选择**和**价值评估**解耦：

- **用当前网络选择动作**: $a^∗=argmax_{a^′}Q(s^′,a^′;θ)$(在线网络)
- **用目标网络评估价值**: $Q_{target}(s^′,a^∗;θ_{target})$(目标网络)

**损失函数与更新:**

- **目标和方法**: 与DQN完全相同

- **公式**:

  -  $L(θ) = MSE( Q(s, a; θ), y_{target}^{DDQN} )$

  -  $y_{target}^{DDQN}=r+γ⋅Q_{target}(s^′,a^∗;θ_{target})$  $where$  $a^* = argmax_{a'} Q(s', a'; θ)$

- **代码修改**:

  ```python
  # 输入当前状态，得到采取各运动得到的概率
  # 根据actions索引在训练网络的输出的第1维度上获取对应索引的Q值
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
  ```

- **动作选择**：

  ```python
    def take_action(self, state):  # 传入某个人的状态
       # numpy[n_states]-->[1,n_states]-->tensor
       state = torch.Tensor(state[np.newaxis, :])
       # 获取每个人的各动作对应的概率[1,n_states]-->[1,n_actions]
       probs = self.policy_net(state)
       action_dist =torch.distributions.Categorical(probs) #创建以probs为标准类型的数据分布
       action = action_dist.sample() # 以该概率分布随机抽样
       action = action.item()
       return action
  ```

#### 1.3 Dueling DQN

**核心思想:** 改进网络架构，将 Q 值分解为状态价值 V(s)和动作优势 $A(s,a): Q(s,a)=V(s)+A(s,a)$。

- $V(s)$: 状态 s本身的价值
- $A(s,a)$: 动作 a相对于平均水平的优势

**目标、方法和公式**：与DoubleDQN完全相同

**网络结构:**

```python
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
```

**损失函数与更新:** 与 DoubleDQN 完全相同，仅网络结构不同。

------

### 类别 2: Actor-Critic (AC) 方法

同时学习策略函数 $π(a∣s)(Actor) $和状态价值函数 $V(s)(Critic)$。Critic 评估状态价值，并指导 Actor 更新。

#### 2.1 基本 Actor-Critic (A2C)

**核心思想:** 使用**优势函数 (Advantage)** $A(s,a)=Q(s,a)−V(s)$指导策略更新，优势函数衡量了在状态 `s` 下采取动作 `a` 比平均情况好多少。用 TD 误差 δ近似 $A(s,a)$: $δ=r+γV(s^′)−V(s)≈A(s,a)$

**网络结构:**

- **Actor (`PolicyNet`)**: 输入 s，输出动作概率分布 $π(⋅∣s)$。

  ```python
  class PolicyNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions):
          super(PolicyNet, self).__init__()
          self.fc1 = nn.Linear(n_states, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, n_actions)
      def forward(self, x):
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return F.softmax(x, dim=1) # [b, n_actions]
  ```

- **Critic (`ValueNet`)**: 输入 s，输出标量 $V(s)$。

  ```python
  class ValueNet(nn.Mod极地):
      def __init__(self, n_states, n_hiddens):
          super(ValueNet, self).__init__()
          self.fc1 = nn.Linear(n_states, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, 1)
      def forward(self, x):
          x = F.relu(self极地fc1(x))
          x = self.fc2(x)
          return x # [b,1]
  ```

**损失函数与更新:**

- **Critic Loss (Value Loss)**: 让 $V(s)$接近真实回报。

  - **目标**: 让 Critic 的预测值 `V(s)` 尽可能接近真实的回报

  - **方法**: 最小化预测值和目标值之间的均方误差 (MSE)

  - **TD目标**: $y_{target}=r+γ⋅V(s^′)⋅(1−done)$

  - **损失**: $L_{critic}=MSE[V(s),y_{target}]$

  - **代码**:

    ```python
    td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
    critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
    ```

- **Actor Loss (Policy Loss)**: 增加优势动作的概率

  - **目标**: 增加优势高的动作的概率，减少优势低的动作的概率

  - **方法**: 梯度上升（通过最小化负的期望回报来实现）

  - **损失**: $L_{actor}=−∑logπ(a∣s)⋅A(s,a)≈−∑logπ(a∣s)⋅δ$

  - **代码**:

    ```python
    log_probs = torch.log(self.actor(states).gather(1, actions)) # log π(a|s) [极地,1]
    td_delta = td_target - self.critic(states)                    # δ ≈ A(s,a) [b,1]
    actor_loss = torch.mean(-log_probs * td_delta.detach())       # 注意 detach()
    ```

- **总更新**:

  ```python
  self.actor_optimizer.zero_grad()
  self.critic_optimizer.zero_grad()
  actor_loss.backward()
  value_loss.backward()
  self.actor_optimizer.step()
  self.critic_optimizer.step()
  ```

**动作选择 (采样):**

```python
def take_action(self, state):
    state = torch.tensor(state, dtype=torch.float).view(1, -1) # [1, n_states]
    probs = self.actor(state)                                   # π(.|s) [1, n_actions]
    action_dist = torch.distributions.Categorical(probs)
    action = action_dist.sample().item()                        # 采样动作
    return action
```

------

### 类别 3: Deep Deterministic Policy Gradient (DDPG)

处理**连续动作空间**的 $Off-Policy$ $AC$ 算法。

**核心思想:** **确定性策略**、**目标网络**、**经验回放**。

- **确定性策略**: Actor 直接输出确定性动作 $a=μ(s)$
- **目标网络**: 使用目标网络计算稳定目标Q值
- **经验回放**: 使用经验回放池（Replay Buffer）来打破数据间的相关性

**网络结构:**

- **Actor (`PolicyNet`)**: 输入 s，输出确定的连续动作 a。

  ```python
  class PolicyNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions, action_bound):
          super(PolicyNet, self).__init__()
          self.action_bound = action_bound # 环境可以接受的动作最大值
          self.fc1 = nn.Linear(n_states, n_hiddens) # 只包含一个隐含层
          self.fc2 = nn.Linear(n_hiddens, n_actions)
  
      # 前向传播
      def forward(self, x):
          x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
          x = F.relu(x)
          x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
          x = torch.tanh(x)  # 将数值调整到 [-1,1]
          x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
          return x (确定的动作值)
  ```

- **Critic (`QValueNet`)**: 输入 $s$和 $a$，输出 $Q(s,a)$。

  ```python
  class QValueNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions):
          super(QValueNet, self).__init__()
          self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, n_hiddens)
          self.fc3 = nn.Linear(n_hiddens, 1)
  
      # 前向传播
      def forward(self, x, a):
          cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]拼接状态和动作
          x = self.fc1(cat)  # -->[b, n_hiddens]
          x = F.relu(x)
          x = self.fc2(x)  # -->[b, n_hiddens]
          x = F.relu(x)
          x = self.fc3(x)  # -->[b, 1]
          return x
  ```

- **目标网络**: 创建 Actor 和 Critic 的目标网络 `target_actor`, `target_critic`，初始化与在线网络相同。

  ```python
  # 策略网络--训练
  self.actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
  # 价值网络--训练
  self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
  # 策略网络--目标
  self.target_actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
  # 价值网络--目标
  self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
  # 初始化价值网络的参数，两个价值网络的参数相同
  self.target_critic.load_state_dict(self.critic.state_dict())
  # 初始化策略网络的参数，两个策略网络的参数相同
  self.target_actor.load_state_dict(self.actor.state_dict())
  ```

**损失函数与更新:**

- **Critic Loss**:

  - **目标**: 让 Critic 的预测值 `Q(s, a)` 接近目标 Q 值

  - **TD目标**: $y_{target}=r+γ⋅Q_{target}(s^′,μ_{target}(s^′))⋅(1−done)$

  - **损失**: $L_{critic}=MSE[Q(s,a),y_{target}]$

  - **代码实现**:

    ```python
    next_actions = self.target_actor(next_states) # μ_target(s')
    next_q_values = self.target_critic(next_states, next_actions) # Q_target(s', a')
    q_targets = rewards + self.gamma * next_q_values * (1-dones) # y_target
    q_values = self.critic(states, actions) # Q(s, a)
    critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
    ```

- **Actor Loss**: 最大化 $Q(s,μ(s))$。

  - **目标**: 最大化 Critic 网络对当前状态和 Actor 所选动作的评分 `Q(s, μ(s))`

  - **方法**: 梯度上升（通过最小化负的 Q 值来实现）

  - **损失**: $L_{actor}=−∑Q(s,μ(s))$

  - **代码**:

    ```python
    actor_q_values = self.actor(states) # μ(s)
    score = self.critic(states, actor_q_values) # Q(s, μ(s))
    actor_loss = -torch.mean(score) # 最小化负的 Q 值
    ```

- **软更新 (Soft Update)**:

  - **目的**: 缓慢更新目标网络，提高训练稳定性

  - **公式**: $θ_{target}←τθ+(1−τ)θ_{target}$

  - **代码**:

    ```python
    self.soft_update(self.actor, self.target_actor) # 软更新策略网络的参数  
    self.soft_update(self.critic, self.target_critic) # 软更新价值网络的参数
    ```

    ```python
    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)
    ```

**动作选择 (探索):**

```python
def take_action(self, state):
    state = torch.tensor(state, dtype=torch.float).view(1, -1)
    action = self.actor(state).item()                   # 确定性动作
    # 添加探索噪声 (e.g., Gaussian)
    action += self.sigma * np.random.randn(self.n_actions)
    return np.clip(action, -self.action_bound, self.action_bound)
```

------

### 类别 4: Proximal Policy Optimization (PPO)

**核心思想:** 通过**裁剪 (Clipping)** 限制策略更新的幅度，避免因为单次更新过大而导致策略性能崩溃。 Policy Gradient 算法

**网络结构:** 与 A2C 类似，包含 $Actor (π(a∣s)) $和 $Critic (V(s))$ 网络。根据动作空间分为离散和连续版本。

- **离散动作**: Actor 输出离散动作概率分布 (Categorical)。
- **连续动作**: Actor 输出高斯分布的均值 $μ$ 和标准差 $σ$，Critic 输出 $V(s)$。

**损失函数与更新 (PPO-Clip):**

- **优势估计 (Advantage Estimation)**: 常使用**广义优势估计 (GAE)**。

  $A_t^{GAE(γ,λ)}=∑_{l=0}^∞(γλ)^lδ_{t+l}$

  其中 $δ_t=r_t+γV(s_{t+1})−V(s_t)$是 TD 误差

  ```python
  next_q_target = self.critic(next_states)
  td_target = rewards + self.gamma * next_q_target * (1 - dones)
  td_value = self.critic(states)
  td_delta = td_target - td_value
  td_delta = td_delta.cpu().detach().numpy()
  advantage = 0  # 优势函数初始化
  advantage_list = []
  
  # 计算优势函数
  for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
      advantage = self.gamma * self.lmbda * advantage + delta # 优势函数GAE的公式
  ```

- **比率 (Ratio)**: $ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $

  ```python
  old_log_probs=torch.log(self.actor(states).gather(1, actions)).detach() #旧策略对数概率
  log_probs = torch.log(self.actor(states).gather(1, actions))           # 新策略对数概率
  ratio = torch.exp(log_probs - old_log_probs)            # r_t(θ) [b,1]
  ```

- **Clipped 目标函数**: $ L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] $

  ```python
  surr1 = ratio * advantage (近端策略优化裁剪目标函数公式的左侧项)
  surr2 = torch.clamp(ratio, 1 - ε, 1 + ε) * advantage (右侧项)
  actor_loss = torch.mean(-torch.min(surr1, surr2)) (策略网络的损失函数)
  ```

  **解释**: 如果 advantage 是正的（好动作），希望增加该动作的概率，但限制比率 `ratio` 最大不超过 `1+ε`。如果 advantage 是负的（坏动作），希望减少该动作的概率，但限制比率 `ratio ` 最小不低于 `1-ε`

- **Critic Loss (Value Loss)**: $L^{VF}=MSE(V(s),V_{target})$

  - 与 AC 中的 Critic 损失类似，最小化价值函数的预测值和目标值之间的差异

  ```python
  critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
  ```

- **熵奖励 (Entropy Bonus)** (可选): $ L^{\mathrm{ENT}} = \hat{\mathbb{E}}_t \left[ \mathcal{H}(\pi_\theta(\cdot|s_t)) \right] $，鼓励探索。

  ```python
  entropy_loss = -entropy.mean() # 最大化熵 -> 最小化极地负熵
  ```

- **总损失**: $L_t^{PPO}=L_t ^{CLIP}+c_1L_t ^{VF}+c_2L_t ^{ENT}$

  ```python
  loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss # c1=0.5, c2=0.01
  ```

**动作选择 (Continuous / Discrete):**

- **离散**: 与 A2C 相同，从 Categorical 分布中采样

  ```python
  class PolicyNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions):
          super(PolicyNet, self).__init__()
          self.fc1 = nn.Linear(n_states, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, n_actions)
  
      def forward(self, x):
          x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
          x = F.relu(x)
          x = self.fc2(x)  # [b, n_actions]
          x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
          return x(动作概率)
      
  def take_action(self, state):
      # 维度变换 [n_state]-->tensor[1,n_states]
      state = torch.tensor(state[np.newaxis, :]).to(self.device)
      # 当前状态下，每个动作的概率分布 [1,n_states]
      probs = self.actor(state)
      action_list = torch.distributions.Categorical(probs)# 创建以probs为标准的概率分布
      action = action_list.sample().item() # 随机选择动作
      return action
  ```

- **连续**: 从高斯分布中采样

  ```python
  class PolicyNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions):
          super(PolicyNet, self).__init__()
          self.fc1 = nn.Linear(n_states, n_hiddens)
          self.fc_mu = nn.Linear(n_hiddens, n_actions)
          self.fc_std = nn.Linear(n_hiddens, n_actions)
  
      # 输出连续动作的高斯分布的均值和标准差
      def forward(self, x):
          x = self.fc1(x)
          x = F.relu(x)
          mu = self.fc_mu(x)
          mu = 2 * torch.tanh(mu)  # 值域 [-2,2]
          std = self.fc_std(x)
          std = F.softplus(std)  # 值域 小于0的部分逼近0，大于0的部分几乎不变
          return mu(均值), std(方差) 
  
  def take_action(self, state):  # 输入当前时刻的状态
      # [n_states]-->[1,n_states]-->tensor
      state = torch.tensor(state[np.newaxis, :]).to(self.device)
      mu, std = self.actor(state) # 预测当前状态的动作，输出动作概率的高斯分布
      action_dict = torch.distributions.Normal(mu, std)  # 构造高斯分布
      action = action_dict.sample().item() # 随机选择动作
      return [action]  # 返回动作值
  ```

------

### 类别 5: Soft Actor-Critic (SAC)

一种基于**最大熵**原则的 $Off-Policy$ 算法，兼具高性能、高样本效率和稳定性

**核心思想:** 在标准最大化累积奖励的目标上，增加**最大化策略熵**的目标。这会让策略在追求高回报的同时，保持随机性，从而鼓励探索

**网络结构:**: $ J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left[ r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right] $​

$H(π(⋅∣s))$是策略的熵，$α$是温度参数，平衡两者

**网络结构:**

- **Actor (`PolicyNet`)**: 输入 s，输出动作概率分布 $π(a∣s)$(离散) 或高斯分布的参数 (连续)。
- **两个 Critic (`QValueNet`)**: 输入 s和 a，输出 $Q(s,a)$。使用两个 Q 网络并取最小值以克服 Q 值过高估计。
- **两个 Target Critic 网络**: 用于计算稳定目标Q值。
- **可调的温度参数 α**: 平衡奖励和熵的重要性

**损失函数与更新 (以连续动作为例):**

1. **Critic Loss**:

   - **目标**: 让 Q 网络的预测值接近目标 Q 值

   - **TD目标**: $ y = r + \gamma \left( \min_{j=1,2} Q_{\text{target},j}(s', \tilde{a}') - \alpha \log \pi(\tilde{a}'|s') \right), \quad \tilde{a}' \sim \pi(\cdot|s') $

   - **损失**: $ \mathcal{L}_{Q_i} = \mathbb{E}_{(s,a,r,s') \sim D} \left[ (Q_i(s,a) - y)^2 \right], \quad i=1,2 $

   - **代码**:

     ```python
     next_probs = self.actor(next_states) # π(·|s')
     next_log_probs = torch.log(next_probs + 1e-8) # log(π(·|s'))
     entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdims=True) # -Σπ*log(π) = 熵
     q1_value = self.target_critic_1(next_states) # Q_target1(s', a)
     q2_value = self.target_critic_2(next_states) # Q_target2(s', a)
     min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdims=True) # E[min Q]
     next_value = min_qvalue + self.log_alpha.exp() * entropy # 目标价值：E[min Q] + α*H(π)
     td_target = rewards + self.gamma * next_value * (1-dones) # y_target
     ```

2. **Actor Loss**:

   - **目标**: 最大化期望 Q 值和策略熵。

   - **损失**: $ \mathcal{L}_{\text{actor}} = \sum_{a} \pi(a|s) \cdot \left( \alpha \log(\pi(a|s)) - \min_{j} Q_j(s, a) \right) = \alpha \mathcal{H}(\pi) - \mathbb{E}_{a \sim \pi} \left[ \min_{j} Q_j(s, a) \right] $

   - **代码**:

     ```python
     probs = self.actor(states)
     log_probs = torch.log(probs + 1e-8)
     entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True) # H(π)
     q1_value = self.critic_1(states)
     q2_value = self.critic_2(states)
     min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True) # E[min Q]
     actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue) # 最小化 -(α*H(π) + E[min Q])
     ```

3. **温度 α的损失** (可选, 自动调节):

   - **目标**: 自动调整 `α`，使策略的熵接近一个目标熵值 $\bar{H}$（如 `-dim(A)`）
   - **损失**: $ \mathcal{L}(\alpha) = -\alpha \mathbb{E}_{a \sim \pi} \left[ \log \pi(a|s) + \bar{H} \right] $

   ```python
   alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
   ```

4. **软更新目标网络**:

   ```python
   self.soft_update(self.critic1, self.target_critic1)
   self.soft_update(self.critic2, self.target_critic2)
   ```

**动作选择:**

```python
def take_action(self, state):
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    if self.discrete:
        probs = self.actor(state)
        action_dist = Categorical(probs)
        action = action_dist.sample().item()
    else:
        mean, log_std = self.actor(state) # SAC 常输出均值和对数标准差
        std = log_std.exp()
        action_dist = Normal(mean, std)
        action = action_dist.sample().item()
    return action
```

------

### 总结对比表

|      算法       | 策略类型  |            主要特点            | 更新方式   | 适用场景              |
| :-------------: | :-------: | :----------------------------: | ---------- | --------------------- |
|     **DQN**     |   离散    | 经验回放，目标网络，值函数近似 | Off-Policy | 离散动作空间 (如游戏) |
| **Double DQN**  |   离散    |    解决 DQN 的 Q 值高估问题    | Off-Policy | 同 DQN，性能更稳定    |
| **Dueling DQN** |   离散    |     网络结构分解为 V 和 A      | Off-Policy | 同 DQN，学习更高效    |
|     **A2C**     | 离散/连续 |  基础 Actor-Critic，优势函数   | On-Policy  | 简单 AC 任务          |
|    **DDPG**     |   连续    | 确定性策略，目标网络，经验回放 | Off-Policy | 连续动作控制          |
|     **PPO**     | 离散/连续 |       裁剪目标函数，稳定       | On-Policy  | 广泛适用，非常流行    |
|     **SAC**     | 离散/连续 |      最大熵原则，稳定高效      | Off-Policy | 连续控制，需要探索    |
