------

### 强化学习算法核心总结

**Code link**: https://github.com/wyoshmx/DRL

------

### 类别 1: Value-Based 方法 (DQN 家族)

核心是学习一个动作价值函数 $Q(s,a)$，然后通过选择具有最高 Q 值的动作（例如 $argmaxaQ(s,a)$）来决策，属于 **Off-Policy** 算法。

#### 1.1 DQN (Deep Q-Network)

**核心思想:** 将神经网络与 Q-Learning 结合，引入**经验回放 (Experience Replay)** 和**目标网络 (Target Network)** 以稳定训练。

- **经验回放**: 存储经历 $(s,a,r,s′,done)$，训练时随机采样，打破数据相关性。
- **目标网络**: 使用独立的目标网络计算 TD 目标 ytarget，提供稳定学习目标。

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
self.q_net = Net(n_states, n_hidden, n_actions)        # 训练网络
self.target_q_net = Net(n_states, n_hidden, n_actions)  # 目标网络
```

**损失函数与更新:**

- **TD目标**: $ytarget=r+γ⋅maxa′Qtarget(s′,a′;θtarget)⋅(1−done)$

- **损失函数**: $L(θ)=MSE[Q(s,a;θ),ytarget]$

- **更新过程**:

  ```python
  # 计算损失
  q_values = self.q_net(states).gather(1, actions)          # Q(s,a) [b,1]
  next_q_values = self.target_q_net(next_states)             # Q_target(s',.) [b,n_actions]
  max_next_q_values = next_q_values.max(1)[0].view(-1, 1)    # max_a' Q_target(s',a') [b,1]
  q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # y_target [b,1]
  loss = F.mse_loss(q_values, q_targets)
  # 更新在线网络
  self.optimizer.zero_grad()
  loss.backward()
  self.optimizer.step()
  # 硬更新目标网络 (也可用软更新)
  if self.count % target_update == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())
  ```

**动作选择 (ϵ-greedy):**

```python
def take_action(self, state):
    state = torch.FloatTensor(state).unsqueeze(0)  # [1, n_states]
    if np.random.rand() < self.epsilon:
        q_values = self.q_net(state)               # [1, n_actions]
        action = q_values.argmax().item()          # Greedy action
    else:
        action = np.random.randint(self.n_actions)  # Random action
    return action
```

#### 1.2 Double DQN

**核心思想:** 解耦动作选择与价值评估，缓解 Q 值高估问题。

- **动作选择**: $a∗=argmaxa′Q(s′,a′;θ)$(在线网络)
- **价值评估**: $Qtarget(s′,a∗;θtarget)$(目标网络)

**损失函数与更新:**

- **TD目标**: $ytargetDDQN=r+γ⋅Qtarget(s′,a∗;θtarget)$

- **代码修改**:

  ```python
  # 用在线网络选择动作
  max_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)  # a* [b,1]
  # 用目标网络评估
  next_q_values = self.target_q_net(next_states)                     # Q_target(s',.) [b,n_actions]
  max_next_q_values = next_q_values.gather(1, max_actions)           # Q_target(s',a*) [b,1]
  q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # y_target [b,1]
  ```

  其余与 DQN 相同。

#### 1.3 Dueling DQN

**核心思想:** 改进网络架构，将 Q 值分解为状态价值 V(s)和动作优势 $A(s,a): Q(s,a)=V(s)+A(s,a)$。

- V(s): 状态 s本身的价值。
- A(s,a): 动作 a相对于平均水平的优势。

**网络结构:**

```python
class DuelingNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc_val = nn.Linear(n_hiddens, 1)      # 价值流 V(s)
        self.fc_adv = nn.Linear(n_hiddens, n_actions) # 优势流 A(s,a)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        val = self.fc_val(x)                       # V(s) [b,1]
        adv = self.fc_adv(x)                       # A(s,a) [b,n_actions]
        # 聚合: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return val + adv - adv.mean(dim=1, keepdim=True)
```

**损失函数与更新:** 与 DQN 完全相同，仅网络结构不同。

------

### 类别 2: Actor-Critic (AC) 方法

同时学习策略函数 $π(a∣s)(Actor)$ 和状态价值函数 $V(s)(Critic)$。Critic 评估状态价值，并指导 Actor 更新。

#### 2.1 基本 Actor-Critic (A2C)

**核心思想:** 使用**优势函数 (Advantage)** $A(s,a)=Q(s,a)−V(s)$指导策略更新。用 TD 误差 δ近似 $A(s,a): δ=r+γV(s′)−V(s)≈A(s,a)$

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

- **Critic (`ValueNet`)**: 输入 s，输出标量 V(s)。

  ```python
  class ValueNet(nn.Module):
      def __init__(self, n_states, n_hiddens):
          super(ValueNet, self).__init__()
          self.fc1 = nn.Linear(n_states, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, 1)
      def forward(self, x):
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return x # [b,1]
  ```

**损失函数与更新:**

- **Critic Loss (Value Loss)**: 让 V(s)接近真实回报。

  - **TD目标**: $ytarget=r+γ⋅V(s′)⋅(1−done)$

  - **损失**: $Lcritic=MSE[V(s),ytarget]$

  - **代码**:

    ```python
    td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones) # [b,1]
    value_loss = F.mse_loss(self.critic(states), td_target.detach())
    ```

- **Actor Loss (Policy Loss)**: 增加优势动作的概率。

  - **损失**: $Lactor=−∑logπ(a∣s)⋅A(s,a)≈−∑logπ(a∣s)⋅δ$

  - **代码**:

    ```python
    log_probs = torch.log(self.actor(states).gather(1, actions)) # log π(a|s) [b,1]
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

处理**连续动作空间**的 Off-Policy AC 算法。

**核心思想:** **确定性策略**、**目标网络**、**经验回放**。

- **确定性策略**: Actor 直接输出确定性动作 a=μ(s)。
- **目标网络**: 使用目标网络计算稳定目标。
- **经验回放**: 存储转移样本，随机采样训练。

**网络结构:**

- **Actor (`PolicyNet`)**: 输入 s，输出确定性动作 a。

  ```python
  class PolicyNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions, action_bound):
          super(PolicyNet, self).__init__()
          self.action_bound = action_bound
          self.fc1 = nn.Linear(n_states, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, n_actions)
      def forward(self, x):
          x = F.relu(self.fc1(x))
          x = torch.tanh(self.fc2(x))          # [-1, 1]
          return x * self.action_bound         # 缩放到环境范围
  ```

- **Critic (`QValueNet`)**: 输入 s和 a，输出 Q(s,a)。

  ```python
  class QValueNet(nn.Module):
      def __init__(self, n_states, n_hiddens, n_actions):
          super(QValueNet, self).__init__()
          self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
          self.fc2 = nn.Linear(n_hiddens, n_hiddens)
          self.fc3 = nn.Linear(n_hiddens, 1)
      def forward(self, s, a):
          x = torch.cat([s, a], dim=1)        # 拼接状态和动作
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          return self.fc3(x)                   # Q(s,a) [b,1]
  ```

- **目标网络**: 创建 Actor 和 Critic 的目标网络 `target_actor`, `target_critic`，初始化与在线网络相同。

**损失函数与更新:**

- **Critic Loss**:

  - **TD目标**: $ytarget=r+γ⋅Qtarget(s′,μtarget(s′))⋅(1−done)$

  - **损失**: $Lcritic=MSE[Q(s,a),ytarget]$

  - **代码**:

    ```python
    next_actions = self.target_actor(next_states)                   # μ_target(s')
    next_q_values = self.target_critic(next_states, next_actions)    # Q_target(s', μ_target(s'))
    q_targets = rewards + self.gamma * next_q_values * (1 - dones)  # y_target
    q_values = self.critic(states, actions)                         # Q(s,a)
    critic_loss = F.mse_loss(q_values, q_targets)
    ```

- **Actor Loss**: 最大化 $Q(s,μ(s))$。

  - **损失**: $Lactor=−∑Q(s,μ(s))$

  - **代码**:

    ```python
    actor_actions = self.actor(states)              # μ(s)
    actor_loss = -torch.mean(self.critic(states, actor_actions)) # -E[Q(s, μ(s))]
    ```

- **软更新 (Soft Update)**:

  - **公式**: $θtarget←τθ+(1−τ)θtarget$

  - **代码**:

    ```python
    # 更新目标网络参数
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
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

一种非常流行且稳定的 Policy Gradient 算法。

**核心思想:** 通过**裁剪 (Clipping)** 限制策略更新的幅度，防止策略崩溃。

**网络结构:** 与 A2C 类似，包含 $Actor (π(a∣s))$ 和 $Critic (V(s))$网络。根据动作空间分为离散和连续版本。

- **离散动作**: Actor 输出离散动作概率分布 (Categorical)。
- **连续动作**: Actor 输出高斯分布的均值 μ和标准差 σ，Critic 输出 V(s)。

**损失函数与更新 (PPO-Clip):**

- **优势估计 (Advantage Estimation)**: 常使用**广义优势估计 (GAE)**。

  $AtGAE(γ,λ)=∑l=0∞(γλ)lδt+l$

  其中 $δt=rt+γV(st+1)−V(st)$是 TD 误差。

  ```python
  # 使用 GAE 计算优势函数
  advantages = []
  advantage = 0
  for delta in reversed(td_deltas): # 逆序处理 TD errors
      advantage = gamma * lmbda * advantage + delta
      advantages.insert(0, advantage)
  advantages = torch.tensor(advantages)
  ```

- **比率 (Ratio)**: $rt(θ)=πθold(at∣st)πθ(at∣st)$

  ```python
  old_log_probs = torch.log( old_policy_probabilities ).detach() # 旧策略对数概率
  new_log_probs = torch.log( new_policy_probabilities )           # 新策略对数概率
  ratio = torch.exp(new_log_probs - old_log_probs)                # r_t(θ) [b,1]
  ```

- **Clipped 目标函数**:

  $LCLIP(θ)=E^t[min(rt(θ)A^t,clip(rt(θ),1−ϵ,1+ϵ)A^t)]$

  ```python
  surr1 = ratio * advantages          # 未裁剪损失
  surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages # 裁剪损失
  actor_loss = -torch.min(surr1, surr2).mean() # 最大化目标 -> 最小化负损失
  ```

- **Critic Loss (Value Loss)**: $L^{\mathrm{VF}} = \mathrm{MSE}\left( V(s), V_{\mathrm{target}} \right)$

  ```python
  critic_loss = F.mse_loss( values, returns ) # returns 可以是 TD(λ) 回报
  ```

- **熵奖励 (Entropy Bonus)** (可选): $L^{\mathrm{ENT}} = \hat{\mathbb{E}}_t\left[ S[\pi_\theta](s_t) \right]$，鼓励探索。

  ```python
  entropy_loss = -entropy.mean() # 最大化熵 -> 最小化负熵
  ```

- **总损失**: $L_t^{\mathrm{PPO}} = L^{\mathrm{CLIP}}_t + c_1 L^{\mathrm{VF}}_t + c_2 L^{\mathrm{ENT}}_t$

  ```python
  loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss # c1=0.5, c2=0.01
  ```

**动作选择:**

- **离散**: 与 A2C 相同，从 Categorical 分布中采样。

  ```python
  probs = self.actor(state) # [1, n_actions]
  action_dist = Categorical(probs)
  action = action_dist.sample().item()
  ```

- **连续**: 从高斯分布中采样。

  ```python
  mu, std = self.actor(state)          # 均值, 标准差
  action_dist = Normal(mu, std)
  action = action_dist.sample().item()  # 采样连续动作
  ```

------

### 类别 5: Soft Actor-Critic (SAC)

一种基于**最大熵**原则的 Off-Policy 算法，兼具高性能、高样本效率和稳定性。

**核心思想:** 在标准最大化累积奖励的目标上，增加**最大化策略熵**的目标: $J(π)=∑t=0TE(st,at)∼ρπ[r(st,at)+αH(π(⋅∣st))]$

$H(π(⋅∣s))$是策略的熵，α是温度参数，平衡两者。

**网络结构:**

- **Actor (`PolicyNet`)**: 输入 s，输出动作概率分布 π(a∣s)(离散) 或高斯分布的参数 (连续)。
- **两个 Critic (`QValueNet`)**: 输入 s和 a，输出 Q(s,a)。使用两个 Q 网络并取最小值以克服 Q 值过高估计。
- **两个 Target Critic 网络**: 用于计算稳定目标。
- **可调的温度参数 α**.

**损失函数与更新 (以连续动作为例):**

1. **Critic Loss**:

   - **TD目标**: $y=r+γ(minj=1,2Qtarget,j(s′,a~′)−αlogπ(a~′∣s′)),a~′∼π(⋅∣s′)$

   - **损失**: $LQi=E(s,a,r,s′)∼D[(Qi(s,a)−y)2],i=1,2$

   - **代码**:

     ```python
     # 计算目标 Q 值
     with torch.no_grad():
         next_actions, next_log_probs = self.actor.sample(next_states)   # 从策略中采样下一动作并计算其对数概率
         next_q1 = self.target_critic1(next_states, next_actions)
         next_q2 = self.target_critic2(next_states, next_actions)
         next_q_min = torch.min(next_q1, next_q2)
         next_value = next_q_min - self.alpha * next_log_probs           # 包含熵奖励的目标值
         q_target = rewards + self.gamma * (1 - dones) * next_value
     # 计算 Critic 损失
     q1 = self.critic1(states, actions)
     q2 = self.critic2(states, actions)
     critic1_loss = F.mse_loss(q1, q_target)
     critic2_loss = F.mse_loss(q2, q_target)
     ```

2. **Actor Loss**:

   - **目标**: 最大化期望 Q 值和策略熵。

   - **损失**: $Lπ=Es∼D[Ea∼π[αlogπ(a∣s)−minj=1,2Qj(s,a)]]$

   - **代码**:

     ```python
     new_actions, new_log_probs = self.actor.sample(states) # 重参数化采样
     q1_new = self.critic1(states, new_actions)
     q2_new = self.critic2(states, new_actions)
     q_new_min = torch.min(q1_new, q2_new)
     actor_loss = (self.alpha * new_log_probs - q_new_min).mean() # 最小化该损失
     ```

3. **温度 α的损失** (可选, 自动调节):

   - **目标**: 使策略熵接近目标熵 Hˉ。
   - **损失**: $L(α)=−αEa∼π[logπ(a∣s)+Hˉ]$

   ```python
   alpha_loss = - (self.log_alpha * (new_log_probs + target_entropy).detach()).mean()
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

| 算法            | 策略类型  | 主要特点                       | 更新方式   | 适用场景              |
| --------------- | --------- | ------------------------------ | ---------- | --------------------- |
| **DQN**         | 离散      | 经验回放，目标网络，值函数近似 | Off-Policy | 离散动作空间 (如游戏) |
| **Double DQN**  | 离散      | 解决 DQN 的 Q 值高估问题       | Off-Policy | 同 DQN，性能更稳定    |
| **Dueling DQN** | 离散      | 网络结构分解为 V 和 A          | Off-Policy | 同 DQN，学习更高效    |
| **A2C**         | 离散/连续 | 基础 Actor-Critic，优势函数    | On-Policy  | 简单 AC 任务          |
| **DDPG**        | 连续      | 确定性策略，目标网络，经验回放 | Off-Policy | 连续动作控制          |
| **PPO**         | 离散/连续 | 裁剪目标函数，稳定             | On-Policy  | 广泛适用，非常流行    |
| **SAC**         | 离散/连续 | 最大熵原则，稳定高效           | Off-Policy | 连续控制，需要探索    |