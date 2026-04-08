#  CartPole-v1 — Deep Q-Network (DQN) Agent

> Reinforcement Learning project 

---

##  Problem Statement

Train a **Deep Q-Network (DQN)** agent to balance a pole on a moving cart using the
OpenAI Gymnasium `CartPole-v1` environment. The agent learns purely through trial and
error — no human guidance, no labelled data.

---

##  Objective

- Implement a DQN agent with **Experience Replay** and **Target Network**
- Train the agent over 500 episodes using ε-greedy exploration
- Validate the trained policy over 20 greedy episodes
- Visualise training reward, loss, and validation performance

---


In CartPole:
- **Agent** = DQN neural network
- **Environment** = CartPole-v1 (a cart with a pole on top)
- **State** = [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions** = Push LEFT or Push RIGHT
- **Reward** = +1 for every timestep the pole stays upright
- **Goal** = Keep pole balanced as long as possible (max 500 steps)

---

## Algorithm: Deep Q-Network (DQN)

DQN was introduced by DeepMind in 2015 (Mnih et al., Nature).
It combines Q-Learning with deep neural networks.

### Core Idea — Bellman Equation
```
Q(s, a) = r + γ · max Q(s', a')
```
- `Q(s,a)` = expected future reward for taking action `a` in state `s`
- `r` = immediate reward
- `γ` = discount factor (0.99) — how much we value future rewards
- `s'` = next state

### Key Components

| Component | What it does |
|-----------|-------------|
| **Q-Network** | 3-layer neural net that predicts Q(s,a) for all actions |
| **Target Network** | A copy of Q-Network updated every 10 episodes — gives stable training targets |
| **Experience Replay** | Stores past (state, action, reward, next_state) in a buffer, trains on random batches |
| **ε-greedy Policy** | Starts exploring randomly (ε=1.0), gradually shifts to learned policy (ε=0.01) |

### Network Architecture
```
Input (4)  →  Linear(128)  →  ReLU  →  Linear(128)  →  ReLU  →  Output (2)
```

---


##  Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Episodes | 500 | Enough for convergence |
| Batch Size | 64 | Balance between speed and stability |
| Gamma (γ) | 0.99 | Value future rewards highly |
| Learning Rate | 0.001 | Standard Adam LR |
| Replay Buffer | 10,000 | Large enough for diverse samples |
| ε Start → End | 1.0 → 0.01 | Full exploration → exploitation |
| ε Decay | 0.995 | Gradual shift per episode |
| Target Update | Every 10 ep | Stable Q-targets |

---

## Base Paper

**"Human-level control through deep reinforcement learning"**  
Mnih et al., Nature 2015  
🔗 https://www.nature.com/articles/nature14236

---

##  Tech Stack

- **Python** 3.10+
- **PyTorch** — neural network & training
- **Gymnasium** — CartPole-v1 environment
- **NumPy** — numerical operations
- **Matplotlib** — result visualisation

---


