import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
#  1. Hyperparameters
# ─────────────────────────────────────────
EPISODES        = 500
BATCH_SIZE      = 64
GAMMA           = 0.99          # discount factor
LR              = 1e-3          # learning rate
MEMORY_SIZE     = 10_000        # replay buffer capacity
EPS_START       = 1.0           # initial exploration
EPS_END         = 0.01          # minimum exploration
EPS_DECAY       = 0.995         # per-episode decay
TARGET_UPDATE   = 10            # sync target net every N episodes
SOLVE_SCORE     = 475           # CartPole-v1 is "solved" at 475

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────
#  2. Q-Network Architecture
# ─────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────
#  3. Replay Memory
# ─────────────────────────────────────────
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.buffer)

# ─────────────────────────────────────────
#  4. DQN Agent
# ─────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.epsilon    = EPS_START

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory    = ReplayMemory(MEMORY_SIZE)
        self.loss_fn   = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.policy_net(state_t).argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ─────────────────────────────────────────
#  5. Training Loop
# ─────────────────────────────────────────
def train():
    env   = gym.make("CartPole-v1")
    state_dim  = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []
    avg_rewards     = []
    losses          = []
    solved_episode  = None

    print(f"\n{'='*55}")
    print(f"  DQN CartPole Training  |  Episodes: {EPISODES}")
    print(f"{'='*55}")

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        ep_losses    = []

        while True:
            action     = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated

            agent.memory.push(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)

            state        = next_state
            total_reward += reward
            if done:
                break

        agent.update_epsilon()
        if ep % TARGET_UPDATE == 0:
            agent.sync_target()

        rewards_history.append(total_reward)
        avg100 = np.mean(rewards_history[-100:])
        avg_rewards.append(avg100)
        losses.append(np.mean(ep_losses) if ep_losses else 0)

        if ep % 50 == 0 or total_reward >= SOLVE_SCORE:
            print(f"  Ep {ep:4d} | Reward: {total_reward:6.1f} | "
                  f"Avg(100): {avg100:6.1f} | ε: {agent.epsilon:.3f}")

        if avg100 >= SOLVE_SCORE and solved_episode is None:
            solved_episode = ep
            print(f"\n  ✅  SOLVED at episode {ep}!  Avg(100) = {avg100:.1f}\n")

    env.close()
    torch.save(agent.policy_net.state_dict(), "dqn_cartpole_model.pth")
    print("\n  Model saved → dqn_cartpole_model.pth")
    return rewards_history, avg_rewards, losses, solved_episode

# ─────────────────────────────────────────
#  6. Validation
# ─────────────────────────────────────────
def validate(n_episodes=20):
    env   = gym.make("CartPole-v1")
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load("dqn_cartpole_model.pth", map_location=device))
    model.eval()

    val_rewards = []
    print(f"\n{'='*55}")
    print(f"  Validation  |  {n_episodes} greedy episodes")
    print(f"{'='*55}")

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        total    = 0
        while True:
            with torch.no_grad():
                action = model(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        val_rewards.append(total)
        print(f"  Val Ep {ep:2d} | Reward: {total:.1f}")

    env.close()
    print(f"\n  Mean Validation Reward: {np.mean(val_rewards):.2f}")
    print(f"  Min: {np.min(val_rewards):.1f}  |  Max: {np.max(val_rewards):.1f}")
    return val_rewards

# ─────────────────────────────────────────
#  7. Plot Results
# ─────────────────────────────────────────
def plot_results(rewards_history, avg_rewards, losses, val_rewards, solved_ep):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#aaaacc")
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.title.set_color("#e0e0ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    eps = range(1, len(rewards_history) + 1)

    # Panel 1: Episode reward + 100-ep average
    axes[0].plot(eps, rewards_history, color="#3a86ff", alpha=0.35, linewidth=0.8, label="Episode Reward")
    axes[0].plot(eps, avg_rewards,     color="#ff006e", linewidth=2.0, label="Avg (100 ep)")
    axes[0].axhline(y=SOLVE_SCORE, color="#8338ec", linestyle="--", linewidth=1.5, label=f"Solved ({SOLVE_SCORE})")
    if solved_ep:
        axes[0].axvline(x=solved_ep, color="#ffbe0b", linestyle=":", linewidth=1.5, label=f"Solved @ ep {solved_ep}")
    axes[0].set_title("Training Reward", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ccccee")
    axes[0].set_ylim(0, 520)

    # Panel 2: Loss curve
    axes[1].plot(eps, losses, color="#06d6a0", linewidth=0.9, alpha=0.8)
    axes[1].set_title("Training Loss (MSE)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")

    # Panel 3: Validation bar chart
    val_eps = range(1, len(val_rewards) + 1)
    colors  = ["#ff006e" if r < SOLVE_SCORE else "#06d6a0" for r in val_rewards]
    axes[2].bar(val_eps, val_rewards, color=colors, edgecolor="#0f0f1a", linewidth=0.5)
    axes[2].axhline(y=np.mean(val_rewards), color="#ffbe0b", linestyle="--", linewidth=2, label=f"Mean={np.mean(val_rewards):.1f}")
    axes[2].axhline(y=SOLVE_SCORE, color="#8338ec", linestyle="--", linewidth=1.5, label=f"Solve threshold ({SOLVE_SCORE})")
    good  = mpatches.Patch(color="#06d6a0", label="≥ 475 (solved)")
    bad   = mpatches.Patch(color="#ff006e", label="< 475")
    axes[2].legend(handles=[good, bad], fontsize=8, facecolor="#1a1a2e", labelcolor="#ccccee")
    axes[2].set_title("Validation Rewards", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Validation Episode")
    axes[2].set_ylabel("Total Reward")
    axes[2].set_ylim(0, 520)

    plt.suptitle("DQN Agent — CartPole-v1", fontsize=16, fontweight="bold", color="#e0e0ff", y=1.02)
    plt.tight_layout()
    plt.savefig("dqn_results.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print("\n  Plot saved → dqn_results.png")
    plt.show()

# ─────────────────────────────────────────
#  8. Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    rewards_history, avg_rewards, losses, solved_ep = train()
    val_rewards = validate()
    plot_results(rewards_history, avg_rewards, losses, val_rewards, solved_ep)
