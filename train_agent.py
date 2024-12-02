import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from thermal_env import MultiZoneThermalControlEnv
import numpy as np

from IPython.core.debugger import set_trace

# 1. Define the PPO agent
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dims, hidden_dim=128):
        super(PPOAgent, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sum(action_dims))
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_logits = self.actor(state)
        state_value = self.critic(state)
        return action_logits, state_value

# 2. Define PPO parameters
class PPO:
    def __init__(self, state_dim, action_dims, hidden_dim=128, lr=3e-4, gamma=0.99, eps_clip=0.2, K=2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K

        self.action_dims = action_dims
        self.policy = PPOAgent(state_dim, action_dims, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOAgent(state_dim, action_dims, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state)
        action_logits, _ = self.policy_old(state)

        # Separate logits for each action dimension
        action_logits_split = torch.split(action_logits, tuple(self.action_dims), dim=-1)
        actions = []
        log_probs = []

        for logits in action_logits_split:
            probs = torch.softmax(logits, dim=-1)  # Normalize logits to probabilities
            dist = Categorical(probs=probs)
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))

        # Save memory
        memory.states.append(state)
        memory.actions.append(torch.stack(actions))
        memory.log_probs.append(torch.stack(log_probs))

        return [a.item() for a in actions]

    
    def update(self, memory):
        states = torch.stack(memory.states)
        actions = torch.stack(memory.actions).squeeze(-1)
        log_probs_old = torch.stack(memory.log_probs).squeeze(-1)
        rewards = memory.rewards
        dones = memory.dones

        # Calculate discounted returns
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(self.K):
            action_logits, state_values = self.policy(states)

            # Debugging: Check for NaNs in logits
            # set_trace()
            if torch.isnan(action_logits).any():
                raise ValueError("Action logits contain NaNs!")

            # Split logits for multi-dimensional action spaces
            log_probs = []
            for i, logits in enumerate(torch.split(action_logits, tuple(self.action_dims), dim=-1)):
                # Clamp logits for numerical stability
                logits = torch.clamp(logits, min=-20, max=20)
                dist = Categorical(probs=torch.softmax(logits, dim=-1))
                log_probs.append(dist.log_prob(actions[:, i]))

            log_probs = torch.stack(log_probs, dim=-1).sum(dim=-1)  # Combine log probabilities
            log_probs_old = log_probs_old.sum(dim=-1)

            # Compute ratios for importance sampling
            ratios = torch.exp(log_probs - log_probs_old.detach())

            # Compute surrogate loss
            advantages = returns - state_values.detach().squeeze(-1)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()

            # Critic loss
            loss_critic = self.mse_loss(state_values.squeeze(-1), returns)

            # Total loss
            loss = loss_actor + 0.5 * loss_critic

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.step()

        # Update old policy weights
        self.policy_old.load_state_dict(self.policy.state_dict())



# 3. Define memory for PPO
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

# 4. Train the PPO agent
def train_ppo():
    # Initialize environment
    env = MultiZoneThermalControlEnv()
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec

    # Hyperparameters
    total_episodes = 1000
    max_timesteps = 200
    update_timestep = 2000
    memory = Memory()

    # PPO Agent
    ppo = PPO(state_dim, action_dims)

    timestep = 0
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            timestep += 1

            # Select action
            action = ppo.select_action(state, memory)
            next_state, reward, done, _ = env.step(action)  # Pass the list of actions
            memory.rewards.append(reward)
            memory.dones.append(done)

            episode_reward += reward
            state = next_state

            if done:
                break

            # Update PPO after every 'update_timestep' steps
            if timestep % update_timestep == 0:
                print("Update Memmory!")
                ppo.update(memory)
                memory.clear()
                timestep = 0

        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    # Save the trained model
    torch.save(ppo.policy.state_dict(), "ppo_thermal_control.pth")
    print("Model saved as 'ppo_thermal_control.pth'.")

if __name__ == "__main__":
    train_ppo()
