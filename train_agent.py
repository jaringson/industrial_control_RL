import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import time
import industrial_env

from IPython.core.debugger import set_trace

# PPO Hyperparameters
class PPOConfig:
    env_id = "IndustrialEnvGym-v0"
    num_envs = 1
    total_timesteps = 2_000_000
    learning_rate = 1e-5
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 10
    minibatch_size = 64
    steps_per_epoch = 4096 #2048
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_action_and_value(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


def collect_trajectories(env, model, config):
    obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []
    obs, _ = env.reset()
    for _ in range(config.steps_per_epoch):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=config.device)
        with torch.no_grad():
            action, logp, value = model.get_action_and_value(obs_tensor)

        action_np = action.cpu().numpy()
        # set_trace()
        next_obs, reward, done, truncated, info = env.step(action_np)

        obs_list.append(obs)
        act_list.append(action_np)
        logp_list.append(logp.cpu().numpy())
        rew_list.append(reward)
        val_list.append(value.cpu().numpy())
        done_list.append(done)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return {
        "obs": np.array(obs_list),
        "actions": np.array(act_list),
        "logp": np.array(logp_list),
        "rewards": np.array(rew_list),
        "values": np.array(val_list),
        "dones": np.array(done_list)
    }


def compute_gae(trajectories, gamma, lam):
    rewards = trajectories["rewards"]
    values = trajectories["values"]
    dones = trajectories["dones"]
    adv = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t]
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    returns = adv + values
    return adv, returns


def train(resume=False, model_path="models/ppo_model.pt"):
    config = PPOConfig()
    writer = SummaryWriter(f"runs/ppo-industrial-{int(time.time())}")
    
    # env = gym.make(config.env_id, num_reservoirs=3)
    env = gym.make(config.env_id, num_zones=3)
    obs_dim = env.observation_space.shape[0]
    act_dim = np.prod(env.action_space.shape) # Gets shape[0]*shape[1]

    model = ActorCritic(obs_dim, act_dim).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if resume:
        checkpoint = torch.load(model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        print(f"Resumed from step {global_step}")
    else:
        global_step = 0

    for update in range(config.total_timesteps // config.steps_per_epoch):
        model.eval()
        data = collect_trajectories(env, model, config)
        adv, returns = compute_gae(data, config.gamma, config.gae_lambda)

        # Flatten
        obs = torch.tensor(data["obs"], dtype=torch.float32, device=config.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32, device=config.device)
        logp_old = torch.tensor(data["logp"], dtype=torch.float32, device=config.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=config.device)
        adv = torch.tensor(adv, dtype=torch.float32, device=config.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        model.train()
        for epoch in range(config.update_epochs):
            inds = np.arange(config.steps_per_epoch)
            np.random.shuffle(inds)
            for start in range(0, config.steps_per_epoch, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = inds[start:end]

                mb_obs = obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_logp_old = logp_old[mb_inds]
                mb_returns = returns[mb_inds]
                mb_adv = adv[mb_inds]

                new_logp, entropy, value = model.evaluate_actions(mb_obs, mb_actions)

                ratio = (new_logp - mb_logp_old).exp()
                clip_adv = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * mb_adv
                policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean()

                value_loss = ((value - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        global_step += config.steps_per_epoch
        writer.add_scalar("charts/episodic_return", np.sum(data["rewards"]), global_step)
        writer.add_scalar("charts/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("charts/value_loss", value_loss.item(), global_step)
        writer.add_scalar("charts/entropy", entropy_loss.item(), global_step)

        if update % 10 == 0:
            print(f"Update {update}: Return={np.sum(data['rewards']):.2f}")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            }, model_path)

    env.close()
    writer.close()


if __name__ == "__main__":
    train(resume=False, model_path="models/ppo_model_8.pt")