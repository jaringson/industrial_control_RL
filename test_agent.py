import torch
import gymnasium as gym
import numpy as np
import industrial_env
from train_agent import ActorCritic, PPOConfig         # Reuse same config & model

def test_model(model_path="models/ppo_model.pt", episodes=5, render=True):
    config = PPOConfig()
    # env = gym.make("IndustrialEnvGym-v0", num_reservoirs=3)
    env = gym.make(config.env_id, num_zones=3)
    obs_dim = env.observation_space.shape[0]
    act_dim = np.prod(env.action_space.shape)

    # Load model
    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(config.device)
            with torch.no_grad():
                action, _, _ = model.get_action_and_value(obs_tensor)

            obs, reward, done, truncated, info = env.step(action.cpu().numpy())
            total_reward += reward
            step += 1

            if render:
                print(f"Step {step}: Reward={reward:.2f}, Obs={obs[:3]}, Actions={action}")
                env.render()

        print(f"Episode {ep+1} finished in {step} steps. Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_model()
