import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from train_agent import ActorCritic, PPOConfig
import industrial_env  # Registers the env

from IPython.core.debugger import set_trace

def test_model_with_visualization(model_path="models/ppo_model_2.pt", max_steps=200):
    config = PPOConfig()
    env = gym.make("IndustrialEnvGym-v0", num_zones=3)
    obs_dim = env.observation_space.shape[0]
    act_dim = np.prod(env.action_space.shape)

    # Load trained model
    checkpoint = torch.load(model_path, map_location=config.device)
    model = ActorCritic(obs_dim, act_dim).to(config.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    obs, _ = env.reset()
    zone_count = env.unwrapped.num_zones

    # Setup plots
    fig, (ax_temp, ax_action) = plt.subplots(2, 1, figsize=(6, 8))
    temp_bar = ax_temp.bar(range(zone_count), obs[:zone_count])
    ax_temp.set_ylim(250, 400)
    ax_temp.set_title("Zone Temperatures (K)")

    action_im = ax_action.imshow(np.zeros((zone_count, zone_count)), vmin=0, vmax=10, cmap='Blues')
    ax_action.set_title("Flow Actions Matrix")
    # plt.tight_layout()

    def update(frame):
        nonlocal obs

        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(config.device)
        with torch.no_grad():
            action, _, _ = model.get_action_and_value(obs_tensor)

        action_np = action.cpu().numpy().reshape((zone_count, zone_count))
        obs, reward, done, truncated, info = env.step(action_np)

        # Update temperature bars
        for i, bar in enumerate(temp_bar):
            bar.set_height(obs[i])

        # Update flow matrix
        action_im.set_array(action_np)

        ax_temp.set_ylabel(f"Step: {frame}, Reward: {reward:.2f}")

        return temp_bar, action_im

    ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=20, blit=False)

    # Save the animation as an MP4
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('videos/animation.mp4', writer=writer)

    plt.show()
    env.close()

if __name__ == "__main__":
    test_model_with_visualization()
