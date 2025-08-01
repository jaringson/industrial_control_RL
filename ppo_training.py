import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from industrial_env.rl_gym_industrial_env_NEW import ServerCoolingEnv

from IPython.core.debugger import set_trace


# Create and verify the environment
env = ServerCoolingEnv(num_zones=3)
check_env(env)  # Check if the environment adheres to Gym's API

# Define the RL model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    # learning_rate=0.00001,
    # n_steps=2048,
    # batch_size=64,
    # gamma=0.99
)

# Train the model
print("Starting training...")
model.learn(total_timesteps=1000000)

# Save the model
model.save("ppo_industrial_env")
print("Model saved.")

# # Load the model
# model.load("ppo_industrial_env")

# Evaluate the trained model
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} \u00b1 {std_reward}")

# Test the trained model
env = ServerCoolingEnv(num_zones=3)  # Create a fresh environment
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}")
    env.render()
    # set_trace()
    if done:
        obs = env.reset()

env.close()
