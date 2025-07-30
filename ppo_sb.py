from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from industrial_env.thermal_env import MultiZoneThermalControlEnv

# 1. Create the environment
env = MultiZoneThermalControlEnv()
# check_env(env)

# 2. Wrap the environment for vectorized training (optional, improves performance)
vec_env = make_vec_env(lambda: env, n_envs=4)  # Use 4 parallel environments

# 3. Initialize the PPO agent
model = PPO("MlpPolicy", vec_env, verbose=1)

# 4. Train the agent
print("Training the agent...")
model.learn(total_timesteps=200_000)  # Adjust timesteps as needed

# 5. Save the trained model
model.save("ppo_thermal_control")
print("Model saved as 'ppo_thermal_control'.")


# 6. Test the trained model
# model = model = PPO.load("ppo_thermal_control")
print("Testing the trained model...")
test_env = MultiZoneThermalControlEnv()
state = test_env.reset()
for step in range(20):
    action, _ = model.predict(state)  # Use the trained model to choose actions
    print(action)
    state, reward, done, _ = test_env.step(action)
    test_env.render()
    if done:
        print("Terminated due to safety constraints.")
        break
