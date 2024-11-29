from thermal_env import MultiZoneThermalControlEnv

env = MultiZoneThermalControlEnv()

state = env.reset()
print("Initial State:", state)

for step in range(10):
    action = env.action_space.sample()  # Random actions
    next_state, reward, done, _ = env.step(action)
    print(f"Step {step + 1}:")
    print(f"  Action: {action}")
    print(f"  Next State: {next_state}")
    print(f"  Reward: {reward}")
    if done:
        print("Environment terminated due to safety constraints.")
        break
