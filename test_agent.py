import torch
from train_agent import PPOAgent
from industrial_env.thermal_env import MultiZoneThermalControlEnv

from IPython.core.debugger import set_trace

# Load environment
env = MultiZoneThermalControlEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.nvec

# Load trained model
model = PPOAgent(state_dim, action_dim)
model.load_state_dict(torch.load("ppo_thermal_control.pth"))
model.eval()  # Set to evaluation mode

# Test the agent
state = env.reset()
done = False
total_reward = 0
while not done:
    with torch.no_grad():
        action_probs, _ = model(torch.FloatTensor(state))
    # Separate logits for each action dimension
    action_probs_split = torch.split(action_probs, tuple(action_dim), dim=-1)
    
    actions = []
    for prob in action_probs_split:
        action = torch.argmax(prob).item()  # Choose the best action
        actions.append(action)
    # set_trace()
    print("Action: ", actions)
    state, reward, done, _ = env.step(actions)
    env.render()  # Render the environment (if implemented)
    total_reward += reward

print(f"Total reward during testing: {total_reward}")
