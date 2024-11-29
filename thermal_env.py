import gym
from gym import spaces
import numpy as np

class MultiZoneThermalControlEnv(gym.Env):
    def __init__(self, num_zones=3, target_temp=22, temp_bounds=(15, 30), interaction_factor=0.1):
        super(MultiZoneThermalControlEnv, self).__init__()
        self.num_zones = num_zones
        self.target_temp = target_temp
        self.temp_bounds = temp_bounds
        self.interaction_factor = interaction_factor
        
        # State: current temperatures for all zones
        self.observation_space = spaces.Box(low=temp_bounds[0], high=temp_bounds[1], shape=(num_zones,), dtype=np.float32)
        
        # Actions: -1 (cool), 0 (maintain), 1 (heat) for each zone
        self.action_space = spaces.MultiDiscrete([3] * num_zones)
        
        # Initial state
        self.state = np.random.uniform(temp_bounds[0], temp_bounds[1], num_zones)
    
    def reset(self):
        self.state = np.random.uniform(self.temp_bounds[0], self.temp_bounds[1], self.num_zones)
        return self.state
    
    def step(self, action):
        energy_usage = 0
        next_state = self.state.copy()
        
        for i in range(self.num_zones):
            # Apply action to change temperature
            if action[i] == 0:  # maintain
                pass
            elif action[i] == 1:  # heat
                next_state[i] += 1
                energy_usage += 1
            elif action[i] == -1:  # cool
                next_state[i] -= 1
                energy_usage += 1
            
            # Interaction with neighboring zones
            for j in range(self.num_zones):
                if i != j:
                    next_state[i] += self.interaction_factor * (self.state[j] - self.state[i])
        
        # Ensure temperatures remain within bounds
        next_state = np.clip(next_state, self.temp_bounds[0], self.temp_bounds[1])
        
        # Calculate reward: penalize energy usage and temperature deviation
        temp_deviation = np.abs(next_state - self.target_temp)
        reward = -energy_usage - np.sum(temp_deviation)
        
        # Check if the state is safe
        done = not np.all((next_state >= self.temp_bounds[0]) & (next_state <= self.temp_bounds[1]))
        
        self.state = next_state
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"Temperatures: {self.state}")

