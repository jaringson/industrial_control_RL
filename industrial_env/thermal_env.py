import gym
from gym import spaces
import numpy as np

class MultiZoneThermalControlEnv(gym.Env):
    def __init__(self, num_zones=5, target_temp=29, temp_bounds=(15, 30), interaction_factor=0.1):
        super(MultiZoneThermalControlEnv, self).__init__()
        self.num_zones = num_zones+1
        self.target_temp = target_temp
        self.temp_bounds = temp_bounds
        self.interaction_factor = interaction_factor
        
        # State: current temperatures for all zones
        self.observation_space = spaces.Box(low=temp_bounds[0], high=temp_bounds[1], shape=(self.num_zones,), dtype=np.float32)
        
        # Actions: -1 (cool), 0 (maintain), 1 (heat) for each zone
        self.action_space = spaces.MultiDiscrete([3] * num_zones)
        
        # Initial state
        self.state = np.random.uniform(temp_bounds[0], temp_bounds[1], self.num_zones)
        self.state[-1] = self.target_temp

        # Seed
        self.random_seed = 0

    def seed(self, seed):
        self.random_seed = seed
        np.random.seed(self.random_seed)

    
    def reset(self):
        self.state = np.random.uniform(self.temp_bounds[0], self.temp_bounds[1], self.num_zones)
        self.target_temp = self.state[-1]
        return self.state
    
    def step(self, action):
        energy_usage = 0
        next_state = self.state.copy()


        for i in range(self.num_zones-1):
            # Apply action to change temperature
            if action[i] == 0:  # maintain
                pass
            elif action[i] == 1:  # heat
                next_state[i] += 1
                energy_usage += 1
            elif action[i] == 2:  # cool
                next_state[i] -= 1
                energy_usage += 1
            
            # Interaction with neighboring zones
            for j in range(self.num_zones-1):
                if np.abs(i-j) > 1:
                    continue
                if i != j:
                    next_state[i] += self.interaction_factor * (self.state[j] - self.state[i])

        # Ensure temperatures remain within bounds
        next_state = np.clip(next_state, self.temp_bounds[0], self.temp_bounds[1])
        
        # Calculate reward: penalize energy usage and temperature deviation
        temp_deviation = np.abs(next_state - self.target_temp)
        # reward = -energy_usage - np.sum(temp_deviation)
        reward = - np.sum(temp_deviation)
        
        # Check if the state is safe
        done = not np.all((next_state >= self.temp_bounds[0]) & (next_state <= self.temp_bounds[1]))
        
        self.state = next_state
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"Temperatures: {self.state}")

