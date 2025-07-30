import numpy as np
import gymnasium
from gymnasium import spaces

from IPython.core.debugger import set_trace

class IndustrialEnvGym(gymnasium.Env):
    def __init__(self, num_reservoirs):
        super(IndustrialEnvGym, self).__init__()

        self.num_reservoirs = num_reservoirs

        # Define action and observation spaces
        # Actions: Heat inputs (Watts) and flow rates (kg/s)
        self.action_space = spaces.Box(
            low=np.array([-20] * num_reservoirs + [0] * (num_reservoirs**2)),
            high=np.array([20] * num_reservoirs + [10] * (num_reservoirs**2)),
            dtype=np.float64
        )

        # Observations: Temperatures (K) and pressures (atm)
        self.observation_space = spaces.Box(
            low=np.array([300] * num_reservoirs + [0.1] * num_reservoirs),
            high=np.array([400] * num_reservoirs + [20] * num_reservoirs),
            dtype=np.float64
        )

        # Initialize state
        self.reset()

        # System parameters
        self.thermal_capacities = np.random.uniform(1000, 2000, num_reservoirs)  # J/(kg*K)
        self.pressure_coefficients = np.random.uniform(0.1, 1, num_reservoirs)  # kg/(m^3*s)
        self.dt = 1.0  # Time step

    def dynamics(self, state, heat_inputs, flow_inputs):
        temperatures, pressures = state
        d_temperatures = np.zeros(self.num_reservoirs)
        d_pressures = np.zeros(self.num_reservoirs)

        for i in range(self.num_reservoirs):
            heat_change = heat_inputs[i]
            for j in range(self.num_reservoirs):
                if i != j:
                    mass_flow = flow_inputs[j, i] - flow_inputs[i, j]
                    heat_change += (temperatures[j] - temperatures[i]) * mass_flow * self.thermal_capacities[i]

            d_temperatures[i] = heat_change / self.thermal_capacities[i]

            net_flow = sum(flow_inputs[i, :]) - sum(flow_inputs[:, i])
            d_pressures[i] = net_flow * self.pressure_coefficients[i]

        return d_temperatures, d_pressures

    def step(self, action):
        # Parse action into heat inputs and flow inputs
        heat_inputs = action[:self.num_reservoirs]
        flow_inputs = action[self.num_reservoirs:].reshape((self.num_reservoirs, self.num_reservoirs))

        # RK4 integration
        state = (self.temperatures, self.pressures)
        k1_temp, k1_pres = self.dynamics(state, heat_inputs, flow_inputs)
        k2_temp, k2_pres = self.dynamics(
            (self.temperatures + 0.5 * self.dt * np.array(k1_temp),
             self.pressures + 0.5 * self.dt * np.array(k1_pres)),
            heat_inputs, flow_inputs)
        k3_temp, k3_pres = self.dynamics(
            (self.temperatures + 0.5 * self.dt * np.array(k2_temp),
             self.pressures + 0.5 * self.dt * np.array(k2_pres)),
            heat_inputs, flow_inputs)
        k4_temp, k4_pres = self.dynamics(
            (self.temperatures + self.dt * np.array(k3_temp),
             self.pressures + self.dt * np.array(k3_pres)),
            heat_inputs, flow_inputs)

        self.temperatures += (self.dt / 6.0) * (np.array(k1_temp) + 2 * np.array(k2_temp) + 2 * np.array(k3_temp) + np.array(k4_temp))
        self.pressures += (self.dt / 6.0) * (np.array(k1_pres) + 2 * np.array(k2_pres) + 2 * np.array(k3_pres) + np.array(k4_pres))
        self.pressures = np.clip(self.pressures, 0.1, None)

        # Calculate reward (example: minimize deviations from target state)
        target_temperatures = np.full(self.num_reservoirs, 350)  # Target temperature (K)
        target_pressures = np.full(self.num_reservoirs, 5)  # Target pressure (atm)

        reward = -np.sum((self.temperatures - target_temperatures)**2) #- np.sum((self.pressures - target_pressures)**2)

        # Check termination condition
        terminated = bool(np.any(self.temperatures < 200) or np.any(self.temperatures > 500) or \
               np.any(self.pressures < 0.1) or np.any(self.pressures > 20))

        # Compile observation
        observation = np.concatenate((self.temperatures, self.pressures))


        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random_seed = seed
            np.random.seed(self.random_seed)
        
        self.temperatures = np.random.uniform(300, 400, self.num_reservoirs)
        self.pressures = np.random.uniform(1, 10, self.num_reservoirs)
        self.flows = np.zeros((self.num_reservoirs, self.num_reservoirs))

        info = {} 

        return np.concatenate((self.temperatures, self.pressures)), info

    def render(self, mode='human'):
        print(f"State: {self.temperatures}, {self.pressures}")

# Example Usage
if __name__ == "__main__":
    env = IndustrialEnvGym(num_reservoirs=3)
    obs = env.reset()
    print("Initial Observation:", obs)

    action = np.array([10, 15, 12, 0, 5, 3, 2, 0, 4, 3, 6, 0])
    obs, reward, terminated, truncated, info = env.step(action)
    print("Next Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
