import numpy as np
import gymnasium
from gymnasium import spaces

class LoadModel:
    def __init__(self, num_zones, theta=0.05, mu=25, sigma=5):
        self.num_zones = num_zones
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.state = np.random.uniform(mu - 10, mu + 10, num_zones)

    def step(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.num_zones)
        self.state += dx
        self.state = np.clip(self.state, 0, 50)  # Optional bounds
        return self.state
    

class ServerCoolingEnv(gymnasium.Env):
    def __init__(self, num_zones):
        super(ServerCoolingEnv, self).__init__()

        self.num_zones = num_zones

        self.load_model = LoadModel(self.num_zones)

        # Agent controls flow between zones (coolant routing)
        self.action_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(num_zones, num_zones + 1),
            dtype=np.float64
        )

        # Observations: Zone temperatures (K) and pressures (atm)
        self.observation_space = spaces.Box(
            low=np.array([290] * num_zones + [0.1] * num_zones),
            high=np.array([400] * num_zones + [20] * num_zones),
            dtype=np.float64
        )

        # Constants
        self.dt = 1.0
        self.thermal_capacities = np.random.uniform(1000, 2000, num_zones)  # J/K per zone
        self.pressure_coefficients = np.random.uniform(0.1, 1.0, num_zones)  # flow to pressure change
        self.heat_transfer_coeff = 0.1  # tunable
        self.cooling_temp = 290.0  # fixed ambient coolant temperature
        self.cooling_coeff = 1.0  # cooling sink strength

        self.reset()

    def dynamics(self, temperatures, pressures, flow_inputs, server_loads, cooling_controls):
        d_temperatures = np.zeros(self.num_zones)
        d_pressures = np.zeros(self.num_zones)

        for i in range(self.num_zones):
            # Server load is external heat input
            heat_change = server_loads[i]

            # Heat exchange with other zones via flow
            for j in range(self.num_zones):
                if i != j:
                    temp_diff = temperatures[j] - temperatures[i]
                    mass_flow = flow_inputs[j, i] - flow_inputs[i, j]
                    heat_change += self.heat_transfer_coeff * temp_diff * mass_flow

            # Heat removal via cooling reservoir
            # heat_change -= self.cooling_coeff * (temperatures[i] - self.cooling_temp)
            heat_change -= self.cooling_coeff * cooling_controls[i] * (temperatures[i] - self.cooling_temp)

            # Temperature rate of change
            d_temperatures[i] = heat_change / self.thermal_capacities[i]

            # Pressure changes from net flow
            net_flow = np.sum(flow_inputs[i, :]) - np.sum(flow_inputs[:, i])
            d_pressures[i] = net_flow * self.pressure_coefficients[i]

        return d_temperatures, d_pressures

    def step(self, action):

        self.timestep += 1

        action = action.reshape(self.num_zones, self.num_zones + 1)
        flow_inputs = action[:, :-1]
        cooling_controls = action[:, -1]

        # action = action.reshape(self.num_zones, self.num_zones)
        # flow_inputs = action
        # cooling_controls = np.ones_like(action[:, -1])

        flow_inputs = np.clip(flow_inputs, 0.0, 10.0)

        # Simulated server loads (can randomize or make time-varying)
        # server_loads = np.random.uniform(0, 50, self.num_zones)  # Watts
        server_loads = self.load_model.step()

        # Runge-Kutta Integration (RK4)
        k1_t, k1_p = self.dynamics(self.temperatures, self.pressures, flow_inputs, server_loads, cooling_controls)
        k2_t, k2_p = self.dynamics(self.temperatures + 0.5 * self.dt * k1_t,
                                   self.pressures + 0.5 * self.dt * k1_p,
                                   flow_inputs, server_loads, cooling_controls)
        k3_t, k3_p = self.dynamics(self.temperatures + 0.5 * self.dt * k2_t,
                                   self.pressures + 0.5 * self.dt * k2_p,
                                   flow_inputs, server_loads, cooling_controls)
        k4_t, k4_p = self.dynamics(self.temperatures + self.dt * k3_t,
                                   self.pressures + self.dt * k3_p,
                                   flow_inputs, server_loads, cooling_controls)

        self.temperatures += (self.dt / 6.0) * (k1_t + 2*k2_t + 2*k3_t + k4_t)
        self.pressures += (self.dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        self.pressures = np.clip(self.pressures, 0.1, 20.0)

        # Reward: penalize overheating and control effort
        target_temp = 310.0  # Ideal operating temp (K)
        temp_penalty = np.sum((self.temperatures - target_temp) ** 2)
        flow_penalty = np.sum(flow_inputs ** 2)
        cooling_penalty = np.sum(cooling_controls ** 2)

        # ---------------- #
        # reward = -temp_penalty - 0.1 * flow_penalty
        
        # ---------------- #
        # reward = -temp_penalty - 0.1 * flow_penalty - 0.5 * cooling_penalty
        # reward = -temp_penalty - 0.5 * cooling_penalty
        # if np.all(np.abs(self.temperatures - target_temp) < 2.0):
        #     reward += 5.0 * (1.0 / (1 + self.timestep))

        # ---------------- #
        time_weighted_penalty = 0.01 * temp_penalty * self.timestep
        # reward = - 0.1 * flow_penalty - time_weighted_penalty
        reward = - time_weighted_penalty - 0.5 * cooling_penalty

        # ---------------- #
        terminated = bool(np.any(self.temperatures < 250) or np.any(self.temperatures > 500))

        observation = np.concatenate((self.temperatures, self.pressures))
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.temperatures = np.random.uniform(280, 340, self.num_zones)
        self.pressures = np.random.uniform(1, 5, self.num_zones)
        self.timestep = 0
        return np.concatenate((self.temperatures, self.pressures)), {}

    def render(self, mode='human'):
        print(f"Temperatures (K): {self.temperatures}")
        print(f"Pressures (atm): {self.pressures}")

# Example run
if __name__ == "__main__":
    env = ServerCoolingEnv(num_zones=3)
    obs, _ = env.reset()
    print("Initial Observation:", obs)
    action = np.random.uniform(0, 5, (3, 3))
    obs, reward, terminated, truncated, info = env.step(action)
    print("Next Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
