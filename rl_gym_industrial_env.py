import numpy as np
from IPython.core.debugger import set_trace

class IndustrialEnvironment:
    def __init__(self, num_reservoirs):
        """
        Initialize the industrial control environment.

        Parameters:
            num_reservoirs (int): Number of reservoirs in the system.
        """
        self.num_reservoirs = num_reservoirs
        
        # State variables
        self.temperatures = np.random.uniform(300, 400, num_reservoirs)  # Kelvin
        self.pressures = np.random.uniform(1, 10, num_reservoirs)  # Atmospheres
        self.flows = np.zeros((num_reservoirs, num_reservoirs))  # Flow rates between reservoirs (kg/s)

        # System parameters
        self.thermal_capacities = np.random.uniform(1000, 2000, num_reservoirs)  # J/(kg*K)
        self.pressure_coefficients = np.random.uniform(0.1, 1, num_reservoirs)  # kg/(m^3*s)

        # Time step for simulation (seconds)
        self.dt = 1.0

    def dynamics(self, state, heat_inputs, flow_inputs):
        """
        Compute the derivatives of the state variables.

        Parameters:
            state (tuple): Current state (temperatures, pressures).
            heat_inputs (np.array): Heat added to each reservoir (Watts).
            flow_inputs (np.array): Flow rates between reservoirs (kg/s), shape (num_reservoirs, num_reservoirs).

        Returns:
            tuple: Derivatives of temperatures and pressures.
        """
        temperatures, pressures = state
        d_temperatures = np.zeros(self.num_reservoirs)
        d_pressures = np.zeros(self.num_reservoirs)

        for i in range(self.num_reservoirs):
            # Net heat change due to heat input
            heat_change = heat_inputs[i]

            # Heat exchange due to flow
            for j in range(self.num_reservoirs):
                if i != j:
                    mass_flow = flow_inputs[j, i] - flow_inputs[i, j]
                    heat_change += (temperatures[j] - temperatures[i]) * mass_flow * self.thermal_capacities[i]

            # Temperature rate of change
            d_temperatures[i] = heat_change / self.thermal_capacities[i]

            # Net flow and pressure rate of change
            net_flow = sum(flow_inputs[i, :]) - sum(flow_inputs[:, i])
            d_pressures[i] = net_flow * self.pressure_coefficients[i]

        return d_temperatures, d_pressures

    def step(self, heat_inputs, flow_inputs):
        """
        Perform one simulation step using the RK4 method.

        Parameters:
            heat_inputs (np.array): Heat added to each reservoir (Watts).
            flow_inputs (np.array): Flow rates between reservoirs (kg/s), shape (num_reservoirs, num_reservoirs).

        Returns:
            tuple: Updated temperatures, pressures, and flows.
        """
        state = (self.temperatures, self.pressures)

        # RK4 Integration
        k1_temp, k1_pres = self.dynamics(state, heat_inputs, flow_inputs)
        k2_temp, k2_pres = self.dynamics((
            self.temperatures + 0.5 * self.dt * np.array(k1_temp),
            self.pressures + 0.5 * self.dt * np.array(k1_pres)
        ), heat_inputs, flow_inputs)
        k3_temp, k3_pres = self.dynamics((
            self.temperatures + 0.5 * self.dt * np.array(k2_temp),
            self.pressures + 0.5 * self.dt * np.array(k2_pres)
        ), heat_inputs, flow_inputs)
        k4_temp, k4_pres = self.dynamics((
            self.temperatures + self.dt * np.array(k3_temp),
            self.pressures + self.dt * np.array(k3_pres)
        ), heat_inputs, flow_inputs)

        # Update state
        self.temperatures += (self.dt / 6.0) * (np.array(k1_temp) + 2 * np.array(k2_temp) + 2 * np.array(k3_temp) + np.array(k4_temp))
        self.pressures += (self.dt / 6.0) * (np.array(k1_pres) + 2 * np.array(k2_pres) + 2 * np.array(k3_pres) + np.array(k4_pres))

        # Clip pressures to remain positive
        self.pressures = np.clip(self.pressures, 0.1, None)

        # Update flows to new inputs
        self.flows = flow_inputs

        return self.temperatures, self.pressures, self.flows

    def reset(self):
        """
        Reset the environment to initial random states.
        """
        self.temperatures = np.random.uniform(300, 400, self.num_reservoirs)
        self.pressures = np.random.uniform(1, 10, self.num_reservoirs)
        self.flows = np.zeros((self.num_reservoirs, self.num_reservoirs))

# Example Usage
env = IndustrialEnvironment(num_reservoirs=3)
heat_inputs = np.array([1000, 1500, 1200])  # Watts
flow_inputs = np.array([
    [0, 5, 3],
    [2, 0, 4],
    [3, 6, 0]
])

temperatures, pressures, flows = env.step(heat_inputs, flow_inputs)
print("Temperatures:", temperatures)
print("Pressures:", pressures)
print("Flows:", flows)
