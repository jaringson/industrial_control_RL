import numpy as np

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

    def step(self, heat_inputs, flow_inputs):
        """
        Perform one simulation step.

        Parameters:
            heat_inputs (np.array): Heat added to each reservoir (Watts).
            flow_inputs (np.array): Flow rates between reservoirs (kg/s), shape (num_reservoirs, num_reservoirs).

        Returns:
            tuple: Updated temperatures, pressures, and flows.
        """
        # Update temperatures based on heat input and flow dynamics
        for i in range(self.num_reservoirs):
            # Net heat change due to heat input
            heat_change = heat_inputs[i] * self.dt

            # Heat exchange due to flow
            for j in range(self.num_reservoirs):
                if i != j:
                    mass_flow = flow_inputs[j, i] - flow_inputs[i, j]
                    heat_change += (self.temperatures[j] - self.temperatures[i]) * mass_flow * self.thermal_capacities[i]

            # Update temperature
            self.temperatures[i] += heat_change / self.thermal_capacities[i]

        # Update pressures based on flow dynamics
        for i in range(self.num_reservoirs):
            net_flow = sum(flow_inputs[i, :]) - sum(flow_inputs[:, i])
            self.pressures[i] += net_flow * self.pressure_coefficients[i] * self.dt

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
