"""
Multi-zone HVAC Gymnasium environment
"""

from typing import Tuple, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiZoneHVACEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        num_zones: int = 3,
        dt: float = 60.0,               # timestep in seconds
        zone_volumes: Optional[np.ndarray] = None,  # m^3
        zone_capacitance: Optional[np.ndarray] = None, # J/K (thermal mass)
        interzone_k: float = 2.0,       # W/K between adjacent zones (simplified)
        wall_k: Optional[np.ndarray] = None, # W/K to outside for each zone
        outside_temp: float = 35.0,     # default outside temp (C)
        supply_temp_bounds: Tuple[float, float] = (12.0, 40.0), # supply air temp (C)
        max_airflow_per_zone: float = 0.5, # m^3/s per zone
        cp_air: float = 1005.0,         # J/(kg*K)
        rho_air: float = 1.2,           # kg/m^3
        COP_cooling: float = 3.0,
        COP_heating: float = 3.0,
        fan_coeff: float = 1e-3,        # fan power ≈ coeff * (total_flow)^2 (kW)
        setpoints: Optional[np.ndarray] = None,
        humidity_control: bool = True
    ):
        super().__init__()

        self.num_zones = int(num_zones)
        self.dt = float(dt)
        self.cp = float(cp_air)
        self.rho = float(rho_air)
        self.COP_cooling = float(COP_cooling)
        self.COP_heating = float(COP_heating)
        self.fan_coeff = float(fan_coeff)

        # Geometry / thermal params
        if zone_volumes is None:
            self.zone_volumes = np.full(self.num_zones, 100.0)  # m^3
        else:
            self.zone_volumes = np.array(zone_volumes, dtype=float)

        # Thermal capacitance J/K = air_mass * cp * effective_multiplier
        if zone_capacitance is None:
            # effective multiplier to account for furnishings etc.
            air_mass = self.zone_volumes * self.rho
            self.zone_C = air_mass * self.cp * 3.0
        else:
            self.zone_C = np.array(zone_capacitance, dtype=float)

        self.interzone_k = float(interzone_k)
        if wall_k is None:
            self.wall_k = np.full(self.num_zones, 5.0)  # W/K to outside
        else:
            self.wall_k = np.array(wall_k, dtype=float)

        # Action space: [supply_temp, flow_zone_0, flow_zone_1, ..., flow_zone_n-1]
        low_action = np.concatenate(([supply_temp_bounds[0]], np.zeros(self.num_zones)))
        high_action = np.concatenate(([supply_temp_bounds[1]], np.full(self.num_zones, max_airflow_per_zone)))
        self.action_space = spaces.Box(low=low_action.astype(np.float32),
                                       high=high_action.astype(np.float32),
                                       dtype=np.float32)

        # Observation: [T_zones (n), W_zones (n), outside_temp, setpoints (n)]
        obs_low = np.concatenate((np.full(self.num_zones, -40.0),    # temps
                                  np.full(self.num_zones, 0.0),      # humidity ratio
                                  np.array([-100.0]),                # outside temp
                                  np.full(self.num_zones, -40.0)))   # setpoints
        obs_high = np.concatenate((np.full(self.num_zones, 80.0),
                                   np.full(self.num_zones, 0.05),
                                   np.array([80.0]),
                                   np.full(self.num_zones, 80.0)))
        self.observation_space = spaces.Box(low=obs_low.astype(np.float32),
                                            high=obs_high.astype(np.float32),
                                            dtype=np.float32)

        # Internal state
        self.outside_temp = float(outside_temp)
        if setpoints is None:
            self.setpoints = np.full(self.num_zones, 22.0)  # degC
        else:
            self.setpoints = np.array(setpoints, dtype=float)

        # humidity
        self.humidity_control = bool(humidity_control)
        # humidity ratio state (kg water/kg dry air), typical indoor 0.004 - 0.02
        self.min_w = 0.0
        self.max_w = 0.05

        # dynamics state placeholders
        self.T = np.zeros(self.num_zones, dtype=float)
        self.W = np.zeros(self.num_zones, dtype=float)

        # other params
        self.max_airflow_per_zone = float(max_airflow_per_zone)
        self.supply_temp_bounds = supply_temp_bounds

        # random generator
        self._rng = np.random.default_rng()

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        # initialize temperatures near setpoints with some noise
        self.T = self.setpoints + self._rng.normal(0.0, 1.0, size=self.num_zones)
        # initialize humidity ratio modestly
        self.W = np.clip(0.008 + self._rng.normal(0.0, 0.001, size=self.num_zones), self.min_w, self.max_w)

        self.outside_temp = self.outside_temp  # could be time-varying later

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.concatenate((self.T, self.W, np.array([self.outside_temp]), self.setpoints)).astype(np.float32)

    def step(self, action: np.ndarray):
        """
        action: array of length num_zones+1
            action[0]: supply temperature (C)
            action[1:]: per-zone airflow (m^3/s), clipped to [0, max_airflow_per_zone]
        """
        action = np.asarray(action, dtype=float).flatten()
        assert action.size == self.num_zones + 1, "Action length mismatch"

        T_supply = float(np.clip(action[0], self.supply_temp_bounds[0], self.supply_temp_bounds[1]))
        flows = np.clip(action[1:], 0.0, self.max_airflow_per_zone)  # m^3/s

        # convert to mass flow (kg/s)
        m_dots = flows * self.rho  # per zone

        # compute heat exchange from supply air (sensible only)
        # For each zone: Q_supply = m_dot * cp * (T_supply - T_zone)
        Q_supply = m_dots * self.cp * (T_supply - self.T)  # W (positive -> heating the zone)

        # conduction between zones - simple chain (or fully connected scaled)
        Q_inter = np.zeros_like(self.T)
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                if i == j:
                    continue
                # pairwise conduction scaling: neighbor-like but keep simple
                diff = (self.T[j] - self.T[i])
                # scale by interzone_k / (num_zones-1) so total exchange is reasonable
                Q_inter[i] += self.interzone_k / max(1, self.num_zones - 1) * diff

        # heat loss/gain to outside through walls
        Q_out = self.wall_k * (self.outside_temp - self.T)  # W (positive means outside hotter)

        # internal heat gains (occupants, equipment) - simple random/constant
        Q_internal = np.full(self.num_zones, 100.0)  # W baseline internal gains
        # You can make this time-varying or based on occupancy schedules.

        # integrate temperatures: C * dT/dt = Q_supply + Q_inter + Q_out + Q_internal
        dT = (Q_supply + Q_inter + Q_out + Q_internal) * (self.dt / self.zone_C)
        self.T = self.T + dT

        # Humidity dynamics (simple): moisture added/removed by supply and internal gains
        if self.humidity_control:
            # humidity ratio of supply (assume dry supply or set via action? assume low humidity)
            W_supply = 0.008  # typical conditioned supply
            # internal moisture generation (kg/s) per zone (people, plants)
            G_internal = np.full(self.num_zones, 1e-4)  # kg/s small
            # dW/dt = (m_dot*(W_supply - W_zone) + G_internal) / (rho_air * zone_volume)
            dW = (m_dots * (W_supply - self.W) + G_internal) * (self.dt / (self.rho * self.zone_volumes))
            self.W = np.clip(self.W + dW, self.min_w, self.max_w)

        # Approximate energy consumption:
        # total sensible heating/cooling power delivered to zones:
        Q_total = np.sum(Q_supply)  # W (positive heating, negative cooling)
        # HVAC power (kW) approximated: P_hvac = |Q_total| / COP
        if Q_total >= 0:
            P_hvac_kw = Q_total / 1000.0 / (self.COP_heating if hasattr(self, "COP_heating") else self.COP_heating)
        else:
            P_hvac_kw = -Q_total / 1000.0 / self.COP_cooling

        # fan power approx (kW) ~ coeff * (sum flows)^2
        total_flow = np.sum(flows)
        P_fan_kw = self.fan_coeff * (total_flow ** 2)

        energy_kw = P_hvac_kw + P_fan_kw

        # Rewards:
        # Penalize temperature deviation and humidity above setpoint + energy use
        temp_dev = np.abs(self.T - self.setpoints)
        temp_penalty = np.sum(temp_dev)  # °C
        # humidity penalty if above some comfortable threshold (e.g., corresponding to W_sp)
        # convert a humidity setpoint (w_sp) corresponding to 50% RH at 22C -> approx 0.008
        humidity_sp = 0.009
        hum_penalty = np.sum(np.maximum(0.0, self.W - humidity_sp)) * 1000.0

        # reward: negative weighted sum
        alpha = 1.0    # weight for temperature dev
        beta = 1.0     # weight for energy (kW)
        gamma = 0.5    # weight for humidity penalty

        reward = - (alpha * temp_penalty + beta * energy_kw + gamma * hum_penalty)

        # episode termination: optional if temps go crazy
        terminated = bool(np.any(self.T < -30.0) or np.any(self.T > 60.0))
        truncated = False

        obs = self._get_obs()
        info = {
            "T": self.T.copy(),
            "W": self.W.copy(),
            "energy_kw": float(energy_kw),
            "P_fan_kw": float(P_fan_kw),
            "P_hvac_kw": float(P_hvac_kw),
            "temp_penalty": float(temp_penalty),
            "hum_penalty": float(hum_penalty),
        }

        return obs, float(reward), terminated, truncated, info

    def render(self, mode="human"):
        # simple console render
        s = f"Temps: {np.round(self.T,2)} C | Humidity: {np.round(self.W,4)} kg/kg | Outside: {self.outside_temp:.1f} C"
        print(s)

    def close(self):
        pass


# Simple smoke-test when run as script
if __name__ == "__main__":
    env = MultiZoneHVACEnv(num_zones=3)
    obs, info = env.reset(seed=42)
    print("Initial obs:", obs)
    for t in range(10):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        print(f"Step {t}: reward={reward:.3f}, energy_kw={info['energy_kw']:.3f}")
        env.render()
        if terminated:
            print("Terminated early")
            break
