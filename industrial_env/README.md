# ServerCoolingEnv 🌡️❄️

A custom Gymnasium environment simulating a server cooling system for reinforcement learning agents. Here is a AI generated image to roughly show the setup.

![Cooling description diagram](/industrial_env/cooling_diagram.png)

## 📘 Overview

`ServerCoolingEnv` models the thermal dynamics of multiple server zones that generate heat and require active cooling to maintain safe operating temperatures. The RL agent controls coolant flow between zones to efficiently manage heat distribution and dissipation via an external cooling reservoir.

This environment is ideal for experimenting with safe RL, multi-agent coordination, energy optimization, and continuous control.

---

## 🛠️ Dynamics

- Each **zone** (e.g. server rack) generates heat via an exogenous **server load**.
- The agent controls the **coolant flow matrix**, deciding how much cooling flow to route between zones.
- Heat is naturally exchanged between zones based on:
  - Temperature differences
  - Mass flow rates
  - A heat transfer coefficient
- Each zone can also dissipate heat into a **cooling reservoir** at fixed temperature (e.g. 290 K).
- Pressure dynamics model the physical limits of flow routing.

The environment uses **4th-order Runge-Kutta integration (RK4)** to simulate the time evolution of temperatures and pressures.

---

## 📏 Observation Space

A `Box` space of shape `(2 * num_zones,)`:

[temperature_zone_1, ... ,temperature_zone_N, pressure_zone_1, ... ,pressure_zone_N]

- Temperatures: `[290 K, 400 K]`
- Pressures: `[0.1 atm, 20 atm]`

---

## 🎮 Action Space

A `Box` space of shape `(num_zones, num_zones)`:

- Each entry represents a **coolant flow rate** (kg/s) from zone *i* to zone *j*
- Allowed range: `[0, 10]`
- Diagonal entries are ignored (no self-flow)

---

## 🧠 Reward Function

The agent receives a reward based on:

- Penalizing deviation from the **target temperature** (e.g. 310 K)
- Penalizing excessive **control effort** (high flow rates)

```python
reward = - sum((T - T_target)^2) - 0.1 * sum(flow^2)
```
This encourages the agent to maintain efficient thermal regulation using minimal control input.

## 🧪 Example Usage

```python
from server_cooling_env import ServerCoolingEnv

env = ServerCoolingEnv(num_zones=3)
obs, _ = env.reset()

action = np.random.uniform(0, 5, (3, 3))
obs, reward, terminated, truncated, info = env.step(action)

env.render()
```

## 📦 Installation

This environment requires:
- Python 3.8+
- Gymnasium
- NumPy
