from gymnasium.envs.registration import register

register(
    id="IndustrialEnvGym-v0",
    # entry_point="industrial_env.rl_gym_industrial_env_OLD:IndustrialEnvGym",
    entry_point="industrial_env.rl_gym_industrial_env:ServerCoolingEnv",
)
