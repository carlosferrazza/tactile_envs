from gymnasium.envs.registration import register

print("Registering Tactile Envs")
register(
     id="tactile_envs/Insertion-v0",
     entry_point="tactile_envs.envs:InsertionEnv",
     max_episode_steps=300,
)