# Register environment
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv

import isolation_env_no_mask

def env_creator(config):
    env = isolation_env_no_mask.env()
    return env

register_env("isolation_v0", lambda config: PettingZooEnv(env_creator(config)))

# Configure PPO
# Both agents share policy parameters
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("isolation_v0")
    .training(gamma=0.9, lr=0.01)
    .multi_agent(
        policies={"shared_policy"},
        # All agents map to the exact same policy.
        policy_mapping_fn=(lambda aid, *args, **kwargs: "shared_policy"),
    )
)

# Build.
algo = config.build()

# Train
print(algo.train())

