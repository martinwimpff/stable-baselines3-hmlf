from stable_baselines3.common.envs.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.envs.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)
from stable_baselines3.common.envs.multi_input_envs import SimpleMultiObsEnv
from stable_baselines3.common.envs.dummy_env import DummyEnv
from stable_baselines3.common.envs.dummy_hybrid import DummyHybrid
from stable_baselines3.common.envs.wrap_environment import wrap_environment
from stable_baselines3.common.envs.wrapper import (
    OneHotWrapper,
    SequenceWrapper,
    SimpleHybridWrapper
)