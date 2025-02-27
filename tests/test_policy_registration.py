import gym
import pytest

from stable_baselines3 import A2C, DDPG, DQN, MPDQN, PADDPG, PDQN, PPO, SAC, SDDPG, SSAC, TD3
from stable_baselines3.a2c import MlpPolicy as A2CMlpPolicy
from stable_baselines3.common.envs.dummy_hybrid import DummyHybrid
from stable_baselines3.ddpg import MlpPolicy as DDPGMlpPolicy
from stable_baselines3.dqn import MlpPolicy as DQNMlpPolicy
from stable_baselines3.mpdqn import MlpPolicy as MPDQNMlpPolicy
from stable_baselines3.paddpg import MlpPolicy as PADDPGMlpPolicy
from stable_baselines3.pdqn import MlpPolicy as PDQNMlpPolicy
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.sddpg import MlpPolicy as SDDPGMlpPolicy
from stable_baselines3.ssac import MlpPolicy as SSACMlpPolicy
from stable_baselines3.td3 import MlpPolicy as TD3MlpPolicy


@pytest.fixture
def dummy_hybrid_env():
    return DummyHybrid([1, 2, 3])


@pytest.fixture
def dummy_discrete_env():
    return gym.make("CartPole-v0")


@pytest.fixture
def dummy_box_env():
    return gym.make("Pendulum-v1")


@pytest.mark.parametrize(
    "algorithm, policy_class_mlp",
    [
        (DDPG, DDPGMlpPolicy),
        (MPDQN, MPDQNMlpPolicy),
        (PDQN, PDQNMlpPolicy),
        (PADDPG, PADDPGMlpPolicy),
        (PPO, PPOMlpPolicy),
        (SDDPG, SDDPGMlpPolicy),
        (TD3, TD3MlpPolicy),
        (SSAC, SSACMlpPolicy)
    ],
)
def test_types_of_registered_hybrid_policys(algorithm, policy_class_mlp, dummy_hybrid_env):
    alg_mlp_object = algorithm(policy_class_mlp, dummy_hybrid_env)
    alg_mlp_str = algorithm("MlpPolicy", dummy_hybrid_env)
    assert alg_mlp_object.policy_class == alg_mlp_str.policy_class
    assert isinstance(alg_mlp_str.policy, policy_class_mlp)


@pytest.mark.parametrize(
    "algorithm, policy_class_mlp",
    [
        (A2C, A2CMlpPolicy),
        (DQN, DQNMlpPolicy),
        (PPO, PPOMlpPolicy),
    ],
)
def test_types_of_registered_discrete_policys(algorithm, policy_class_mlp, dummy_discrete_env):
    alg_mlp_object = algorithm(policy_class_mlp, dummy_discrete_env)
    alg_mlp_str = algorithm("MlpPolicy", dummy_discrete_env)
    assert alg_mlp_object.policy_class == alg_mlp_str.policy_class
    assert isinstance(alg_mlp_str.policy, policy_class_mlp)


@pytest.mark.parametrize(
    "algorithm, policy_class_mlp",
    [
        (A2C, A2CMlpPolicy),
        (DDPG, DDPGMlpPolicy),
        (PPO, PPOMlpPolicy),
        (SAC, SACMlpPolicy),
        (TD3, TD3MlpPolicy),
    ],
)
def test_types_of_registered_box_policys(algorithm, policy_class_mlp, dummy_box_env):
    alg_mlp_object = algorithm(policy_class_mlp, dummy_box_env)
    alg_mlp_str = algorithm("MlpPolicy", dummy_box_env)
    assert alg_mlp_object.policy_class == alg_mlp_str.policy_class
    assert isinstance(alg_mlp_str.policy, policy_class_mlp)