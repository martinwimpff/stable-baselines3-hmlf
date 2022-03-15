import gym
import pytest

from stable_baselines3 import A2C, DDPG, DQN, MPDQN, PADDPG, PDQN, PPO, SAC, SDDPG, TD3
from stable_baselines3.common.envs import DummyHybrid, wrap_environment
from stable_baselines3.common.envs.wrapper import OneHotWrapper, SequenceWrapper, SimpleHybridWrapper
from stable_baselines3.common.spaces import OneHotHybrid, SimpleHybrid


@pytest.mark.parametrize(
    "algorithm",
    [A2C, DQN, DDPG, TD3, PPO, SAC]
)
@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v0"),
        gym.make("Pendulum-v1"),
    ],
)
def test_no_wrapper(algorithm, env):
    wrapped_env = wrap_environment(algorithm, env)
    assert wrapped_env is env


@pytest.fixture
def dummy_hybrid_env():
    return DummyHybrid([1, 2, 3])


@pytest.mark.parametrize(
    "algorithm", [PPO, MPDQN, PDQN]
)
def test_simple_hybrid_wrapper(algorithm, dummy_hybrid_env):
    wrapped_env = wrap_environment(algorithm, dummy_hybrid_env)
    assert isinstance(wrapped_env, SimpleHybridWrapper)


@pytest.mark.parametrize(
    "algorithm",
    [
        PADDPG,
    ],
)
def test_one_hot_wrapper(algorithm, dummy_hybrid_env):
    wrapped_env = wrap_environment(algorithm, dummy_hybrid_env)
    assert isinstance(wrapped_env, OneHotWrapper)


@pytest.mark.parametrize(
    "algorithm",
    [
        SDDPG,
    ],
)
def test_sequence_wrapper(algorithm, dummy_hybrid_env):
    sequence = [1, 0, 0, 1]
    wrapped_env = wrap_environment(algorithm, dummy_hybrid_env, sequence)
    assert isinstance(wrapped_env, SequenceWrapper)
    assert wrapped_env.sequence_curator.sequence == sequence

    with pytest.raises(AssertionError):
        wrap_environment(algorithm, dummy_hybrid_env, [])

    with pytest.raises(AssertionError):
        wrap_environment(algorithm, dummy_hybrid_env, "abc")


@pytest.mark.parametrize(
    "algorithm",
    [
        "Baumhaus",
        "",
        SimpleHybrid, 
        OneHotHybrid
    ]
)
def test_unknown(algorithm, dummy_hybrid_env):
    with pytest.raises(NotImplementedError):
        wrap_environment(algorithm, dummy_hybrid_env)