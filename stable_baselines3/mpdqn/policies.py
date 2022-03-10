import copy
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch as th
from torch import nn
from gym.spaces import Space, Box

from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.td3.policies import Actor
from stable_baselines3.pdqn import MlpPolicy as PDQNPolicy
from stable_baselines3.pdqn.policies import build_state_parameter_space
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.spaces import SimpleHybrid


class MPDQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for MP-DQN.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule_q: Learning rate schedule for Q-Network (could be constant)
    :param lr_schedule_parameter: Learning rate schedule for parameter network (could be constant)
    :param net_arch_q: The specification of the Q-Network.
    :param net_arch_parameter: The specification of the parameter network.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: SimpleHybrid,
        lr_schedule_q: Schedule,
        lr_schedule_parameter: Schedule,
        net_arch_q: Optional[List[int]] = None,
        net_arch_parameter: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert type(action_space) is SimpleHybrid

        self.action_space_parameter = Box(action_space.continuous_low, action_space.continuous_high)
        self.observation_space_q = build_state_parameter_space(observation_space, action_space)
        self.action_space_q = copy.copy(action_space[0])

        super(MPDQNPolicy, self).__init__(
            observation_space,
            self.action_space_q,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.net_arch_q = self.get_net_arch(net_arch_q, features_extractor_class)
        self.net_arch_parameter = self.get_net_arch(net_arch_parameter, features_extractor_class)

        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args_q = {
            "observation_space": self.observation_space_q,
            "action_space": self.action_space_q,
            "net_arch": self.net_arch_q,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.net_args_parameter = {
            "observation_space": self.observation_space,
            "action_space": self.action_space_parameter,
            "net_arch": self.net_arch_parameter,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target, self.parameter_net = None, None, None

        self._build(lr_schedule_q, lr_schedule_parameter)

        # For formatting inside forward Q
        self.discrete_action_size = self.action_space_q.n
        self.state_size = observation_space.shape[0]
        self.offsets = np.cumsum(action_space._get_dimensions_of_continuous_spaces())
        self.offsets = np.insert(self.offsets, 0, 0)

    def _format_q_observation(self, obs: th.Tensor, action_parameters: th.Tensor, batch_size: int) -> th.Tensor:
        # Sets up multi pass structure (see https://arxiv.org/pdf/1905.04388.pdf)
        # Shape is as in P-DQN, but all parameters are set to zero
        observations = th.cat((obs, th.zeros_like(action_parameters)), dim=1)
        # Repeat for each action
        observations = observations.repeat(self.discrete_action_size, 1)

        for i in range(self.discrete_action_size):
            row_from = i * batch_size  # Beginning of current batch
            row_to = (i + 1) * batch_size  # End of current batch
            col_from = self.state_size + self.offsets[i]  # Beginning of current parameter slice
            col_to = self.state_size + self.offsets[i + 1]  # End of current parameter slice
            observations[row_from:row_to, col_from:col_to] = action_parameters[:, self.offsets[i] : self.offsets[i + 1]]

        return observations

    def _format_q_output(self, q_values: th.Tensor, batch_size: int) -> th.Tensor:
        # TODO Documentation
        Q = []
        for i in range(self.discrete_action_size):
            Qi = q_values[i * batch_size : (i + 1) * batch_size, i]
            if len(Qi.shape) == 1:
                Qi = Qi.unsqueeze(1)
            Q.append(Qi)
        Q = th.cat(Q, dim=1)
        return Q

    def forward_q(self, obs: th.Tensor, action_parameters: th.Tensor, deterministic: bool = True) -> th.Tensor:
        batch_size = action_parameters.shape[0]
        observations = self._format_q_observation(obs, action_parameters, batch_size)

        q_values = self.q_net(observations)
        Q = self._format_q_output(q_values, batch_size)
        return Q

    def _forward_q_target(self, obs: th.Tensor, action_parameters: th.Tensor, deterministic: bool = True) -> th.Tensor:
        batch_size = action_parameters.shape[0]
        observations = self._format_q_observation(obs, action_parameters, batch_size)

        q_values = self.q_net_target(observations)
        Q = self._format_q_output(q_values, batch_size)
        return Q





    def get_net_arch(self, net_arch: Optional[List[int]], features_extractor_class: Type[BaseFeaturesExtractor]):
        return PDQNPolicy.get_net_arch(self, net_arch, features_extractor_class)

    def _build(self, lr_schedule_q: Schedule, lr_schedule_parameter: Schedule) -> None:
        """
        Create the network and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        return PDQNPolicy._build(self, lr_schedule_q, lr_schedule_parameter)

    def _make_q_net(self) -> QNetwork:
        return PDQNPolicy._make_q_net(self)

    def _make_parameter_net(self) -> Actor:
        return PDQNPolicy._make_parameter_net(self)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return PDQNPolicy.forward(self, obs, deterministic)

    def forward_target(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return PDQNPolicy.forward_target(self, obs, deterministic)

    def forward_parameters(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return PDQNPolicy.forward_parameters(self, obs, deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        tensor = PDQNPolicy._predict(self, obs, deterministic)
        return tensor

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        return PDQNPolicy._get_constructor_parameters(self)

    def _update_features_extractor(
        self, net_kwargs: Dict[str, Any], observation_space: Space, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.
        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        return PDQNPolicy._update_features_extractor(self, net_kwargs, observation_space, features_extractor)

    def make_features_extractor(self, observation_space: Space) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return PDQNPolicy.make_features_extractor(self, observation_space)


MlpPolicy = MPDQNPolicy


class CnnPolicy(MPDQNPolicy):
    """
    Policy class for DQN when using images as input.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch_q: The specification of the Q-Network.
    :param net_arch_parameter: The specification of the parameter network.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: SimpleHybrid,
        lr_schedule_q: Schedule,
        lr_schedule_parameter: Schedule,
        net_arch_q: Optional[List[int]] = None,
        net_arch_parameter: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule_q,
            lr_schedule_parameter,
            net_arch_q,
            net_arch_parameter,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
