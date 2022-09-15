import gym
from stable_baselines3 import DQN
from functools import partial
from stable_baselines3.dqn import policies
from stable_baselines3.common.type_aliases import Schedule
from parameter_initialization_deep_rl.common.init import init_weights, init_bias
from torch import nn
import torch as th
from typing import Any, Dict, List, Optional, Type
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN
)


class DQNPolicy(policies.DQNPolicy):
    """
    Policy class with Q-Value Net and target net for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :kwargs: includes weight_init, bias_init, policy_net scaling and value_net scaling in a dictionary (optionally)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        **kwargs
    ):
        self.weight_init = kwargs.pop("weight_init", "xavier_normal")
        self.bias_init = kwargs.pop("bias_init", "lecun_uniform")
        self.policy_net_scaling = kwargs.pop("policy_net_scaling", 1.0)
        self.value_net_scaling = kwargs.pop("value_net_scaling", 1.0)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        # Initialize q net as well as target
        q_net_layers = [module for module in self.q_net.modules(
        ) if isinstance(module, nn.Linear)]
        for id, layer in enumerate(q_net_layers):
            if id == len(q_net_layers) - 1:
                layer.apply(partial(init_weights, module=layer,
                                    weight_init=self.weight_init, gain=self.value_net_scaling))
                layer.apply(partial(init_bias, module=layer,
                                    bias_init=self.bias_init, fan_in=layer.weight.size()[1]))
            else:
                layer.apply(partial(init_weights, module=layer,
                                    weight_init=self.weight_init))
                layer.apply(partial(init_bias, module=layer,
                                    bias_init=self.bias_init, fan_in=layer.weight.size()[1]))
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Initialize the feature extractor in case it is a CNN
        if(self.features_extractor.__class__ == NatureCNN):
            for m in self.features_extractor.modules():
                if isinstance(m, nn.Conv2d):
                    m.apply(partial(init_weights, module=m,
                                    weight_init=self.weight_init))
                    if m.bias is not None:
                        m.apply(partial(init_bias, module=m,
                                        bias_init=self.bias_init, fan_in=nn.init._calculate_fan_in_and_fan_out(m.weight)[0]))


class CnnDQNPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
        :param weight_init: method to initialize the weights
    :param bias_init: method to initialize the biases
    :param policy_net_scaling: scaling factor of last layer of policy net
    :param value_net_scaling: scaling factor of last layer of value net
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        weight_init: str = "xavier_normal",
        bias_init: str = "lecun_uniform",
        policy_net_scaling: int = 1.0,
        value_net_scaling: int = 1.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            weight_init=weight_init,
            bias_init=bias_init,
            policy_net_scaling=policy_net_scaling,
            value_net_scaling=value_net_scaling,
        )
