from modulefinder import Module
from stable_baselines3.common import policies
from stable_baselines3.common.type_aliases import Schedule
from functools import partial
import numpy as np
from torch import nn
import gym
from parameter_initialization_deep_rl.common.init import init_weights, init_bias
import torch as th
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN
)


class ActorCriticPolicy(policies.ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C and PPO.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param use_sde: Whether to use State Dependent Exploration or not
    :kwargs: includes weight_init, bias_init, policy_net scaling and value_net scaling in a dictionary (among others)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        use_sde: bool = False,
        **kwargs
    ):
        self.weight_init = kwargs.pop("weight_init", "xavier_normal")
        self.bias_init = kwargs.pop("bias_init", "lecun_uniform")
        self.policy_net_scaling = kwargs.pop("policy_net_scaling", 1.0)
        self.value_net_scaling = kwargs.pop("value_net_scaling", 1.0)
        super().__init__(observation_space, action_space,
                         lr_schedule, use_sde=use_sde, **kwargs)
        # Initialize all layers with specified weight and bias init and scale last layer of action and value net accordingly
        for module in self.mlp_extractor.modules():
            if isinstance(module, nn.Linear):
                module.apply(partial(
                    init_weights, weight_init=self.weight_init, module=module))
                module.apply(partial(init_bias, module=module,
                                     bias_init=self.bias_init, fan_in=module.weight.size()[1]))
        for module in self.action_net.modules():
            if isinstance(module, nn.Linear):
                module.apply(
                    partial(init_weights, weight_init=self.weight_init, module=module, gain=self.policy_net_scaling))
                module.apply(partial(init_bias, module=module,
                                     bias_init=self.bias_init, fan_in=module.weight.size()[1]))
        for module in self.value_net.modules():
            if isinstance(module, nn.Linear):
                module.apply(
                    partial(init_weights, weight_init=self.weight_init, module=module, gain=self.value_net_scaling))
                module.apply(partial(init_bias, module=module,
                                     bias_init=self.bias_init, fan_in=module.weight.size()[1]))

        # Initialize the feature extractor in case it is a CNN
        if(self.features_extractor.__class__ == NatureCNN):
            for m in self.features_extractor.modules():
                if isinstance(m, nn.Conv2d):
                    m.apply(partial(init_weights, module=m,
                                    weight_init=self.weight_init))
                    if m.bias is not None:
                        m.apply(partial(init_bias, module=m,
                                        bias_init=self.bias_init, fan_in=nn.init._calculate_fan_in_and_fan_out(m.weight)[0]))


class CnnActorCriticPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C and PPO.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
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
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
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
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
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
