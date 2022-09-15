from stable_baselines3.sac import policies
from sympy import primitive
from parameter_initialization_deep_rl.common.init import init_weights, init_bias
import gym
from functools import partial
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
import torch as th
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN
)


class SACPolicy(policies.SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

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
        # Initialize the actor
        actor_layers = [module for module in self.actor.modules(
        ) if isinstance(module, nn.Linear)]
        for id, layer in enumerate(actor_layers):
            if id >= len(actor_layers) - 2:
                layer.apply(partial(init_weights, module=layer,
                                    weight_init=self.weight_init, gain=self.policy_net_scaling))
                layer.apply(partial(init_bias, module=layer,
                                    bias_init=self.bias_init, fan_in=layer.weight.size()[1]))
            else:
                layer.apply(partial(init_weights, module=layer,
                                    weight_init=self.weight_init))
                layer.apply(partial(init_bias, module=layer,
                                    bias_init=self.bias_init, fan_in=layer.weight.size()[1]))
        # Initialize the critic
        critic_layers = [module for module in self.critic.modules(
        ) if isinstance(module, nn.Linear)]
        for id, layer in enumerate(critic_layers):
            if id == len(critic_layers) - 1 or id == (len(critic_layers) / 2) - 1:
                layer.apply(partial(init_weights, module=layer,
                                    weight_init=self.weight_init, gain=self.value_net_scaling))
                layer.apply(partial(init_bias, module=layer,
                                    bias_init=self.bias_init, fan_in=layer.weight.size()[1]))
            else:
                layer.apply(partial(init_weights, module=layer,
                                    weight_init=self.weight_init))
                layer.apply(partial(init_bias, module=layer,
                                    bias_init=self.bias_init, fan_in=layer.weight.size()[1]))
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize the feature extractor in case it is a CNN
        if(self.features_extractor.__class__ == NatureCNN):
            for m in self.features_extractor.modules():
                if isinstance(m, nn.Conv2d):
                    m.apply(partial(init_weights, module=m,
                                    weight_init=self.weight_init))
                    if m.bias is not None:
                        m.apply(partial(init_bias, module=m,
                                        bias_init=self.bias_init, fan_in=nn.init._calculate_fan_in_and_fan_out(m.weight)[0]))


class CnnSACPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not, dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
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
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
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
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            weight_init=weight_init,
            bias_init=bias_init,
            policy_net_scaling=policy_net_scaling,
            value_net_scaling=value_net_scaling,
        )
