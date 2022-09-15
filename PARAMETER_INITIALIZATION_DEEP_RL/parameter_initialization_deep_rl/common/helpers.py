from typing import Union, Callable
import random
from torch.nn.modules.activation import (
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
)
import os
from matplotlib import pyplot as plt
import numpy as np
import csv
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


def create_folder(path):
    """Create folder at the specified path

    Args:
        path: path to the folder to be created

    Returns:
        String: path to the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plot_performance(title, x, y, graph_label, x_label, y_label, h=None, save_path=None):
    """Plot performance of RL agent

    Args:
        title (_type_): Title of plot
        x (_type_): x values
        y (_type_): y values
        graph_label (_type_): label of the graph to be plotted
        x_label (_type_): label of the x-axis
        y_label (_type_): label of the y-axis
        h (optional): bounds of confidence interval. Defaults to None.
        save_path (optional): path to the folder where the plot is ought to be saved. Defaults to None.
    """
    plt.plot(x, y, label=graph_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if h is not None:
        plt.fill_between(x, y-h, y+h, alpha=.1)
    plt.legend()
    if save_path is not None:
        plt.savefig(f"{save_path}/{title}.png")


def configure_policy(include_bias=True):
    """Create a collection of all policy configuations

    Args:
        include_bias (bool, optional): Whether to also vary the bias init. Defaults to True.

    Returns:
        List of all configurations tested in the experiment as dictionaries 
    """
    # Weight initialization method
    weight_init = ["lecun_normal", "xavier_normal",
                   "kaiming_normal", "orthogonal", "sparse"]
    bias_init = ["random_uniform", "zeros", "0.01"]
    activation_fn = [Tanh, Sigmoid, ReLU, LeakyReLU]
    policy_net_scaling = [0.01, 1]

    policy_kwargs = []
    for weight in weight_init:
        for activation in activation_fn:
            if include_bias:
                for bias in bias_init:
                    policy_kwargs.append(dict(
                        weight_init=weight,
                        bias_init=bias,
                        activation_fn=activation,
                    ))
            else:
                policy_kwargs.append(dict(
                    weight_init=weight,
                    bias_init="random_uniform",
                    activation_fn=activation,
                ))

    return policy_kwargs


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def log_trial(log_data: dict):
    """Log results of one configuration to the global_loggin.csv file

    Args:
        log_data (dict): data that is required for the log string to be created
    """
    path = "../global_logging.csv"

    print(f"{log_data['id']},{log_data['env']},{log_data['algo']},{log_data['policy']},{log_data['weight_init']},{log_data['bias_init']},{log_data['activation_fn']},{log_data['policy_net_scaling']},{log_data['performance']},{log_data['trials']},{log_data['peak_perf']}")

    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            fieldnames = ["id", "env", "algo", "policy", "weight_init", "bias_init", "activation_fn",
                          "policy_net_scaling", "performance", "trials", "peak_perf"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(path, "a", newline="") as f:
        fieldnames = ["id", "env", "algo", "policy", "weight_init", "bias_init", "activation_fn",
                      "policy_net_scaling", "performance", "trials", "peak_perf"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(log_data)
