from debugpy import configure
import numpy as np
import math
from scipy import stats
from typing import Any, Union, Dict
import gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from parameter_initialization_deep_rl.common.helpers import (
    plot_performance,
    create_folder,
    log_trial,
)
from stable_baselines3.common.logger import configure
import shutil
import uuid
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def create_numpy_arr_from_logs(log_dirs):
    """Creates a list of average results that were obtained during the different runs

    Args:
        log_dirs: a list of folders where the results of the different runs are logged

    Returns:
        A list of average results that were obtained during the different runs and the number of runs
    """
    avg_returns = []
    for log_dir in log_dirs:
        logs = np.load(log_dir)
        results = logs["results"]
        avg_return = results.mean(axis=1)
        avg_returns.append(avg_return)
    avg_returns = np.array(avg_returns)
    n = avg_returns.shape[0]
    return avg_returns, n


def mean_confidence_intervals(avg_returns, n, confidence=0.95):
    """Calculate the mean and the bound of convidence interval for one confifuration averaged over all runs

    Args:
        avg_returns: list of average results that were obtained during the different runs
        n: Number of runs
        confidence: confidence interval percentage. Defaults to 0.95.

    Returns:
        _type_: _description_
    """
    AVG = avg_returns.mean(axis=0)
    STD = avg_returns.std(axis=0)
    H = STD * stats.t.ppf((1 + confidence) / 2., n-1)
    return AVG, H


def delete_folder(folder):
    """Delete folder

    Args:
        folder (String): Path to the folder that is to be deleted
    """
    shutil.rmtree(folder, ignore_errors=True)


def run_single_trial(
    env_train: Union[gym.Env, VecEnv],
    env_test: Union[gym.Env, VecEnv],
    log_dir: str,
    algorithm: BaseAlgorithm,
    policy: BasePolicy,
    hyperparameter: Dict[str, Any],
    policy_kwargs: Dict[str, Any],
    seed: int,
    train_timesteps: int,
    n_eval_episodes: int,
    eval_freq: int,
    deterministic: bool,
    render: bool,
    verbose: bool,
):
    """Train an agent for one single run

    Args:
        env_train (Union[gym.Env, VecEnv]): Training environment
        env_test (Union[gym.Env, VecEnv]): Test environment
        log_dir (str): Path to folder in which results are logged (for all runs)
        algorithm (BaseAlgorithm): Algorithm used to train the agent
        policy (BasePolicy): Policy that is used by the algorithm
        hyperparameter (Dict[str, Any]): Hyperparameter of the algorithm
        policy_kwargs (Dict[str, Any]): Addiotional arguments for the algorithm
        seed (int): random seed to be used
        train_timesteps (int): Number of timesteps to train the algorithms
        n_eval_episodes (int): Number of episodes the algorithm is evaluated in each evaluation round
        eval_freq (int): Number of times the algorithm is to be evaluated during one run
        deterministic (bool): Whether the algorithm should use a deterministic policy
        render (bool): Whether to render the actions during evaluation
        verbose (bool): Whether to log all training details to the console

    Returns:
        String: path to the folder in which the results of this specific run are logged
    """
    trial_log_dir = create_folder(f"{log_dir}/{seed}")

    # Create evaluation callback
    eval_callback = EvalCallback(
        env_test,
        log_path=trial_log_dir,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        deterministic=deterministic,
        render=render
    )

    # Instantiate the model
    model = algorithm(
        policy=policy,
        env=env_train,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=verbose,
        tensorboard_log=f"{log_dir}/tensorboard/",
        **hyperparameter
    )

    # Train the model
    model.learn(
        total_timesteps=train_timesteps,
        callback=eval_callback,
    )

    log_folder = f"{trial_log_dir}/evaluations.npz"

    return log_folder


def create_sample(avg_return_per_eval_ep, n_last_episodes=None):
    """Get a list of the performance scores P for every indepedent run

    Args:
        avg_return_per_eval_ep: list of list of average returns per episode for every trials round for every run
        n_last_episodes: Number of episodes to consider. Defaults to None.

    Returns:
        _type_: _description_
    """
    lower_bound = np.maximum(
        0, avg_return_per_eval_ep.shape[1] - n_last_episodes) if n_last_episodes is not None else 0
    sample = np.mean(avg_return_per_eval_ep[:, lower_bound:], axis=1)
    return sample


def generate_env(env_name: str, n_envs: int, seed: int):
    """Generate an OpenAi gym environment

    Args:
        env_name (str): name of the environment
        n_envs (int): number of parallel environments
        seed (int): seed with which to initialize the environment

    Returns:
        Instantiated environment(s)
    """
    if env_name == "PongNoFrameskip-v4":
        env_train = make_atari_env(
            'PongNoFrameskip-v4', n_envs=n_envs, seed=seed)
        env_train = VecFrameStack(env_train, n_stack=4)
        env_test = make_atari_env(
            'PongNoFrameskip-v4', n_envs=n_envs, seed=seed)
        env_test = VecFrameStack(env_test, n_stack=4)
    elif env_name == "BipedalWalker-v3":
        env_train = make_vec_env(env_id=env_name, n_envs=n_envs, seed=seed)
        env_train = VecNormalize(env_train)
        env_test = make_vec_env(env_id=env_name, seed=seed)
        env_test = VecNormalize(env_test)
        env_test.seed(seed=seed)
    else:
        env_train = make_vec_env(env_id=env_name, n_envs=n_envs, seed=seed)
        env_test = Monitor(gym.make(env_name))
        env_test.seed(seed=seed)
    return env_train, env_test


def run_multiple_trials(
    env_name: str,
    algorithm: BaseAlgorithm,
    policy: BasePolicy,
    hyperparameter: Dict[str, Any],
    policy_kwargs: Dict[str, Any],
    n_envs: int = 1,
    num_trials: int = 20,
    train_timesteps: int = 5e4,
    n_eval_episodes: int = 10,
    n_trial_rounds: int = 20,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True,
    continue_training: Dict[str, int] = None,
):
    """Runs a configuration for multiple runs, averages the results and logs them into the global_logging.csv file

    Args:
        env_name (str): name of the environment the agent should be trained in
        algorithm (BaseAlgorithm): algorithm used to train the agent
        policy (BasePolicy): policy used by the algorithm
        hyperparameter (Dict[str, Any]): hyperparameters for the algorithm
        policy_kwargs (Dict[str, Any]): additional kwargs for the policy
        n_envs (int, optional): number of parallel environments. Defaults to 1.
        num_trials (int, optional): number of independent runs. Defaults to 20.
        train_timesteps (int, optional): number of timesteps the agent is trained. Defaults to 5e4.
        n_eval_episodes (int, optional): number of episodes the agent is tested per evaluation round. Defaults to 10.
        n_trial_rounds (int, optional): number of times the agent is evaluated during one run. Defaults to 20.
        deterministic (bool, optional): Whether to use a deterministic policy. Defaults to True.
        render (bool, optional): Whether to render the behavior during evaluation. Defaults to False.
        verbose (bool, optional): Whether to log all details during training into the console. Defaults to True.
        continue_training (Dict[str, int], optional): Whether to continue training at a specific seed. Defaults to None.

    Returns:
        String: log string that is also appended to the global_logging.csv file
    """
    # generate uuid
    uid = continue_training["uid"] if continue_training else uuid.uuid1(
    ).__str__()
    print(f"UUID: {uid}")
    log_dir = f"../logs/{uid}"
    seeds = range(1, 1 + num_trials)
    log_dirs = [f"{log_dir}/{s}/evaluations.npz" for s in seeds]
    i = seeds.index(continue_training["last_seed"]) if continue_training else 0
    # Loop through the number of independent runs with different seeds
    while i < len(seeds):
        seed = seeds[i]
        print(f"Run: {seed}/{num_trials}")
        # Create train and test environment
        env_train, env_test = generate_env(
            env_name=env_name, n_envs=n_envs, seed=seed)
        # single run with specified seed
        run_single_trial(
            env_train=env_train,
            env_test=env_test,
            log_dir=log_dir,
            algorithm=algorithm,
            policy=policy,
            hyperparameter=hyperparameter,
            policy_kwargs=policy_kwargs,
            seed=seed,
            train_timesteps=train_timesteps,
            n_eval_episodes=n_eval_episodes,
            eval_freq=math.ceil(train_timesteps / n_trial_rounds / n_envs),
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )
        i += 1
    # Generate statistics
    avg_returns, n = create_numpy_arr_from_logs(log_dirs)
    sample = create_sample(avg_returns)
    performance_score = np.average(sample)
    AVG, H = mean_confidence_intervals(avg_returns, n)

    # Log the performance and trial results as well as the configuration in the global_logging.csv file
    log_data = {
        "id": uid,
        "env": env_name,
        "algo": algorithm.__name__,
        "policy": policy.__name__,
        "weight_init": policy_kwargs["weight_init"],
        "bias_init": policy_kwargs["bias_init"],
        "activation_fn": policy_kwargs["activation_fn"].__name__,
        "performance": performance_score,
        "trials": "".join(f"{str(x)} " for x in AVG),
        "peak_perf": np.max(AVG)
    }
    log_data["policy_net_scaling"] = policy_kwargs["policy_net_scaling"] if "policy_net_scaling" in policy_kwargs else ""

    log_trial(log_data)

    # Delete log and model folder
    delete_folder(log_dir)

    return log_data
