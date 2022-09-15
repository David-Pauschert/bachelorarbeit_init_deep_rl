# Installation
Before getting started, the repository needs to be cloned. After that, the additional code that is organized as a package needs to be installed, so starting from the root folder of the project:
```sh
pip install -e ./PARAMETER_INITIALIZATION_DEEP_RL/
```
Setting the -e flag is important to install the package in development mode and thus to get the latest version of the code. Depending on configuration of the user, additional packages such as StableBaselines3, Numpy, etc. might need to be installed as well.

## Part 1: Measuring the impact of weight initialization
### Training Agents:
- An example script for running the experiment with different configurations for the algorithm **SAC** in the environment **BipedalWalker-v3** is provided in the folder ./train_agents. 
- In case the user wants to conduct the experiment in another environment with a different algorithm the hyperparameters need to be adjusted accordingly.
- The algorithms that can be used are SAC, PPO, TD3, DQN and A2C while the available policies are SACPolicy, ActorCriticPolicy, TD3Policy and DQNPolicy.
- All functions including **run_multiple_trials** are documented in the package PARAMETER_INITIALIZATION_DEEP_RL.
- The results of the experiment are logged to the **global_logging.csv** file

### Analyzing results:
- The script for analyzing the results in the log file *global_logging.csv* is provided under *./analyze_results/analyze.ipynb*.
- All plots are saved to *./analyze_results/plots*.


## Part 2: Measuring the benefits of pretraining with BC
- In order to to perform the pretraining with behavior cloning the [imitation](https://github.com/HumanCompatibleAI/imitation) library needs to be installed.
- An example script for pretraining DQN with samples coming from the naive heuristic is provided under *./experiment_part2/pretraining_dqn_naive.ipynb* and an example script for pretraining PPO with samples coming from the expert heuristic is provided under *./experiment_part2/pretraining_ppo_expert.ipynb*.
- The log data as well as the plots obtained in the second part of our experiment can be found under *./experiment_part2/logs/*.