import numpy as np

# from graphs import plot_rewards, plot_victories
import pickle

import gym, os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from basis.environment import * 
from basis.dependencies import *
from basis.graphs import *
from q_extractor_features import *
from graphs import *
from callback import *

extractor = Q_Simple_Extractor()
name_model = 'dqn_mouse_game'
directory = "DQN/"

isTraining = False
isStochastic = True
SAVE_IMAGES = True
EPISODES = 30 if not isTraining else 0
TIMESTEPS = 1e7 if isTraining else 0

GAMMA = 0.95 if isTraining else 0.0
EPSILON = 0.99 if isTraining else 0.005
# achieves half of epsilon at 1/K-th of the number of timesteps.
# EPS_DECAY = 2**(-4 / EPISODES) if isTraining else 1

SHOW_EVERY = int(EPISODES/10) if isTraining else 1

NAME_COMPLEMENT = '_stochastic' if isStochastic else '_deterministic'
NAME_WEIGHTS = 'q_weights_10x10' + NAME_COMPLEMENT

SAVE_WEIGHTS = False # True if isTraining else False

N_MAX_STEPS = 650

if not os.path.exists(directory):
    os.mkdir(directory)

# True - trains de agent. False - evaluates the NN.
isTraining = True

env = GameEnv(stochastic=isStochastic)
env = Monitor(env, directory) 

#### applying stable_baselines3 model.
model = DQN('MlpPolicy', env, tensorboard_log=directory, buffer_size=300_000, 
            learning_starts=5_000, exploration_fraction=0.5, exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, verbose=0, policy_kwargs=dict( net_arch=[128, 64] ))

def get_expl_variables(model):
    start = model.exploration_rate
    end = model.exploration_final_eps
    fraction = max(0, (model.exploration_rate - model.exploration_final_eps)/
               (model.exploration_initial_eps - model.exploration_final_eps) * model.exploration_fraction)
    return start, end, fraction

def set_expl_variables(model, start=1.0, end=0.05, fraction=1.0):
    model.exploration_initial_eps = start
    model.exploration_final_eps = end
    model.exploration_fraction = fraction
    model._setup_model()

def load_model(model, zip_dir, buffer_dir, env):
    model = model.load(zip_dir, env=env)
    
    model.load_replay_buffer(buffer_dir + "/replay_buffer")

    start, end, fraction = get_expl_variables(model)
    set_expl_variables(model, start, end, fraction)
    return model

if os.path.exists(directory + name_model+".zip"):
    print("model loaded.")
    model = load_model(model, directory + name_model + ".zip", directory, env)

elif os.path.exists(directory + name_model + "/" + name_model + ".zip"):
    print("callback loaded.")
    model = load_model(model, directory + name_model + "/" + name_model + ".zip", directory + name_model, env=env)

if isTraining:
    timesteps = TIMESTEPS

    print("initial: {}, final: {}, frac: {}, remaining: {}".format(
                        model.exploration_initial_eps, 
                        model.exploration_final_eps, 
                        model.exploration_fraction,
                        model._current_progress_remaining))

    callback = SaveOnBestTrainingRewardCallback(check_freq=int(TIMESTEPS/10), 
                log_dir=directory, name=name_model, img_folder="imagesDQN/")
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # plot rewards of training
    plt_training([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Mouse Game", name_model)

    # double-save
    model.save(directory)
    print("model saved.")

else:
    rewards = evaluate_policy(model, env, n_eval_episodes=EPISODES, render=True, return_episode_rewards=True)

    # plot results from policy evaluation
    scores=rewards[0]

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(scores)) + 1, scores)
    plt.title("Pacman reward evaluation | DQN")
    plt.savefig(directory + "Pacman reward evaluation | DQN" + '.png')
    plt.show()
    plt.close()

env.close()
