from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import shutil, os, numpy as np
from graphs import plt_iter_training
from iCSV import *

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, 
                 name='best_model', verbose=1, img_folder="images/"):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, name)
        self.best_mean_reward = -np.inf
        self.name = name
        self.episode = 0
        self.img_folder= img_folder

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        plt_iter_training(self.img_folder, [], [])
        ini_csv(self.log_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')

            if len(x) > 0:
                
                save_in_csv(x[self.episode : ], y[self.episode : ])
                
                plt_iter_training(self.img_folder, x, y)                

                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - e: {:.4f}".format(
                        self.best_mean_reward, 
                        mean_reward,
                        self.model.exploration_rate))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    
                    path = self.save_path + "/" + self.name

                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(path)

                # buffer's callback
                # self.model.save_replay_buffer(self.log_dir + "/replay_buffer")
                # print("replay buffer saved.")

                # if self.save_in_drive:
                file_paths = []

                for root, directories, files in os.walk(self.img_folder):
                    for filename in files:
                        # Join the two strings in order to form the full filepath.
                        filepath = os.path.join(root, filename)
                        file_paths.append(filepath)

                for root, directories, files in os.walk("tmp/"):
                    for filename in files:
                        # Join the two strings in order to form the full filepath.
                        filepath = os.path.join(root, filename)
                        file_paths.append(filepath)

                # for f in file_paths:
                    # shutil.copy(f, self.drive_path + "/" + f)

                # update former episode's benchmark
                self.episode = len(x)
                    
        return True

"""
tensorboard --logdir tmp/figure
"""