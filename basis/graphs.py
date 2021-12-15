from basis.dependencies import *

def plot_rewards(moving_avg, title='rewards', save=True):
  plt.plot([i for i in range(len(moving_avg))], moving_avg)
  plt.ylabel(f"reward")
  plt.xlabel('episode #')
  if save: 
    plt.savefig('DQN/images/'+ title +'.png')
  plt.show()

def plot_victories(victories_avg, title='victories_iterations', save=True):
  plt.plot([i for i in range(len(victories_avg))], victories_avg)    
  plt.title("victories per number iterations")
  if save:
    plt.savefig('DQN/images/' + title + '.png')
  plt.show()