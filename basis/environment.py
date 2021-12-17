import gym
from numpy.lib.utils import info
from basis.dependencies import *
from basis.agent  import Agent
from basis.player import Player
from basis.hunter import Hunter
from basis.target import Target
from basis.graphs import *
from basis.game_display import *
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

"""
should contain:
init
make
reset
render
step
close
(validate_action)
"""

SIZE = 10
MOVE_PENALTY = 1
HUNTER_PENALTY = 300
TARGET_REWARD = 100
COLLISION_PENALTY = 10

class GameEnv(gym.Env):
    # change it not to overlay
    def __init__(self, width=SIZE, height=SIZE, stochastic=True):
        self.width = width
        self.height = height
        # reward in one timestep
        self.reward = 0
        # cumulative reward
        self.cum_reward = 0

        self.player = None
        self.hunter = None
        self.target = None

        self.isStochastic = stochastic
        self.observation_space = Box(low=0, high=SIZE, shape=(8,), dtype=np.int32)

        # from 0 to 3.
        self.action_space = Discrete(4,)

        self.done = False
        self.info = {}

        self.timestep = 0
        self.reward = 0

        self.won = False

        self.display = Display(width, height)

    def make(env="myEnv"):
        return

    def reset(self):
        self.done = False
        self.reward = 0
        self.cum_reward = 0
        self.timestep = 0 
        self.won = False

        pX, pY, hX, hY, tX, tY = self.get_positions()

        self.player = Player(pX,pY)
        self.hunter = Hunter(hX, hY)
        self.target = Target(tX, tY)

        self.update_observation_space()
        return self.observation_space

    def step(self, action):
        if self.isStochastic:
            self.hunter.move(self)

        moved = self.player.action(self, action)

        self.update_observation_space()
        
        self.update_reward(moved=moved)
        self.cum_reward += self.reward

        self.info['won'] = self.won
        return self.observation_space, self.reward, self.done, self.info

    def render(self, mode="human"):
        # redefine
        time_fast=10
        time_slow=500
        self.display.render(self, time_slow=time_slow, time_fast=time_fast)

    def close(self):
        self.display.quit()

    def update_observation_space(self):
        self.observation_space = np.array([self.width, self.height, 
                                           self.player.x, self.player.y, 
                                           self.hunter.x, self.hunter.y, 
                                           self.target.x, self.target.y])

    def update_reward(self, moved=True):
        if self.player == self.hunter:
            self.reward = -HUNTER_PENALTY
            self.done = True
        
        elif self.player == self.target:
            self.reward = TARGET_REWARD
            self.won = True
            self.done = True
        
        else:
            self.reward = -MOVE_PENALTY
        
        if not moved:
            self.reward -= COLLISION_PENALTY

    def get_positions(self):
        w, h = self.width - 1, self.height - 1

        pX, pY, hX, hY, tX, tY = 0, 0, int(w/2), int(h/2), w, h

        if self.isStochastic:
            tX = np.random.randint(0, self.width)
            tY = np.random.randint(0, self.height)
            pX, pY, hX, hY = tX, tY, tX, tY

            while(abs(pX - tX) <= 1 and (pY - tY) <= 1):
                pX = np.random.randint(0, self.width)
                pY = np.random.randint(0, self.height)

            while(abs(hX - tX) <= 1 and (hY - tY) <= 1 or hX == pX and hY == pY):
                hX = np.random.randint(0, self.width)
                hY = np.random.randint(0, self.height)

        return pX, pY, hX, hY, tX, tY