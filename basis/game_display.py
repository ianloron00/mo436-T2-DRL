# needs pygame: 
# https://www.activestate.com/blog/how-to-use-pygame-for-game-development/

# last error
# https://github.com/hill-a/stable-baselines/issues/470

import pygame as pg
from basis.dependencies import *
import sys
import os
from pygame.locals import *

char_addr = './game-images/'

# "pixel" (image) size. Change it to change size of window.
P = 128

class Display:
    def __init__(self, W=16, H=8):

        pg.init()  # initialize pygame
        self.w = W
        self.h = H
        self.screen = pg.display.set_mode((P * W, P * H + P))

        self.icon = pg.image.load(os.path.join(char_addr, 'rat.png'))

        player = pg.image.load(os.path.join(char_addr, 'rat.png'))
        target = pg.image.load(os.path.join(char_addr, 'cheese.png'))
        hunter = pg.image.load(os.path.join(char_addr, 'cat.png'))

        self.player = pg.transform.scale(player, (P, P))
        self.target = pg.transform.scale(target, (P, P))
        self.hunter = pg.transform.scale(hunter, (P, P))

        self.font = pg.font.Font(pg.font.get_default_font(), 48)

        pg.display.quit()

    def char_mv(self, character, x, y):
        x = P * x
        y = P * y
        self.screen.blit(character, (x,y))
    
    def display_score(self, score):
        text = self.font.render("score: " + str(score), True, (255, 255, 255))
        self.screen.blit(text, dest=(0, P * self.h + int(P/4)))

    # fix indentation
    def render(self, env, time_slow=500, time_fast=150):
        pg.display.set_caption('RL Environment')
        pg.display.set_icon(self.icon)

        # initialize and paint screen
        self.screen = pg.display.set_mode((P * env.width, P * env.height + P))
        self.screen.fill((40,40,70))

        self.display_score(env.cum_reward)

        for event in pg.event.get():

            if event.type == pg.QUIT:
                sys.exit()

        playerX, playerY = env.player.x, env.player.y
        hunterX, hunterY = env.hunter.x, env.hunter.y
        targetX, targetY = env.target.x, env.target.y

        if playerX != None:
            self.char_mv(self.player, playerX, playerY)
        if hunterX != None:
            self.char_mv(self.hunter, hunterX, hunterY)
        if targetX != None:
            self.char_mv(self.target, targetX, targetY)
        pg.display.update()

        if not env.done: 
            pg.time.wait(time_fast)
        else:
            pg.time.wait(time_slow)
    
        if env.done:
            pg.display.quit()
            # sys.exit()

    def quit(self):
        pg.display.quit()
        # sys.exit()
