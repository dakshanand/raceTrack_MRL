# environment.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Vaibhav Gupta

import math
import numpy as np
import pygame
import random

import layout_parser
from variables import *

class Agent:

    #HELPFUL FUNCTIONS
    def possible_actions(self, velocity):
        '''
        *** Performs two tasks, can be split up ***
        Universe of actions:  alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]

        Uses constraints to filter out invalid actions given the velocity

        0 <= v_x < 5
        0 <= v_y < 5
        v_x and v_y cannot be made both zero (you can't take
        an action which would make them zero simultaneously)
        Returns list of possible actions given the velocity
        '''
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        alpha = [np.array(x) for x in alpha]

        beta = []
        for i,x in zip(range(9),alpha):
            new_vel = np.add(velocity,x)
            if (new_vel[0] < 5) and (new_vel[0] >= 0) and (new_vel[1] < 5) and \
             (new_vel[1] >= 0) and ~(new_vel[0] == 0 and new_vel[1] == 0):
                beta.append(i)
        beta = np.array(beta)

        return beta

    def map_to_1D(self,action):
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        for i,x in zip(range(9),alpha):
            if action[0]==x[0] and action[1]==x[1]:
                return i

    def map_to_2D(self,action):
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        return alpha[action]

    #CONSTRUCTOR
    def __init__(self):
        pass

    def get_action(self, state, policy):
        '''
        Returns action given state using policy
        '''
        # return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
        # action = np.random.choice(possible_actions)
        return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
