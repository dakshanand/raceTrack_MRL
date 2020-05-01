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

class Environment:

    def __init__(self, layout):
        '''
        initialize step_count to be 0
        '''
        self.layout = layout
        self.rows = self.layout.racetrack.width
        self.cols = self.layout.racetrack.height

        self.step_count = 0

    def start(self):
        '''
        Makes the velocity of the car to be zero
        Returns the randomly selected start state.
        '''
        state = np.zeros(4,dtype='int')
        state[0], state[1] = random.choice(self.layout.startStates)

        '''
        state[2] and state[3] are already zero
        '''
        return state

    def step(self, state, action):
        '''
        Returns the reward and new state when action is taken on state
        Checks the following 2 cases maintaining the order:
            1. car finishes race by crossing the finish line
            2. car goes out of track
        Ends the episode by returning reward as None
        and state as usual (which will be terminating)
        '''
        reward = -1

        if (self.is_finish_line_crossed(state, action)):
            new_state = self.get_new_state(state, action)
            self.step_count += 1
            return None, new_state

        elif (self.is_out_of_track(state, action)):
            new_state = self.start()
        else:
            new_state = self.get_new_state(state, action)

        self.step_count += 1

        return reward, new_state

    def is_out_of_track(self, state, action):
        '''
        Returns True if the car goes out of track if action is taken on state
                False otherwise
        '''
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        if new_cell[0] < 0 or new_cell[0] >= self.rows or new_cell[1] < 0 or new_cell[1] >= self.cols:
            return True

        else:
            return self.layout.racetrack[new_cell[0]][new_cell[1]] == -1


    def reset(self):
        self.step_count = 0

    def rotate_vector(self, x, y, angle):
        newX = x * math.cos(angle) - y * math.sin(angle)
        newY = x * math.sin(angle) + y * math.cos(angle)
        return [newX, newY]

    def noisy_action(self, action):
        from random import random
        r = random()

        # action succeeds normally
        if r <= 0.9:
            return action
        # left 45
        elif r <= 0.93:
            return self.rotate_vector(action[0], action[1], -math.pi/4)
        # right 45
        elif r <= 0.96:
            return self.rotate_vector(action[0], action[1], math.pi/4)
        # left 90
        elif r <= 0.97:
            return self.rotate_vector(action[0], action[1], -math.pi/2)
        # right 90
        elif r <= 0.98:
            return self.rotate_vector(action[0], action[1], math.pi/2)
        #action fails
        else:
            return [0, 0]

    def get_new_state(self, state, action):
        '''
        Get new state after applying action on this state
        Assumption: The car keeps on moving with the current
        velocity and then action is applied to
        change the velocity
        '''
        action = self.noisy_action(action)
        new_state = state.copy()
        new_state[0] = state[0] - state[2]
        new_state[1] = state[1] + state[3]
        new_state[2] = state[2] + action[0]
        new_state[3] = state[3] + action[1]
        return new_state

    def select_randomly(self,NUMPY_ARR):
        '''
        Returns a value uniform randomly from NUMPY_ARR
        Here NUMPY_ARR should be 1 dimensional
        '''
        return np.random.choice(NUMPY_ARR)

    def set_zero(NUMPY_ARR):
        '''
        Returns NUMPY_ARR after making zero all the elements in it
        '''
        NUMPY_ARR[:] = 0
        return NUMPY_ARR

    def is_finish_line_crossed(self, state, action):
        '''
        Returns True if the car crosses the finish line
                False otherwise
        '''
        return False
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        '''
        new_cell's row index will be less
        '''
        rows = np.array(range(new_cell[0],old_cell[0]+1))
        cols = np.array(range(old_cell[1],new_cell[1]+1))
        fin = set([tuple(x) for x in self.layout.finishStates])
        row_col_matrix = [(x,y) for x in rows for y in cols]
        intersect = [x for x in row_col_matrix if x in fin]

        return len(intersect) > 0
