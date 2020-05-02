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
import pickle

import layout_parser
from variables import *

class Environment:
    '''
        Assumption: The car keeps on moving with the current
        velocity and then action is applied to
    '''

    def __init__(self, layout):
        '''
        initialize step_count to be 0
        '''
        self.layout = layout
        self.dist = pickle.load( open( "dist.p", "rb" ) )
        self.rows = self.layout.racetrack.width
        self.cols = self.layout.racetrack.height
        self.max_vel_mag = 3
        self.step_size = 0.1
        self.step_count = 0

    def start(self):
        '''
        Makes the velocity of the car to be zero
        Returns the randomly selected start state.
        '''
        # state = np.zeros(4,dtype='int')
        state = np.zeros(4,dtype='float')
        state[0], state[1] = random.choice(self.layout.startStates)
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
        self.step_count += 1
        status, final_pos = self.move_the_car(np.array(state[:2]), np.array(state[2:4]))
        del_dist = self.dist[int(state[0])][int(state[1])] - self.dist[int(final_pos[0])][int(final_pos[1])]

        # print state[:2], final_pos, del_dist
        state[:2] = final_pos

        if status == 'collision':
            accx = self.get_deacceleration(state[2])
            accy = self.get_deacceleration(state[3])
            acceleration = np.array([accx, accy])
            state[2:4] = self.update_velocity(state[2:4], acceleration)
        elif status == 'finish':
            reward = 100
        else:
            # acceleration = self.noisy_action(action)
            acceleration = action
            state[2:4] = self.update_velocity(state[2:4], acceleration)

        return reward + del_dist, state

    def move_the_car(self, initial_pos, velocity_vector):
        if (velocity_vector[0] <= 0.0001) and (velocity_vector[1] <= 0.0001):
            # print 'moving the car', initial_pos, velocity_vector
            return '', initial_pos

        current_pos = initial_pos
        magnitude = np.linalg.norm(velocity_vector)
        unit_direction = velocity_vector / magnitude
        total_steps = int(round(magnitude / self.step_size))
        step = self.step_size * unit_direction
        # print 'moving the car', initial_pos, velocity_vector, step, total_steps

        for i in range(total_steps):
            current_pos = current_pos + step
            # print current_pos
            xf, yf = (current_pos).astype(int)
            if self.layout.racetrack[xf][yf] == WALL_CELL:
                # print 'WALL_CELL'
                current_pos = current_pos - step
                direction_to_move = self.find_direction_to_move(current_pos, step)
                velocity_vector = np.array(direction_to_move) * step * (total_steps - i)
                _, current_pos = self.move_the_car(current_pos, velocity_vector)
                return 'collision', current_pos
            elif self.layout.racetrack[xf][yf] == FINISH_CELL:
                # print 'FINISH_CELL'
                return 'finish', current_pos
        return '', current_pos

    def find_direction_to_move(self, current_pos, step):
        xi, yi = (current_pos).astype(int)
        final_pos = current_pos + step
        xf, yf = (final_pos).astype(int)
        if self.layout.racetrack[xf][yi] != WALL_CELL:
            # print '[1,0]'
            return [1,0]
        elif self.layout.racetrack[xi][yf] != WALL_CELL:
            # print '[0,1]'
            return [0,1]
        else:
            # print '[0,0]'
            return [0,0]

    def update_velocity(self, velocity, acceleration):
        vxf, vyf = velocity + acceleration
        if vxf > self.max_vel_mag: vxf = self.max_vel_mag
        elif vxf < -self.max_vel_mag: vxf = -self.max_vel_mag

        if vyf > self.max_vel_mag: vyf = self.max_vel_mag
        elif vyf < -self.max_vel_mag: vyf = -self.max_vel_mag

        return [vxf, vyf]

    def get_deacceleration(self, velocity):
        if abs(velocity) <= 1:
            deacceleration = -velocity
        elif velocity > 1:
            deacceleration = -1
        else:
            deacceleration = 1
        return deacceleration

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


################################################################################
##############################UNUSED FUNCTIONS##################################
################################################################################

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
