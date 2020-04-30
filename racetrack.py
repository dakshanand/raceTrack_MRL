
# coding: utf-8

# In[1]:

import math
import numpy as np
import pygame
import matplotlib.pyplot as plt
import sys, random
np.set_printoptions(threshold=sys.maxsize)

import layout


WHITE = (255,255,255)
RED = (255, 0, 0)
GREEN = (0, 255 ,0)
BLUE = (0, 0, 255)


class Environment:

    def __init__(self, data, layout2):
        '''
        initialize step_count to be 0
        '''
        self.data = data
        self.layout = layout2
        self.step_count = 0

    def start(self):
        '''
        Makes the velocity of the car to be zero
        Returns the randomly selected start state.
        '''
        state = np.zeros(4,dtype='int')
        # state[0] = ROWS-1
        # state[1] = self.select_randomly(self.data.start_line[:,1])

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
        # self.data.episode['A'].append(action)
        reward = -1

        if (self.is_finish_line_crossed(state, action)):
            new_state = self.get_new_state(state, action)

            # self.data.episode['R'].append(reward)
            # self.data.episode['S'].append(new_state)
            self.step_count += 1

            return None, new_state

        elif (self.is_out_of_track(state, action)):
            new_state = self.start()
        else:
            new_state = self.get_new_state(state, action)

        # self.data.episode['R'].append(reward)
        # self.data.episode['S'].append(new_state)
        self.step_count += 1

        return reward, new_state

    def is_out_of_track(self, state, action):
        '''
        Returns True if the car goes out of track if action is taken on state
                False otherwise
        '''
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        if new_cell[0] < 0 or new_cell[0] >= ROWS or new_cell[1] < 0 or new_cell[1] >= COLS:
            return True

        else:
            # return self.data.racetrack[tuple(new_cell)] == -1
            return self.layout.racetrack[new_cell[0]][new_cell[1]] == -1

    #MEMBER FUNCTIONS

    def reset(self):
        # self.data.episode = dict({'S':[],'A':[],'probs':[],'R':[None]})
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
        # fin = set([tuple(x) for x in self.data.finish_line])
        fin = set([tuple(x) for x in self.layout.finishStates])
        row_col_matrix = [(x,y) for x in rows for y in cols]
        intersect = [x for x in row_col_matrix if x in fin]

        return len(intersect) > 0


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


class Visualizer:

    def __init__(self,data):
        self.data = data
        self.window = False

    def setup(self):
        '''
        Does things which occur only at the beginning
        '''
        self.cell_edge = 7
        self.width = ROWS * self.cell_edge
        self.height = COLS * self.cell_edge
        self.blockSize = (self.cell_edge,self.cell_edge)
        self.create_window()
        self.window = True

    def create_window(self):
        '''
        Creates window and assigns self.display variable
        '''
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack")

    def close_window(self):
        self.window = False
        pygame.quit()

    def draw(self, state = np.array([])):
        self.display.fill(0)
        for i in range(ROWS):
            for j in range(COLS):
                if self.data.racetrack[i][j] != -1:
                    if self.data.racetrack[i][j] == 1:
                        color = GREEN
                    elif self.data.racetrack[i][j] == 2:
                        color = RED
                    elif self.data.racetrack[i][j] == 3:
                        color = BLUE
                    pygame.draw.rect(self.display, color, ((i*self.cell_edge,j*self.cell_edge), self.blockSize), 1)

        if len(state)>0:
            pygame.draw.rect(self.display, WHITE ,((state[0]*self.cell_edge, state[1]*self.cell_edge), self.blockSize), 0)

        pygame.display.update()

        global count

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
                self.close_window()
                return 'stop'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                pygame.image.save(vis.display, str(count)+'.png')
                count += 1
                self.loop = False

        return None

    def visualize_racetrack(self, state = np.array([])):
        '''
        Draws Racetrack in a pygame window
        '''
        if self.window == False:
            self.setup()
        self.loop = True
        while(self.loop):
            ret = self.draw(state)
            break
            if ret!=None:
                return ret

    # def visualize_episode():
    #     for i in range(self.data.episode['S']):
    #         vis.visualize_racetrack(i)

################################################################################
layout_name = 'f1'
layout2 = layout.getLayout( layout_name )
print layout2
ROWS = layout2.racetrack.width
COLS = layout2.racetrack.height

################################################################################


# data = Data()
# vis = Visualizer(data)
# print data.racetrack
# while 1:
#     vis.visualize_racetrack()
# vis.visualize_racetrack()

data = []
vis = Visualizer(layout2)
env = Environment(data,layout2)
agent = Agent()


env.reset()
state = env.start()


print state
print ROWS, COLS


reward = -1
action = (0, 0)
action = (1,1)
while 1:
    if reward == None:
        print "---------------------------------DONE--------------------------------"
        env.reset()
        reward = -1
        state = env.start()
    reward, state = env.step(state, action)
    action = (0, 0)
    action = (1,1)
    # print state, reward
    vis.visualize_racetrack(state)
