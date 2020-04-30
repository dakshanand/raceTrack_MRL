
# coding: utf-8

# In[1]:

import math
import numpy as np
import pygame
import matplotlib.pyplot as plt


# In[2]:

ROWS = 200
COLS = 100

WHITE = (255,255,255)
RED = (255, 0, 0)
GREEN = (0, 255 ,0)
BLUE = (0, 0, 255)


# In[1]:

class Generator:

    #HELPFUL FUNCTIONS
    def widen_hole_transformation(self,racetrack,start_cell,end_cell):

        delta = 1
        while(1):
            if ((start_cell[1] < delta) or (start_cell[0] < delta)):
                racetrack[0:end_cell[0],0:end_cell[1]] = -1
                break

            if ((end_cell[1]+delta > COLS) or (end_cell[0]+delta > ROWS)):
                racetrack[start_cell[0]:ROWS,start_cell[1]:COLS] = -1
                break

            delta += 1

        return racetrack

    def calculate_valid_fraction(self, racetrack):
        '''
        Returns the fraction of valid cells in the racetrack
        '''
        return (len(racetrack[racetrack==0])/(ROWS*COLS))

    def mark_finish_states(self, racetrack):
        '''
        Marks finish states in the racetrack
        Returns racetrack
        '''
        last_col = racetrack[0:ROWS,COLS-1]
        last_col[last_col==0] = 2
        return racetrack

    def mark_start_states(self, racetrack):
        '''
        Marks start states in the racetrack
        Returns racetrack
        '''
        last_row = racetrack[ROWS-1,0:COLS]
        last_row[last_row==0] = 1
        return racetrack


    #CONSTRUCTOR
    def __init__(self):
        pass

    def generate_racetrack(self):
        '''
        racetrack is a 2d numpy array
        codes for racetrack:
            0,1,2 : valid racetrack cells
            -1: invalid racetrack cell
            1: start line cells
            2: finish line cells
        returns randomly generated racetrack
        '''
        racetrack = np.zeros((ROWS,COLS),dtype='int')

        frac = 1
        while frac > 0.5:

            #transformation
            random_cell = np.random.randint((ROWS,COLS))
            random_hole_dims = np.random.randint((ROWS//4,COLS//4))
            start_cell = np.array([max(0,x - y//2) for x,y in zip(random_cell,random_hole_dims)])
            end_cell = np.array([min(z,x+y) for x,y,z in zip(start_cell,random_hole_dims,[ROWS,COLS])])

            #apply_transformation
            racetrack = self.widen_hole_transformation(racetrack, start_cell, end_cell)
            frac = self.calculate_valid_fraction(racetrack)

        racetrack = self.mark_start_states(racetrack)
        racetrack = self.mark_finish_states(racetrack)

        return racetrack


# In[4]:

class Data:

    #HELPFUL FUNCTIONS
    def get_start_line(self):
        '''
        Gets start line
        '''
        self.start_line = np.array([np.array([ROWS-1,j]) for j in range(COLS) if self.racetrack[ROWS-1,j] == 1])

    def get_finish_line(self):
        '''
        Gets finish line
        '''
        self.finish_line = np.array([np.array([i,COLS-1]) for i in range(ROWS) if self.racetrack[i,COLS-1] == 2])

    def __init__(self):
        '''
            racetrack: 2 dimensional numpy array
            Q(s,a): 5 dimensional numpy array
            C(s,a): 5 dimensional numpy array
            pi: target policy
            start_line: start_line is the set of start states
            finish_line: finish_line is the set of finish states
            hyperparameters like epsilon
            episode to be an empty list
        '''
        self.load_racetrack()
        self.get_start_line()
        self.get_finish_line()
        self.load_Q_vals()
        self.load_C_vals()
        self.load_pi()
        self.load_rewards()
        self.epsilon = 0.1
        self.gamma = 1
        self.episode = dict({'S':[],'A':[],'probs':[],'R':[None]})

    def save_rewards(self,filename = 'rewards'):
        '''
        saves self.rewards in rewards.npy file
        '''
        self.rewards = np.array(self.rewards)
        np.save(filename,self.rewards)
        self.rewards = list(self.rewards)

    def load_rewards(self):
        '''
        loads rewards from rewards.npy file
        '''
        self.rewards = list(np.load('rewards.npy'))

    def save_pi(self,filename = 'pi.npy'):
        '''
        saves self.pi in pi.npy file
        '''
        np.save(filename,self.pi)

    def load_pi(self):
        '''
        loads pi from pi.npy file
        '''
        self.pi = np.load('pi.npy')

    def save_C_vals(self,filename = 'C_vals.npy'):
        '''
        saves self.C_vals in C_vals.npy file
        '''
        np.save(filename,self.C_vals)

    def load_C_vals(self):
        '''
        loads C_vals from C_vals.npy file
        '''
        self.C_vals = np.load('C_vals.npy')

    def save_Q_vals(self,filename = 'Q_vals.npy'):
        '''
        saves self.Q_vals in Q_vals.npy file
        '''
        np.save(filename,self.Q_vals)

    def load_Q_vals(self):
        '''
        loads Q_vals from Q_vals.npy file
        '''
        self.Q_vals = np.load('Q_vals.npy')

    def save_racetrack(self,filename = 'racetrack.npy'):
        '''
        saves self.racetrack in racetrack.npy file
        '''
        np.save(filename,self.racetrack)

    def load_racetrack(self):
        '''
        loads racetrack from racetrack.npy file
        '''
        self.racetrack = np.load('racetrack.npy')


# In[5]:

class Environment:

    #HELPFUL FUNCTIONS
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
        new_state = self.get_new_state(state, action)
        old_cell, new_cell = state[0:2], new_state[0:2]

        '''
        new_cell's row index will be less
        '''
        rows = np.array(range(new_cell[0],old_cell[0]+1))
        cols = np.array(range(old_cell[1],new_cell[1]+1))
        fin = set([tuple(x) for x in self.data.finish_line])
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

        if new_cell[0] < 0 or new_cell[0] >= ROWS or new_cell[1] < 0 or new_cell[1] >= COLS:
            return True

        else:
            return self.data.racetrack[tuple(new_cell)] == -1

    #CONSTRUCTOR
    def __init__(self, data, gen):
        '''
        initialize step_count to be 0
        '''
        self.data = data
        self.gen = gen
        self.step_count = 0

    #MEMBER FUNCTIONS

    def reset(self):
        self.data.episode = dict({'S':[],'A':[],'probs':[],'R':[None]})
        self.step_count = 0

    def start(self):
        '''
        Makes the velocity of the car to be zero
        Returns the randomly selected start state.
        '''
        state = np.zeros(4,dtype='int')
        state[0] = ROWS-1
        state[1] = self.select_randomly(self.data.start_line[:,1])
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
        self.data.episode['A'].append(action)
        reward = -1

        if (self.is_finish_line_crossed(state, action)):
            new_state = self.get_new_state(state, action)

            self.data.episode['R'].append(reward)
            self.data.episode['S'].append(new_state)
            self.step_count += 1

            return None, new_state

        elif (self.is_out_of_track(state, action)):
            new_state = self.start()
        else:
            new_state = self.get_new_state(state, action)

        self.data.episode['R'].append(reward)
        self.data.episode['S'].append(new_state)
        self.step_count += 1

        return reward, new_state


# In[6]:

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


# In[7]:

class Monte_Carlo_Control:

    #HELPFUL FUNCTIONS

    def evaluate_target_policy(self):
        env.reset()
        state = env.start()
        self.data.episode['S'].append(state)
        rew = -1
        while rew!=None:
            action = agent.get_action(state,self.generate_target_policy_action)
            rew, state = env.step(state,action)

        self.data.rewards.append(sum(self.data.episode['R'][1:]))


    def plot_rewards(self):
        ax, fig = plt.subplots(figsize=(30,15))
        x = np.arange(1,len(self.data.rewards)+1)
        plt.plot(x*10, self.data.rewards, linewidth=0.5, color = '#BB8FCE')
        plt.xlabel('Episode number', size = 20)
        plt.ylabel('Reward',size = 20)
        plt.title('Plot of Reward vs Episode Number',size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig('RewardGraph.png')
        plt.close()

    def save_your_work(self):
        self.data.save_Q_vals()
        self.data.save_C_vals()
        self.data.save_pi()
        self.data.save_rewards()

    def determine_probability_behaviour(self, state, action, possible_actions):
        best_action = self.data.pi[tuple(state)]
        num_actions = len(possible_actions)

        if best_action in possible_actions:
            if action == best_action:
                prob = 1 - self.data.epsilon + self.data.epsilon/num_actions
            else:
                prob = self.data.epsilon/num_actions
        else:
            prob = 1/num_actions

        self.data.episode['probs'].append(prob)

    def generate_target_policy_action(self, state, possible_actions):
        '''
        Returns target policy action, takes state and
        returns an action using this policy
        '''
        if self.data.pi[tuple(state)] in possible_actions:
            action = self.data.pi[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        return action

    def generate_behavioural_policy_action(self, state, possible_actions):
        '''
        Returns behavioural policy action
        which would be epsilon-greedy pi policy, takes state and
        returns an action using this epsilon-greedy pi policy
        '''
        if np.random.rand() > self.data.epsilon and self.data.pi[tuple(state)] in possible_actions:
            action = self.data.pi[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        self.determine_probability_behaviour(state, action, possible_actions)

        return action

    #CONSTRUCTOR
    def __init__(self, data):
        '''
        Initialize, for all s ∈ S, a ∈ A(s):
            data.Q(s, a) ← arbitrary (done in Data)
            data.C(s, a) ← 0 (done in Data)
            pi(s) ← argmax_a Q(s,a)
            (with ties broken consistently)
            (some consistent approach needs to be followed))
        '''
        self.data = data
        for i in range(ROWS):
            for j in range(COLS):
                if self.data.racetrack[i,j]!=-1:
                    for k in range(5):
                        for l in range(5):
                            self.data.pi[i,j,k,l] = np.argmax(self.data.Q_vals[i,j,k,l])

    def control(self,env,agent):
        '''
        Performs MC control using episode list [ S0 , A0 , R1, . . . , ST −1 , AT −1, RT , ST ]
        G ← 0
        W ← 1
        For t = T − 1, T − 2, . . . down to 0:
            G ← gamma*G + R_t+1
            C(St, At ) ← C(St,At ) + W
            Q(St, At ) ← Q(St,At) + (W/C(St,At))*[G − Q(St,At )]
            pi(St) ← argmax_a Q(St,a) (with ties broken consistently)
            If At != pi(St) then exit For loop
            W ← W * (1/b(At|St))
        '''
        env.reset()
        state = env.start()
        self.data.episode['S'].append(state)
        rew = -1
        while rew!=None:
            action = agent.get_action(state,self.generate_behavioural_policy_action)
            rew, state = env.step(state,action)

        G = 0
        W = 1
        T = env.step_count

        for t in range(T-1,-1,-1):
            G = data.gamma * G + self.data.episode['R'][t+1]
            S_t = tuple(self.data.episode['S'][t])
            A_t = agent.map_to_1D(self.data.episode['A'][t])

            S_list = list(S_t)
            S_list.append(A_t)
            SA = tuple(S_list)

            self.data.C_vals[SA] += W
            self.data.Q_vals[SA] += (W*(G-self.data.Q_vals[SA]))/(self.data.C_vals[SA])
            self.data.pi[S_t] = np.argmax(self.data.Q_vals[S_t])
            if A_t!=self.data.pi[S_t]:
                break
            W /= self.data.episode['probs'][t]


# In[8]:

class Visualizer:

    def __init__(self,data):
        self.data = data
        self.window = False

    def setup(self):
        '''
        Does things which occur only at the beginning
        '''
        self.cell_edge = 3
        self.width = COLS*self.cell_edge
        self.height = ROWS*self.cell_edge
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
                if self.data.racetrack[i,j]!=-1:
                    if self.data.racetrack[i,j] == 0:
                        color = RED
                    elif self.data.racetrack[i,j] == 1:
                        color = BLUE
                    elif self.data.racetrack[i,j] == 2:
                        color = GREEN
                    pygame.draw.rect(self.display,color,((j*self.cell_edge,i*self.cell_edge),(self.cell_edge,self.cell_edge)),1)

        if len(state)>0:
            pygame.draw.rect(self.display, WHITE ,((state[1]*self.cell_edge,state[0]*self.cell_edge),(self.cell_edge,self.cell_edge)), 0)

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

    def visualize_episode():
        for i in range(self.data.episode['S']):
            vis.visualize_racetrack(i)

import sys
np.set_printoptions(threshold=sys.maxsize)


# data = Data()
# vis = Visualizer(data)
# print data.racetrack
# while 1:
#     vis.visualize_racetrack()
# vis.visualize_racetrack()

data = Data()
vis = Visualizer(data)
gen = Generator()
env = Environment(data,gen)
mcc = Monte_Carlo_Control(data)
agent = Agent()


env.reset()
state = env.start()

reward = -1
action = (1,1)
while 1:
    if reward == None:
        print "---------------------------------DONE--------------------------------"
        env.reset()
        reward = -1
        state = env.start()
    reward, state = env.step(state, action)
    action = (1,1)
    # print state, reward
    vis.visualize_racetrack(state)
