# coding: utf-8

import math
import random
import numpy as np

import layout_parser
from variables import *
from visualizer import Visualizer
from environment import Environment

layout_name = 'f1'
layout = layout_parser.getLayout( layout_name )
# print layout.racetrack
# print layout
vis = Visualizer(layout)
env = Environment(layout)

env.reset()
state = env.start()
print "start_state", state


################################################################################
################################################################################

done = False
reward = -1
action = (1,1)
while 1:
    if done:
        print "----------------------------DONE----------------------------"
        env.reset()
        reward = -1
        action = (1,1)
        state = env.start()
    state, reward, done = env.step(state, action)
    alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
    action = random.choice(alpha)
    # action = (1,1)
    # print 'state', state, 'action', action
    for i in range(100000): pass
    vis.visualize_racetrack(state)

################################################################################
################################################################################

# from qLearningAgents import *
# agent = DQNBaselineAgent()
# reward = -1
# action = (1,1)
# while 1:
#     if reward == None:
#         print "---------------------------------DONE--------------------------------"
#         env.reset()
#         reward = -1
#         state = env.start()
#     action = agent.getAction(state)
#     print state, action
#     newState, reward, done = env.step(state, action)
#     agent.update(state, action, newState, reward, done)
#     state = newState
#     vis.visualize_racetrack(state)
