# coding: utf-8

import math
import numpy as np

import layout_parser
from variables import *
from visualizer import Visualizer
from environment import Environment
# from qLearningAgents import *

layout_name = 'f1'
layout = layout_parser.getLayout( layout_name )
# print layout.racetrack
print layout
vis = Visualizer(layout)
env = Environment(layout)

env.reset()
state = env.start()
print "start_state", state


################################################################################
################################################################################

reward = -1
action = (1,1)
while 1:
    if reward == None:
        print "----------------------------DONE----------------------------"
        env.reset()
        reward = -1
        action = (1,1)
        state = env.start()
    reward, state = env.step(state, action)
    action = (1,1)
    print 'state', state, 'action', action
    # vis.visualize_racetrack(state)

################################################################################
################################################################################

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
#     reward, newState = env.step(state, action)
#     agent.update(state, action, newState, reward)
#     state = newState
#     vis.visualize_racetrack(state)
