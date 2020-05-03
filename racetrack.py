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


# ################################################################################
# ################################################################################
#
# done = False
# reward = -1
# action = (1,1)
# while 1:
#     if done:
#         print "----------------------------DONE----------------------------"
#         env.reset()
#         reward = -1
#         action = (1,1)
#         state = env.start()
#     state, reward, done = env.step(state, action)
#     alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
#     action = random.choice(alpha)
#     # action = (1,1)
#     # print 'state', state, 'action', action
#     for i in range(100000): pass
#     vis.visualize_racetrack(state)
#
# ################################################################################
# ################################################################################
import time
from qLearningAgents import *
# agent = DQNBaselineAgent()
agent = HierarchicalDDPGAgent()
reward = -1
action = (1,1)
numEpisodes = 1000
totalSteps = 0
episodeSteps = 0
trainingRewards = []

for episode in range(numEpisodes):
    totalSteps += episodeSteps
    episodeReward = 0
    episodeSteps = 0
    while 1:
        episodeReward += reward
        episodeSteps += 1

        action = agent.getAction(state)
        # print state, action
        nextState, reward, done = env.step(state, action)
        shapedReward = env.getShapedReward(state, nextState)
        agent.update(state, action, nextState, (reward + shapedReward), done)

        if done or episodeSteps > 200:
            if done:
                print "---------------------------------DONE--------------------------------"
            else:
                print "-------------------------------CRASHED-------------------------------"
            env.reset()
            reward = -1
            state = env.start()
            trainingRewards.append(episodeReward)
            if episode % 10 == 0 and episode:
                print trainingRewards, totalSteps
            # time.sleep(2)
            break
        # if episode > 100:
        #     print state, nextState, action, reward
        state = nextState
#            vis.visualize_racetrack(state)
