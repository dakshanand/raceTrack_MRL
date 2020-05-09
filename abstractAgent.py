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
from keras.models import model_from_json

class Agent:

    def map_to_1D(self,action):
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        for i,x in zip(range(9),alpha):
            if action[0]==x[0] and action[1]==x[1]:
                return i

    def map_to_2D(self,action):
        alpha = [(-1,-1),(-1,0),(0,-1),(-1,1),(0,0),(1,-1),(0,1),(1,0),(1,1)]
        return alpha[action]

    def saveModel(self, model, file_name):
        model_json = model.to_json()
        with open('weights/' + file_name + '.json', "w") as json_file:
            json_file.write(model_json)
        model.save_weights('weights/' + file_name + '.h5')

    def loadModel(self, file_name):
        json_file = open('weights/' + file_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights/' + file_name + '.h5')
        return loaded_model

    #CONSTRUCTOR
    def __init__(self, layout):
        self.layout = layout

    def update(self, state, action, nextState, reward):
        pass

    def get_action(self, state):
        '''
        Returns action given state using policy
        '''
        # return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
        # action = np.random.choice(possible_actions)
        # return self.map_to_2D(policy(state, self.possible_actions(state[2:4])))
        pass
