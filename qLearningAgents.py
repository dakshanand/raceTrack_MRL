# qLearningAgents.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Vaibhav Gupta

import tensorflow as tf
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import Add, Concatenate
import keras.backend as K

from collections import deque
import numpy as np
import random,util,math

from abstractAgent import Agent
from featureExtractors import *

class DQNBaselineAgent(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay = 0.999
        self.nb_features = 4
        self.nb_actions = 9
        self.Agent = DqnModule(featureExtractor = simpleExtractor, nb_features = self.nb_features)

    def getAction(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(9)
            return self.map_to_2D(action)

        qValues = self.Agent.getQValues(state)
        action = np.argmax(qValues)
        return self.map_to_2D(action)

    def update(self, state, action, nextState, reward):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
        self.Agent.update(state, self.map_to_1D(action), nextState, reward, done = 0)

class DqnModule():
    '''
        This class only deals with numerical actions
    '''
    def __init__(self, featureExtractor, nb_features, batch_size = 32, discount = 0.8, nb_actions = 9):
        self.batch_size = batch_size
        self.discount = discount
        self.nb_actions = nb_actions
        self.nb_features = nb_features
        self.replay_memory_buffer = deque(maxlen=50000)
        self.extractor = featureExtractor
        self.createModel()

    def createModel(self):
        self.model = Sequential()
        self.model.add(Dense(64, init='lecun_uniform', input_shape=(self.nb_features,)))
        self.model.add(Activation('relu'))

        self.model.add(Dense(64, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(32, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.nb_actions, init='lecun_uniform'))
        self.model.add(Activation('linear'))

        # rms = RMSprop(lr=0.000001, rho=0.6)
        adamOptimizer = Adam(lr=0.001)

        self.model.compile(loss='mse', optimizer=adamOptimizer)

    # def getQValue(self, state, action):
    #     qValues = self.model.predict(self.extractor(state))
    #     return qValues[action]
    #
    # def getAction(self, state, legalActions):
    #     qValues = self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]
    #     maxQ, bestAction = float('-inf'), None
    #     for action in legalActions:
    #         if qValues[action] > maxQ:
    #             maxQ, bestAction = qValues[action], action
    #     return bestAction

    def getQValues(self, state):
        return self.model.predict(np.array([self.extractor(state)]), batch_size=1)[0]

    def update(self, state, action, nextState, reward, done):
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.replayExperience()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.extractor(state),
            action, reward, self.extractor(next_state), done))

    def replayExperience(self):
        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        random_sample = self.get_random_sample_from_replay_mem()
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.discount * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])
        target_vec[[indexes], [actions]] = targets

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_random_sample_from_replay_mem(self):
        return random.sample(self.replay_memory_buffer, self.batch_size)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list
