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
from environment import *

identifier = ''

class DQNBaselineAgent(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay = 0.9995
        self.nb_features = 4
        self.nb_actions = 9
        self.discount = .95
        self.Agent = DqnModule(featureExtractor = simpleExtractor, nb_features = self.nb_features, discount = self.discount)
        print '----------'
        print '############ DQNBaselineAgent ############'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print '----------'
        self.last_saved_num = -1


    def getAction(self, state, testing = False):
        if testing and self.training_episode_num > self.last_saved_num:
            self.saveModel(self.Agent.model, 'DQNBaselineAgent_' + identifier + '_' + str(self.training_episode_num))
            self.last_saved_num = self.training_episode_num

        if not testing and (np.random.rand() < self.epsilon):
            action = np.random.randint(self.nb_actions)
            return self.map_to_2D(action)

        qValues = self.Agent.getQValues(state)
        action = np.argmax(qValues)
        return self.map_to_2D(action)

    def update(self, state, action, nextState, reward, done):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
        self.Agent.update(state, self.map_to_1D(action), nextState, reward/50.0, done)

class DqnModule():
    '''
        This class only deals with numerical actions
    '''
    def __init__(self, featureExtractor, nb_features, batch_size = 32, discount = 0.9, nb_actions = 9):
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

class HierarchicalDDPGAgent(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.nb_finishFeatures = 4
        self.nb_collisionFeatures = 4
        self.nb_actions = 9
        self.nb_features = 4
        self.arbitrator_actions = 2
        self.epsilon = 1.
        self.min_epsilon = 0.01
        self.decay = .995
        self.discount = .9
        self.arbitratorDecay = .9995
        self.finishAgent = DqnModule(nb_features = self.nb_finishFeatures, featureExtractor = simpleExtractor, discount = self.discount)
        self.collisionAgent = DqnModule(nb_features = self.nb_collisionFeatures, featureExtractor = simpleExtractor, discount = self.discount)
        self.arbitrator = DDPGModule(nb_features = self.nb_features, featureExtractor = simpleExtractor, nb_actions = self.arbitrator_actions, decay = self.arbitratorDecay)
        # self.subModules = [self.ghostAgent, self.foodAgent, self.puddleAgent]
        self.last_saved_num = -1
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)
        print '----------'
        print '############ HierarchicalDDPGAgent ############'
        print 'Arbitrator Epsilon Decay = %f, Discount Factor = %.2f' % (self.arbitrator.decay, self.discount)
        print '----------'

    def getAction(self, state, testing):
        if testing and self.training_episode_num > self.last_saved_num:
            self.saveModel(self.finishAgent.model, 'ghostAgent_' + identifier + '_' + str(self.training_episode_num))
            self.saveModel(self.collisionAgent.model, 'foodAgent_' + identifier + '_' + str(self.training_episode_num))
            self.saveModel(self.arbitrator.actor_model, 'actor_' + identifier + '_' + str(self.training_episode_num))
            self.saveModel(self.arbitrator.critic_model, 'critic_' + identifier + '_' + str(self.training_episode_num))
            self.last_saved_num = self.training_episode_num

        if not testing and (np.random.rand() < self.epsilon):
            action = np.random.randint(self.nb_actions)
            self.arbitratorAction = [-1, -1]
            return self.map_to_2D(action)
        self.arbitratorAction = self.arbitrator.getAction(state, testing)[0]
        scaleParameters = self.arbitratorAction

        finishQValues = self.finishAgent.getQValues(state)
        collisionQValues = self.collisionAgent.getQValues(state)
        scalarizedQValues = scaleParameters[0] * finishQValues + scaleParameters[1] * collisionQValues

        bestAction = self.map_to_2D(np.argmax(scalarizedQValues))
        return bestAction

    def getFinishReward(self, reward, shapedReward):
        if reward == TIME_STEP_PENALTY + FINISH_REWARD + COLLISION_PENALTY:
            reward = TIME_STEP_PENALTY + FINISH_REWARD
        elif reward == TIME_STEP_PENALTY + COLLISION_PENALTY:
            reward = TIME_STEP_PENALTY
        else:
            pass

        reward += shapedReward

        return reward / 50.0

    def getCollisionReward(self, reward, shapedReward):
        if reward == TIME_STEP_PENALTY + FINISH_REWARD + COLLISION_PENALTY:
            reward = -TIME_STEP_PENALTY + COLLISION_PENALTY
        elif reward == TIME_STEP_PENALTY + COLLISION_PENALTY:
            reward = -TIME_STEP_PENALTY + COLLISION_PENALTY
        else:
            reward = -TIME_STEP_PENALTY

        return reward / 50.0

    def update(self, state, action, nextState, reward, done):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        action = self.map_to_1D(action)
        shapedReward = Environment.getShapedReward(state, nextState)
        if self.arbitratorAction[0] != -1:
            self.arbitrator.update(state, self.arbitratorAction, nextState, float(reward) / 50.0, done)

        self.finishAgent.update(state, action, nextState, self.getFinishReward(reward - shapedReward, shapedReward), done)
        self.collisionAgent.update(state, action, nextState, self.getCollisionReward(reward - shapedReward, shapedReward), done)

class DDPGModule:
    def __init__(self, nb_features, featureExtractor, nb_actions, decay):
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.alpha = 0.0001
        self.epsilon = .9
        self.min_epsilon = .01
        self.gamma = .9
        self.tau   = .01
        self.batch_size = 32
        self.extractor = featureExtractor
        self.nb_features = nb_features
        self.nb_actions = nb_actions
        self.decay = decay
        # print '----------'
        # print '### DDPG Module ###'
        # print 'Epsilon Decay = %s, Discount Factor = %.2f, alpha = %f' % (self.decay, self.gamma, self.alpha)
        # print 'Input Features = %d' % (self.nb_features)
        # print '----------'

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

        self.replay_memory_buffer = deque(maxlen=50000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
	       [None, self.nb_actions]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
        	actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.alpha).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        	self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
        	self.critic_action_input) # where we calcaulte de/dC for feeding above

    	# Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=(self.nb_features,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        output = Dense(self.nb_actions, activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.nb_features,))
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(128)(state_h1)

        action_input = Input(shape=(self.nb_actions,))
        action_h1 = Dense(64)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input,action_input], output=output)

        adam = Adam(lr=self.alpha)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #
    def update(self, state, action, nextState, reward, done):
        if self.alpha < 0.000001:
            return
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
        self.add_to_replay_memory(state, action, reward, nextState, done)
        self.train()
        self.update_target()

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((self.extractor(state),
            action, reward, self.extractor(next_state), done))

    def _train_actor(self, samples):

        cur_states, actions, rewards, new_states, _ =  self.get_attributes_from_sample(samples)
        predicted_actions = self.actor_model.predict(cur_states)
        grads = self.sess.run(self.critic_grads, feed_dict={
        	self.critic_state_input:  cur_states,
        	self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
        	self.actor_state_input: cur_states,
        	self.actor_critic_grad: grads
        })

    def _train_critic(self, samples):

        cur_states, actions, rewards, new_states, dones = self.get_attributes_from_sample(samples)
        target_actions = self.target_actor_model.predict(new_states)
        future_rewards = self.target_critic_model.predict([new_states, target_actions])

        rewards += self.gamma * future_rewards * (1 - dones)

        evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
        #print(evaluation.history)

    def train(self):
        if len(self.replay_memory_buffer) < self.batch_size:
            return

        rewards = []
        samples = random.sample(self.replay_memory_buffer, self.batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def get_attributes_from_sample(self, random_sample):
        array = np.array(random_sample)

        current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
        actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
        rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
        new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
        dones = np.stack(array[:,4]).reshape((array.shape[0],-1))

        return current_states, actions, rewards, new_states, dones

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def getAction(self, state, testing):
        state = self.extractor(state).reshape((1, self.nb_features))
        if not testing and (np.random.random() < self.epsilon):
            noise = np.random.uniform(size = self.nb_actions)
            clipped_noise = np.clip(noise, -1.0, 1.0)
            action = self.actor_model.predict(state) + clipped_noise
            return action
        action = self.actor_model.predict(state)
        return action

class GmQAgent(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.nb_finishFeatures = 4
        self.nb_collisionFeatures = 4
        self.nb_actions = 9
        self.nb_features = 4
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = .999
        self.discount = .9
        self.finishAgent = DqnModule(nb_features = self.nb_finishFeatures, featureExtractor = simpleExtractor, discount = self.discount)
        self.collisionAgent = DqnModule(nb_features = self.nb_collisionFeatures, featureExtractor = simpleExtractor, discount = self.discount)
        # self.subModules = [self.ghostAgent, self.foodAgent, self.puddleAgent]
        self.last_saved_num = -1
        # self.foodAgent.model = self.loadModel(name)
        # self.ghostAgent.model = self.loadModel(name)

        print '----------'
        print '############ GmQAgent ############'
        print 'Epsilon Decay = %s, Discount Factor = %.2f' % (self.decay, self.discount)
        print '----------'
        self.last_saved_num = -1

    def getAction(self, state, testing):
        if testing and self.training_episode_num > self.last_saved_num:
            self.saveModel(self.finishAgent.model, 'ghostAgent_' + identifier + '_' + str(self.training_episode_num))
            self.saveModel(self.collisionAgent.model, 'foodAgent_' + identifier + '_' + str(self.training_episode_num))
            self.last_saved_num = self.training_episode_num

        if not testing and (np.random.rand() < self.epsilon):
            action = np.random.randint(self.nb_actions)
            return self.map_to_2D(action)

        finishQValues = self.finishAgent.getQValues(state)
        collisionQValues = self.collisionAgent.getQValues(state)
        finalQValues = finishQValues + collisionQValues

        bestAction = self.map_to_2D(np.argmax(finalQValues))
        return bestAction

    def getFinishReward(self, reward, shapedReward):
        if reward == TIME_STEP_PENALTY + FINISH_REWARD + COLLISION_PENALTY:
            reward = TIME_STEP_PENALTY + FINISH_REWARD
        elif reward == TIME_STEP_PENALTY + COLLISION_PENALTY:
            reward = TIME_STEP_PENALTY
        else:
            pass

        reward += shapedReward

        return reward / 50.0

    def getCollisionReward(self, reward, shapedReward):

        if reward == TIME_STEP_PENALTY + FINISH_REWARD + COLLISION_PENALTY:
            reward = -TIME_STEP_PENALTY + COLLISION_PENALTY
        elif reward == TIME_STEP_PENALTY + COLLISION_PENALTY:
            reward = -TIME_STEP_PENALTY + COLLISION_PENALTY
        else:
            reward = -TIME_STEP_PENALTY

        return reward / 50.0

    def update(self, state, action, nextState, reward, done):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.decay

        action = self.map_to_1D(action)
        shapedReward = Environment.getShapedReward(state, nextState)
        self.finishAgent.update(state, action, nextState, self.getFinishReward(reward - shapedReward, shapedReward), done)
        self.collisionAgent.update(state, action, nextState, self.getCollisionReward(reward - shapedReward, shapedReward), done)

class TestingAgent(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.finishAgent = self.loadModel('')
        self.collisionAgent = self.loadModel('')

    def computeActionFromQValues(self, state):
        finishFeatures = simpleExtractor(state)
        qValues1 = self.finishAgent.predict(np.array([finishFeatures]), batch_size=1)[0]
        collisionFeatures = simpleExtractor(state)
        qValues2 = self.collisionAgent.predict(np.array([collisionFeatures]), batch_size=1)[0]
        qValues = (qValues1 + qValues2)
        bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
        print qValues, bestAction
        return bestAction

    def update(self, state, action, nextState, reward):
        print reward
        return

class TestingAgentDDPG(Agent):
    def __init__(self, **args):
        Agent.__init__(self, **args)
        self.actor = self.loadModel('actor_3_1000')

        self.finishAgent = self.loadModel('')
        self.collisionAgent = self.loadModel('')

    def computeActionFromQValues(self, state):

        # ddpgState = CustomizedExtractor().getFeatures(state).reshape((1, 26))
        ddpgState = simpleExtractor(state)

        scaleParameters = self.actor.predict(ddpgState)[0]
        finishFeatures = simpleExtractor(state)
        finishQValues = self.finishAgent.predict(np.array([finishFeatures]), batch_size=1)[0]
        collisionFeatures = simpleExtractor(state)
        collisionQValues = self.collisionAgent.predict(np.array([collisionFeatures]), batch_size=1)[0]
        qValues = scaleParameters[0] * finishQValues + scaleParameters[1] * collisionQValues

        bestAction = ActionMapping.NumbertoAction[np.argmax(qValues)]
        return bestAction

    def update(self, state, action, nextState, reward):
        return
