import tensorflow as tf
import os
import numpy as np
from SimpleQAgent.networks.deep_q_network import DeepQNetwork, DuelingDQNetwork
from SimpleQAgent.agents.base_agent import BaseAgent
from SimpleQAgent.memory.replay_memory import ReplayMemory

class DQNAgent(BaseAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, mem_size, batch_size, 
                 eps_min = 0.01, eps_decay = 5e-7, replace = 1000, algo = None, 
                 env_name = None, ckpt_dir = "tmp/dqn", net_type = "linear"):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.ckpt_dir = ckpt_dir
        self.net_type = net_type
        self.action_space = [i for i in range(self.n_actions)]
        
        self.memory = ReplayMemory(mem_size = mem_size)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.learn_step_counter = 0

        self.q_main = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                   n_actions = self.n_actions,
                                   name = self.env_name + "_" + self.algo + "_dq_main",
                                   ckpt_dir = self.ckpt_dir)

        self.q_target = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                     n_actions = self.n_actions,
                                     name = self.env_name + "_" + self.algo + "_dq_target",
                                     ckpt_dir = self.ckpt_dir)

    def learn(self):
        for i in range(3):
            if len(self.memory) < self.batch_size:
                return

            self.update_target_network()

            states, actions, rewards, next_states, done = self.sample_memory()

            with tf.GradientTape() as tape:

                action_masks = tf.one_hot(actions, self.n_actions)

                q_pred = self.q_main(states)
                q_pred = tf.reduce_sum(action_masks * q_pred, axis = -1)

                q_next = self.q_target(next_states)

                q_target = rewards + self.gamma * ((1 - done)) * tf.reduce_max(q_next, axis = -1)

                loss = self.q_main.loss(q_target, q_pred)

            grads = tape.gradient(loss, self.q_main.trainable_variables)
            self.q_main.optimizer.apply_gradients(zip(grads, self.q_main.trainable_variables))

        self.learn_step_counter += 1

        self.decrement_epsilon()



class DDQNAgent(BaseAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, mem_size, batch_size, 
                 eps_min = 0.01, eps_decay = 5e-7, replace = 1000, algo = None, 
                 env_name = None, ckpt_dir = "tmp/ddqn", net_type = "linear"):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.ckpt_dir = ckpt_dir
        self.net_type = net_type
        self.action_space = [i for i in range(self.n_actions)]
        
        self.memory = ReplayMemory(mem_size = mem_size)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.learn_step_counter = 0

        self.q_main = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                   n_actions = self.n_actions,
                                   name = self.env_name + "_" + self.algo + "_ddq_main",
                                   ckpt_dir = self.ckpt_dir)

        self.q_target = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                     n_actions = self.n_actions,
                                     name = self.env_name + "_" + self.algo + "_ddq_target",
                                     ckpt_dir = self.ckpt_dir)

    def learn(self):
        for i in range(3):
            if len(self.memory) < self.batch_size:
                return

            self.update_target_network()

            states, actions, rewards, next_states, done = self.sample_memory()

            with tf.GradientTape() as tape:

                action_masks = tf.one_hot(actions, self.n_actions)

                q_pred = self.q_main(states)
                q_pred = tf.reduce_sum(action_masks * q_pred, axis = -1)

                q_next = self.q_target(next_states)
                q_eval = self.q_main(next_states)

                max_actions = tf.argmax(q_eval, axis = -1)
                max_actions_masks = tf.one_hot(max_actions, self.n_actions)

                q_target = rewards + self.gamma * ((1 - done)) * tf.reduce_sum(
                        max_actions_masks * q_next, axis = -1)

                loss = self.q_main.loss(q_target, q_pred)

            grads = tape.gradient(loss, self.q_main.trainable_variables)
            self.q_main.optimizer.apply_gradients(zip(grads, self.q_main.trainable_variables))

        self.learn_step_counter += 1

        self.decrement_epsilon()


class DuelingDQNAgent(BaseAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, mem_size, batch_size, 
                 eps_min = 0.01, eps_decay = 5e-7, replace = 1000, algo = None, 
                 env_name = None, ckpt_dir = "tmp/duelingdqn", net_type = "linear"):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.ckpt_dir = ckpt_dir
        self.net_type = net_type
        self.dueling = True
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayMemory(mem_size = mem_size)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.learn_step_counter = 0

        self.q_main = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                       n_actions = self.n_actions,
                                       name = self.env_name + "_" + self.algo + "_dueling_dq_main",
                                       ckpt_dir = self.ckpt_dir)

        self.q_target = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                         n_actions = self.n_actions,
                                         name = self.env_name + "_" + self.algo + "_dueling_dq_target",
                                         ckpt_dir = self.ckpt_dir)

    def learn(self):
        for i in range(3):
            if len(self.memory) < self.batch_size:
                return

            self.update_target_network()

            states, actions, rewards, next_states, done = self.sample_memory()

            with tf.GradientTape() as tape:

                action_masks = tf.one_hot(actions, self.n_actions)

                q_pred = self.q_main.get_q_value(states)
                q_pred = tf.reduce_sum(action_masks * q_pred, axis = -1)

                q_next = self.q_target.get_q_value(next_states)

                q_target = rewards + self.gamma * ((1 - done)) * tf.reduce_max(
                        q_next, axis = -1)

                loss = self.q_main.loss(q_target, q_pred)

            grads = tape.gradient(loss, self.q_main.trainable_variables)
            self.q_main.optimizer.apply_gradients(zip(grads, self.q_main.trainable_variables))

        self.learn_step_counter += 1

        self.decrement_epsilon()


class D3QNAgent(BaseAgent):
    def __init__(self, gamma, epsilon, lr, n_actions, mem_size, batch_size, 
                 eps_min = 0.01, eps_decay = 5e-7, replace = 1000, algo = None, 
                 env_name = None, ckpt_dir = "tmp/d3qn", net_type = "linear"):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.ckpt_dir = ckpt_dir
        self.net_type = net_type
        self.dueling = True
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayMemory(mem_size = mem_size)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.learn_step_counter = 0

        self.q_main = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                       n_actions = self.n_actions,
                                       name = self.env_name + "_" + self.algo + "_d3q_main",
                                       ckpt_dir = self.ckpt_dir)

        self.q_target = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                         n_actions = self.n_actions,
                                         name = self.env_name + "_" + self.algo + "_d3q_target",
                                         ckpt_dir = self.ckpt_dir)

    def learn(self):
        for i in range(3):
            if len(self.memory) < self.batch_size:
                return

            self.update_target_network()

            states, actions, rewards, next_states, done = self.sample_memory()

            with tf.GradientTape() as tape:

                action_masks = tf.one_hot(actions, self.n_actions)

                q_pred = self.q_main.get_q_value(states)
                q_pred = tf.reduce_sum(action_masks * q_pred, axis = -1)

                q_next = self.q_target.get_q_value(next_states)
                q_eval = self.q_main.get_q_value(next_states)

                max_actions = tf.argmax(q_eval, axis = -1)
                max_action_masks = tf.one_hot(max_actions, self.n_actions)

                q_target = rewards + self.gamma * ((1 - done)) * tf.reduce_sum(
                        max_action_masks * q_next, axis = -1)

                loss = self.q_main.loss(q_target, q_pred)

            grads = tape.gradient(loss, self.q_main.trainable_variables)
            self.q_main.optimizer.apply_gradients(zip(grads, self.q_main.trainable_variables))

        self.learn_step_counter += 1

        self.decrement_epsilon()

