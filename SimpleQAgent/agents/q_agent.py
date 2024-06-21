import numpy as np
import tensorflow as tf
from SimpleQAgent.networks.deep_q_network import DeepQNetwork, DuelingDQNetwork
from SimpleQAgent.memory.replay_memory import ReplayMemory
import os

class DQNAgent():
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

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size = mem_size)

        self.q_main = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                   n_actions = self.n_actions,
                                   name = self.env_name + "_" + self.algo + "_dq_main",
                                   ckpt_dir = self.ckpt_dir)

        self.q_target = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                     n_actions = self.n_actions,
                                     name = self.env_name + "_" + self.algo + "_dq_target",
                                     ckpt_dir = self.ckpt_dir)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([observation], dtype = tf.float32)
            actions = self.q_main(state)
            action = tf.argmax(actions[0]).numpy()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = \
                self.memory.sample_memory(self.batch_size)

        states = tf.convert_to_tensor(state, dtype = tf.float32)
        actions = tf.convert_to_tensor(action, dtype = tf.int32)
        rewards = tf.convert_to_tensor(reward, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype = tf.float32)
        dones = tf.convert_to_tensor(done, dtype = tf.float32)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.set_weights(self.q_main.get_weights())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decay
        else:
            self.epsilon = self.eps_min

    def save_models(self):
        self.q_main.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_main.load_checkpoint()
        self.q_target.load_checkpoint()

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



class DDQNAgent():
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

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size = mem_size)

        self.q_main = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                   n_actions = self.n_actions,
                                   name = self.env_name + "_" + self.algo + "_ddq_main",
                                   ckpt_dir = self.ckpt_dir)

        self.q_target = DeepQNetwork(net_type = self.net_type, lr = self.lr,
                                     n_actions = self.n_actions,
                                     name = self.env_name + "_" + self.algo + "_ddq_target",
                                     ckpt_dir = self.ckpt_dir)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([observation], dtype = tf.float32)
            actions = self.q_main(state)
            action = tf.argmax(actions[0]).numpy()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = \
                self.memory.sample_memory(self.batch_size)

        states = tf.convert_to_tensor(state, dtype = tf.float32)
        actions = tf.convert_to_tensor(action, dtype = tf.int32)
        rewards = tf.convert_to_tensor(reward, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype = tf.float32)
        dones = tf.convert_to_tensor(done, dtype = tf.float32)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.set_weights(self.q_main.get_weights())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decay
        else:
            self.epsilon = self.eps_min

    def save_models(self):
        self.q_main.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_main.load_checkpoint()
        self.q_target.load_checkpoint()

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


class DuelingDQNAgent():
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
        self.action_space = [i for i in range(self.n_actions)]

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size = mem_size)

        self.q_main = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                       n_actions = self.n_actions,
                                       name = self.env_name + "_" + self.algo + "_dueling_dq_main",
                                       ckpt_dir = self.ckpt_dir)

        self.q_target = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                         n_actions = self.n_actions,
                                         name = self.env_name + "_" + self.algo + "_dueling_dq_target",
                                         ckpt_dir = self.ckpt_dir)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([observation], dtype = tf.float32)
            advantage, value = self.q_main(state)
            action = tf.argmax(advantage).numpy()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = \
                self.memory.sample_memory(self.batch_size)

        states = tf.convert_to_tensor(state, dtype = tf.float32)
        actions = tf.convert_to_tensor(action, dtype = tf.int32)
        rewards = tf.convert_to_tensor(reward, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype = tf.float32)
        dones = tf.convert_to_tensor(done, dtype = tf.float32)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.set_weights(self.q_main.get_weights())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decay
        else:
            self.epsilon = self.eps_min

    def save_models(self):
        self.q_main.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_main.load_checkpoint()
        self.q_target.load_checkpoint()

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


class D3QNAgent():
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
        self.action_space = [i for i in range(self.n_actions)]

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            
        self.learn_step_counter = 0

        self.memory = ReplayMemory(mem_size = mem_size)

        self.q_main = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                       n_actions = self.n_actions,
                                       name = self.env_name + "_" + self.algo + "_d3q_main",
                                       ckpt_dir = self.ckpt_dir)

        self.q_target = DuelingDQNetwork(net_type = self.net_type, lr = self.lr,
                                         n_actions = self.n_actions,
                                         name = self.env_name + "_" + self.algo + "_d3q_target",
                                         ckpt_dir = self.ckpt_dir)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([observation], dtype = tf.float32)
            advantage, value = self.q_main(state)
            action = tf.argmax(advantage).numpy()

        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = \
                self.memory.sample_memory(self.batch_size)

        states = tf.convert_to_tensor(state, dtype = tf.float32)
        actions = tf.convert_to_tensor(action, dtype = tf.int32)
        rewards = tf.convert_to_tensor(reward, dtype = tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype = tf.float32)
        dones = tf.convert_to_tensor(done, dtype = tf.float32)

        return states, actions, rewards, next_states, dones

    def update_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.set_weights(self.q_main.get_weights())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decay
        else:
            self.epsilon = self.eps_min

    def save_models(self):
        self.q_main.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_main.load_checkpoint()
        self.q_target.load_checkpoint()

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

