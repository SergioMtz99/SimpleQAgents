import tensorflow as tf
import numpy as np

class BaseAgent():
    def __init__(self):

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = tf.convert_to_tensor([observation], dtype = tf.float32)

            if self.dueling:
                advantage, value = self.q_main(state)
                action = tf.argmax(advantage[0]).numpy()
            else:
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
