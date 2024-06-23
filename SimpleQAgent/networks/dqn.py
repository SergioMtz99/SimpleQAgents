import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from SimpleQAgent.networks.base import LinearBase, ConvBase
from SimpleQAgent.networks.head import QHead, DuelingHead

class DeepQNetwork(Model):
    def __init__(self, net_type, lr, n_actions, name, 
                 ckpt_dir):
        super().__init__()
        self.net_type = net_type
        self.lr = lr
        self.n_actions = n_actions
        self.checkpoint_dir = ckpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        if self.net_type == "linear":
            self.network = Sequential([
                LinearBase(),
                QHead(n_actions = self.n_actions)])
        else:
            self.network = Sequential([
                ConvBase(),
                QHead(n_actions = self.n_actions)])
        
        self.optimizer = Adam(learning_rate = self.lr)
        self.loss = MeanSquaredError()

    def call(self, state):
        actions = self.network(state)

        return actions

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        self.save_weights(f"{self.checkpoint_file}.weights.h5")

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_weights(f"{self.checkpoint_file}.weights.h5")


class DuelingDQNetwork(Model):
    def __init__(self, net_type, lr, n_actions, name, 
                 ckpt_dir):
        super().__init__()
        self.net_type = net_type
        self.lr = lr
        self.n_actions = n_actions
        self.checkpoint_dir = ckpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        if self.net_type == "linear":
            self.network = Sequential([
                LinearBase(),
                DuelingHead(n_actions = self.n_actions)])
        else:
            self.network = Sequential([
                ConvBase(),
                DuelingHead(n_actions = self.n_actions)])
        
        self.optimizer = Adam(learning_rate = self.lr)
        self.loss = MeanSquaredError()

    def call(self, state):
        advantage, value = self.network(state)

        return advantage, value

    def get_q_value(self, state):
        advantage, value = self.call(state)

        Q = tf.add(value,
                   (advantage - tf.reduce_mean(advantage, axis = 1, keepdims = True)))

        return Q

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        self.save_weights(f"{self.checkpoint_file}.weights.h5")

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_weights(f"{self.checkpoint_file}.weights.h5")
