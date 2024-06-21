import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from SimpleQAgent.networks.networks import LinearNetwork, ConvNetwork
from SimpleQAgent.networks.networks import DuelingConvNetwork, DuelingLinearNetwork

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
            self.network = LinearNetwork(self.n_actions)
        else:
            self.network = ConvNetwork(self.n_actions)
        
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
            self.network = DuelingLinearNetwork(self.n_actions)
        else:
            self.network = DuelingConvNetwork(self.n_actions)
        
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
