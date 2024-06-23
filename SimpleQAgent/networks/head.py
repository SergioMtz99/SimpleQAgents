import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

class QHead(Layer):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def build(self, input_shape):
        self.fc_q = Dense(units = self.n_actions)

        super().build(input_shape)

    def call(self, x):
        Q = self.fc_q(x)

        return Q

class DuelingHead(Layer):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def build(self, input_shape):
        self.fc_a = Dense(units = self.n_actions)
        self.fc_v = Dense(units = 1)

        super().build(input_shape)

    def call(self, x):
        advantage = self.fc_a(x)
        value = self.fc_v(x)

        return advantage, value
