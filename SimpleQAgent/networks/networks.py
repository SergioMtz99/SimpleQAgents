from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Model

class LinearNetwork(Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def build(self, input_shape):

        self.fc1 = Dense(units = 512, activation = "relu")

        self.fc2 = Dense(units = 256, activation = "relu")

        self.fc3 = Dense(units = self.n_actions)

        super().build(input_shape)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class ConvNetwork(Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def build(self, input_shape):

        self.conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, 
                            padding = "same", activation = "relu")

        self.conv2 = Conv2D(filters = 64, kernel_size = 4, strides = 2,
                            padding = "same", activation = "relu")

        self.conv3 = Conv2D(filters = 64, kernel_size = 4, strides = 1,
                            padding = "same", activation = "relu")

        self.flatten = Flatten()

        self.fc1 = Dense(units = 512, activation = "relu")
        self.fc2 = Dense(units = self.n_actions)

        super().build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.fc1(x)

        return self.fc2(x)


class DuelingLinearNetwork(Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def build(self, input_shape):

        self.fc1 = Dense(units = 512, activation = "relu")

        self.fc2 = Dense(units = 256, activation = "relu")

        self.fc_a = Dense(units = self.n_actions)
        self.fc_v = Dense(units = 1)

        super().build(input_shape)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        advantage = self.fc_a(x)
        value = self.fc_v(x)

        return advantage, value


class DuelingConvNetwork(Model):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions

    def build(self, input_shape):

        self.conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 2, 
                            padding = "same", activation = "relu")

        self.conv2 = Conv2D(filters = 64, kernel_size = 4, strides = 2,
                            padding = "same", activation = "relu")

        self.conv3 = Conv2D(filters = 64, kernel_size = 4, strides = 1,
                            padding = "same", activation = "relu")

        self.flatten = Flatten()

        self.fc1 = Dense(units = 512, activation = "relu")

        self.fc_a = Dense(units = self.n_actions)
        self.fc_v = Dense(units = 1)

        super().build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.fc1(x)

        advantage = self.fc_a(x)
        value = self.fc_v(x)

        return advantage, value

