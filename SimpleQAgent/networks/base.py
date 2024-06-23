from tensorflow.keras.layers import Layer, Dense, Conv2D, Flatten

class LinearBase(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.fc1 = Dense(units = 512, activation = "relu")
        self.fc2 = Dense(units = 256, activation = "relu")

        super().build(input_shape)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)

        return x


class ConvBase(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.conv1 = Conv2D(filters = 32, kernel_size = 8, strides = 4,
                            activation = "relu")
        self.conv2 = Conv2D(filters = 64, kernel_size = 4, strides = 2,
                            activation = "relu")
        self.conv3 = Conv2D(filters = 64, kernel_size = 3, strides = 1,
                            activation = "relu")

        self.flatten = Flatten()

        super().build(input_sape)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        return x
