import tensorflow as tf


class Predictor(tf.keras.Model):
    """出力を任意の次元にする"""

    def __init__(self, out_dim):
        super().__init__()

        self.out_layer = tf.keras.layers.Dense(out_dim)

    def call(self, inputs):
        return self.out_layer(inputs)
