import tensorflow as tf


class DCNV2(tf.keras.Model):
    def __init__(self, feature_num_vocabs, feat_dim, out_dim, num_cross, num_linear):
        super().__init__()

        self.num_cross = num_cross
        self.num_linear = num_linear

        self.num_features = len(feature_num_vocabs)

        input_dim = feat_dim * self.num_features

        self.embedding_layers = [
            tf.kera.layers.Embedding(feature_num_vocabs[i], feat_dim)
            for i in range(self.num_features)
        ]
        self.cross_in_layers = [
            tf.keras.layers.Dense(input_dim) for _ in range(self.num_cross)
        ]
        self.cross_out_layers = [
            tf.keras.layers.Dense(self.input_dim, activation="gelu")
            for _ in range(self.num_linear)
        ]
        self.out_layer = tf.keras.layers.Dense(self.out_dim)

    def call(self, inputs):
        X = []
        for i in range(self.num_features):
            X.append(self.embedding_layers[i](tf.gather(inputs, i, axis=1)))
        X = tf.concat(X, axis=1)
        X0 = tf.identity(X)

        for i in range(self.num_cross):
            X = X0 * self.cross_out_layers[i](self.cross_in_layers[i](X)) + X

        for i in range(self.num_linear):
            X = self.linear_layers[i](X)

        X = self.out_layer(X)

        return X
