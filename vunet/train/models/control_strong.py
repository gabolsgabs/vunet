import tensorflow as tf


def my_matmul(activations, units):
    """
    activations -> [batch, time, cond, channels]
    units -> [channels, cond, freqs]
    expected matmul -> [time, cond] x [cond, freqs] = [time, freqs]
    """
    # the dimension to broadcast has to be first [batch, channels, time, cond]
    a = tf.transpose(activations, perm=[0, 3, 1, 2])
    # output tf.matmul -> [batch, channels, time, freqs]
    output = tf.matmul(a, units)
    # back to [batch, freqs, time, channels], original feature map input
    return tf.transpose(output, perm=[0, 3, 2, 1])


class FilmStrong(tf.keras.Model):
    def __init__(self, type_gammas_betas="simple", in_freq=False, **kwargs):
        super(FilmStrong, self).__init__(**kwargs)
        self.type_gammas_betas = type_gammas_betas  # simple or complex
        self.in_freq = in_freq

    def build(self, input_shape):
        """data_shape -> [batch, freq, time, channles]
        cond_shape -> [batch, z_dim, time]
        """
        data_shape, cond_shape = input_shape
        self.z_dim = cond_shape[1]  # z_dim
        self.channels = data_shape[-1]
        self.time_frames = data_shape[2]
        if self.type_gammas_betas == "simple":
            shape_gammas_betas = [1, self.z_dim, 1]
        if self.type_gammas_betas == "complex":
            shape_gammas_betas = [self.channels, self.z_dim, 1]
        if self.in_freq:
            shape_gammas_betas[-1] = data_shape[1]

        self.gammas = self.add_weight(
            shape=shape_gammas_betas,
            trainable=True,
            name="gammas",
            initializer="random_normal",
        )
        self.betas = self.add_weight(
            shape=shape_gammas_betas,
            trainable=True,
            name="betas",
            initializer="random_normal",
        )

    def prepare_cond(self, conditions):
        conditions = tf.expand_dims(conditions, -1)
        # reshape the phoneme matrix for deeper layers
        conditions = tf.image.resize(
            conditions, (self.z_dim, self.time_frames), method="nearest"
        )
        conditions = tf.nn.softmax(conditions, axis=1)
        # to have the right dimension [batch, time, cond, channels]
        conditions = tf.transpose(conditions, perm=[0, 2, 1, 3])
        return conditions

    def call(self, inputs):
        x, conditions = inputs
        gammas = self.gammas
        betas = self.betas
        conditions = self.prepare_cond(conditions)
        gammas = my_matmul(conditions, gammas)
        betas = my_matmul(conditions, betas)
        return tf.add(betas, tf.multiply(x, gammas))
