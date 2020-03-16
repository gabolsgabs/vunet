import tensorflow as tf


def matmul_time(activations, units):
    """
    activations -> [batch, time, cond, channels]
    units -> [channels, cond, 1]
    expected matmul -> [time, cond] x [cond, 1] = [time, 1]
    """
    # the dimension to broadcast has to be first [batch, channels, time, cond]
    a = tf.transpose(activations, perm=[0, 3, 1, 2])
    # output tf.matmul -> [batch, channels, time, 1]
    output = tf.matmul(a, units)
    # back to [batch, 1, time, channels], original feature map input
    return tf.transpose(output, perm=[0, 3, 2, 1])


def matmul_freq(activations, units):
    """
    activations -> [batch, freq, 1, channels]
    units -> [batch, 1, time, channels]
    expected matmul -> [freq, 1] x [1, time] = [freq, time]
    """
    # the dimension to broadcast has to be first [batch, channels, freq, time]
    a = tf.transpose(activations, perm=[0, 3, 1, 2])
    u = tf.transpose(units, perm=[0, 3, 1, 2])
    # output tf.matmul -> [batch, channels, freq, time]
    output = tf.matmul(a, u)
    # back to [batch, freq, time, channels], original feature map input
    return tf.transpose(output, perm=[0, 2, 3, 1])


class Controller(tf.keras.Model):
    def __init__(
        self, config, shape, multi_function, **kwargs
    ):
        super(Controller, self).__init__(**kwargs)
        self.shape = shape
        self.multi_function = multi_function

    def build(self, input_shape):
        self.activations = self.add_weight(
            shape=self.shape, trainable=True, name='controller',
            initializer='random_normal',
        )

    def call(self, gamma_beta):
        # x = place to infer x
        return self.multi_function(self.activations, gamma_beta)


class FilmAttention(tf.keras.Model):
    def __init__(
        self,
        type_gammas_betas='simple',
        type_time_activations=None,
        type_freq_activations=None,
        sofmax=False,
        **kwargs
    ):
        super(FilmAttention, self).__init__(**kwargs)
        self.type_gammas_betas = type_gammas_betas  # simple or complex
        self.type_time_activations = type_time_activations  # simple or complex
        self.type_freq_activations = type_freq_activations  # simple or complex
        self.sofmax = sofmax
        self.channels = 1

    def build(self, input_shape):
        """ input_shape -> [batch, freq, time, channles]"""
        data_shape, cond_shape = input_shape
        self.units = cond_shape[1]  # z_dim
        self.channels = data_shape[-1]
        self.time_frames = data_shape[2]
        if self.type_gammas_betas == 'simple':
            shape_gammas_betas = [1, self.units, 1]
        if self.type_gammas_betas == 'complex':
            shape_gammas_betas = [self.channels, self.units, 1]

        self.gammas = self.add_weight(
            shape=shape_gammas_betas, trainable=True,
            name='gammas', initializer='random_normal'
        )
        self.betas = self.add_weight(
            shape=shape_gammas_betas, trainable=True,
            name='betas', initializer='random_normal'
        )
        if self.type_time_activations is not None:
            # internal shape for matmul [batch, time, cond, channels]
            if self.type_time_activations == 'simple':
                shape_time = [1, self.time_frames, self.units, 1]
            if self.type_time_activations == 'complex':
                shape_time = [1, self.time_frames, self.units, self.channels]
            self.time_activations = Controller(
                config='simple', shape=shape_time,
                multi_function=matmul_time,
            )
        if self.type_freq_activations is not None:
            # internal shape for matmul [batch, freq, 1, channels]
            if self.type_freq_activations == 'simple':
                shape_freq = [1, data_shape[1], 1, 1]
            if self.type_freq_activations == 'complex':
                shape_freq = [1, data_shape[1], 1, self.channels]
            self.freq_activations = Controller(
                config='simple', shape=shape_freq,
                multi_function=matmul_freq,
            )

    def call(self, inputs):
        x, conditions = inputs
        shape_tile = list(x.shape)
        shape_tile[0] = 1        # not tile in the batch dimension
        shape_tile[2] = 1        # not tile in the time dimension
        if self.sofmax:
            # to keep the barycentre small
            self.gammas = tf.nn.softmax(self.gammas, axis=0)
            self.betas = tf.nn.softmax(self.betas, axis=0)
        if 'complex' in [
            self.type_gammas_betas, self.type_freq_activations,
            self.type_time_activations
        ]:
            shape_tile[-1] = 1    # not tile in the channels dimension
        if self.type_freq_activations is not None:
            shape_tile[1] = 1    # not tile in the freq dimension
        if self.type_time_activations is None:   # we use the original cond
            conditions = tf.expand_dims(conditions, -1)
            # reshape for deeper layers
            conditions = tf.image.resize(
                conditions, (self.units, self.time_frames))
            # to have the right dimension [batch, time, cond, channels]
            conditions = tf.transpose(conditions, perm=[0, 2, 1, 3])
            g = matmul_time(conditions, self.gammas)
            b = matmul_time(conditions, self.betas)
        if self.type_time_activations is not None:
            g = self.time_activations(self.gammas)
            b = self.time_activations(self.betas)
        if self.type_freq_activations is not None:
            g = self.freq_activations(g)
            b = self.freq_activations(b)
        g = tf.tile(g, shape_tile)
        b = tf.tile(b, shape_tile)
        return tf.add(b, tf.multiply(x, g))
