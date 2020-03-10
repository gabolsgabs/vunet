import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import initializers, constraints, regularizers


def shift_bit_length(x):
    return 1 << (x-1).bit_length()


def FiLM_simple_layer():
    """multiply scalar to a tensor"""
    def func(args):
        x, gamma, beta = args
        shape = list(x.shape)
        shape[0] = 1
        # avoid tile with the num of batch -> it is the same for both tensors
        g = tf.tile(tf.expand_dims(tf.expand_dims(gamma, 2), 3), shape)
        b = tf.tile(tf.expand_dims(tf.expand_dims(beta, 2), 3), shape)
        return tf.add(b, tf.multiply(x, g))
    return Lambda(func)


def FiLM_complex_layer():
    """multiply tensor to tensor"""
    def func(args):
        x, gamma, beta = args
        shape = list(x.shape)
        # avoid tile with the num of batch -> same for both tensors
        shape[0] = 1
        # avoid tile with the num of channels -> same for both tensors
        shape[-1] = 1
        g = tf.tile(tf.expand_dims(gamma, 1), shape)
        b = tf.tile(tf.expand_dims(beta, 1), shape)
        return tf.add(b, tf.multiply(x, g))
    return Lambda(func)


# SEE https://www.tensorflow.org/api_docs/python/tf/gather
def slice_tensor(position):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, position]
    return Lambda(func)


def slice_tensor_range(init, end):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, init:end]
    return Lambda(func)


def do_matmul_time(activations, units):
    """
    activations -> [batch, time, cond, channels]
    units -> [channels, cond, 1]
    expected matmul -> [time, cond] x [cond, 1] = [time, 1]
    """
    # the dimension to broadcast has to be first [batch, channels, time, cond]
    a = tf.transpose(activations, perm=[0, 3, 1, 2])
    u = tf.nn.softmax(units, axis=0)    # to keep the barycentre small
    # output tf.matmul -> [batch, channels, time, 1]
    output = tf.matmul(a, u)
    # back to [batch, 1, time, channels], original feature map input
    return tf.transpose(output, perm=[0, 3, 2, 1])


def do_matmul_freq(activations, units):
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


class FiLM_attention(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        config='simple',
        learn_time_attention=False,
        learn_freq_attention=False,
        init_time_act_with_cond=False,
        weight_initializer='random_normal',
        weight_constraint=None,
        weight_regularizer=None,
        **kwargs
    ):
        super(FiLM_attention, self).__init__(**kwargs)
        self.units = units      # num of conditions z_dim
        self.config = config    # simple or complex
        self.learn_time_attention = learn_time_attention
        self.learn_freq_attention = learn_freq_attention
        self.init_time_act_with_cond = init_time_act_with_cond  # init
        self.channels = 1
        self.weight_initializer = initializers.get(weight_initializer)
        self.weight_constraint = constraints.get(weight_constraint)
        self.weight_regularizer = regularizers.get(weight_regularizer)

    def build(self, input_shape):
        """ input_shape -> [batch, freq, time, channles]"""
        if self.config == 'complex':
            self.channels = input_shape[-1]
        self.time_frames = input_shape[2]

        self.gammas = self.add_weight(
            shape=(self.channels, self.units, 1), trainable=True, name='gammas',
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            constraint=self.weight_constraint
        )
        self.betas = self.add_weight(
            shape=(self.channels, self.units, 1), trainable=True, name='betas',
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            constraint=self.weight_constraint
        )
        # TO DO!!!tiene q ser aprendido desede el cond!
        if self.learn_time_attention:
            # internal shape for matmul [batch, time, cond, channels]
            shape_time = [1, self.time_frames, self.units, self.channels]
            self.time_activations = self.add_weight(
                shape=shape_time, trainable=True, name='time_activations',
                initializer=self.weight_initializer,
                regularizer=self.weight_regularizer,
                constraint=self.weight_constraint
            )
        # TO DO!!!tiene q ser aprendido desede del spec!
        if self.learn_freq_attention:
            # internal shape for matmul [batch, freq, 1, channels]
            shape_freq = [1, input_shape[1], 1, self.channels]
            self.freq_activations = self.add_weight(
                shape=shape_freq, trainable=True, name='freq_activations',
                initializer=self.weight_initializer,
                regularizer=self.weight_regularizer,
                constraint=self.weight_constraint
            )

    def call(self, x, conditions):
        # to have the right dimension [batch, time, cond, channels]
        conditions = tf.expand_dims(
            tf.transpose(conditions, perm=[0, 2, 1]), -1)
        conditions = tf.image.resize(conditions, (self.units, self.time_frames))
        shape = list(x.shape)
        shape[0] = 1        # not tile in the batch dimension
        shape[2] = 1        # not tile in the time dimension
        if self.config == 'complex':
            shape[-1] = 1    # not tile in the channels dimension
        if self.learn_freq_attention:
            shape[1] = 1    # not tile in the freq dimension
        if not self.learn_time_attention:
            g = do_matmul_time(conditions, self.gammas)
            b = do_matmul_time(conditions, self.betas)
        if self.learn_time_attention:
            # init of the activations with the original annotations
            if self.init_time_act_with_cond:
                self.time_activations.assign(
                    tf.tile(conditions, [1, 1, 1, self.channels])
                )
            g = do_matmul_time(self.time_activations, self.gammas)
            b = do_matmul_time(self.time_activations, self.betas)
        if self.learn_freq_attention:
            g = do_matmul_freq(self.freq_activations, g)
            b = do_matmul_freq(self.freq_activations, b)
        g = tf.tile(g, shape)
        b = tf.tile(b, shape)
        return tf.add(b, tf.multiply(x, g))
