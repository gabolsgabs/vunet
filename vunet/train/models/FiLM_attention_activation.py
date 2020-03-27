import tensorflow as tf
from vunet.train.config import config


def dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.Model):
    def __init__(
        self, length, units=config.ATTENTION_UNITS,
        num_heads=config.NUM_HEADS, **kwargs
    ):
        """ Adaptation from
        https://www.tensorflow.org/tutorials/text/transformer
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        # attention
        self.num_heads = num_heads
        self.units = units
        self.length = length
        # queries
        self.wq = tf.keras.layers.Dense(
            units*num_heads, trainable=True, kernel_initializer='random_normal'
        )
        # keys
        self.wk = tf.keras.layers.Dense(
            units*num_heads, trainable=True, kernel_initializer='random_normal'
        )
        # For merging the different heads
        self.dense = tf.keras.layers.Dense(
            1, trainable=True, kernel_initializer='random_normal'
        )

    def merge_last_two(self, data):
        # merge channels and features
        shape = list(data.shape)
        batch_size = tf.shape(data)[0]
        return tf.reshape(data, (batch_size, shape[1], shape[-1]*shape[-2]))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, units).
        Transpose the result such that the shape is
        (batch_size, num_heads, time, units)
        """
        # [batch, time, num_heads, units]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.units))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        v, k, q = inputs
        batch_size = tf.shape(q)[0]
        output_shape = list(v.shape)
        # get queries and keys
        # [batch_size, len_q, units*num_heads]
        q = self.wq(self.merge_last_two(tf.transpose(q, perm=[0, 2, 1, 3])))
        # [batch_size, len_k, units*num_heads]
        k = self.wk(self.merge_last_two(tf.transpose(k, perm=[0, 2, 1, 3])))
        # we do not want to compute new values only select the right ones
        v = self.merge_last_two(tf.transpose(v, perm=[0, 2, 1, 3]))
        v = tf.expand_dims(v, axis=1)

        # [batch_size, num_heads, len_q, units]
        q = self.split_heads(q, batch_size)
        # [batch_size, num_heads, len_k, units]
        k = self.split_heads(k, batch_size)

        # scaled -> [batch_size, num_heads, len_q, units]
        # attention -> [batch_size, num_heads, len_q, len_k]
        scaled_attention, attention_weights = dot_product_attention(q, k, v)
        # [batch_size, len_q, units, channels, num_heads]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 3, 1])
        # [batch_size, len_q, features*output_channels]
        output = tf.squeeze(self.dense(scaled_attention), axis=-1)
        # [batch_size, len_q, features, channels]
        output = tf.reshape(
            output, (batch_size, self.length, output_shape[1], output_shape[-1])
        )
        # [batch_size, features, len_q, channels]
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        return output, attention_weights


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


class FilmAttention(tf.keras.Model):
    def __init__(
        self,
        type_gammas_betas='simple',
        mode='time',     # time or sequence
        **kwargs
    ):
        super(FilmAttention, self).__init__(**kwargs)
        self.type_gammas_betas = type_gammas_betas  # simple or complex
        self.mode = mode

    def build(self, input_shape):
        """ data_shape -> [batch, freq, time, channles]
            cond_shape -> [batch, z_dim, time]
        """
        data_shape, cond_shape = input_shape
        self.z_dim = cond_shape[1]  # z_dim
        self.channels = data_shape[-1]
        self.time_frames = data_shape[2]
        if self.type_gammas_betas == 'simple':
            shape_gammas_betas = [1, self.z_dim, 1]
        if self.type_gammas_betas == 'complex':
            shape_gammas_betas = [self.channels, self.z_dim, 1]

        self.gammas = self.add_weight(
            shape=shape_gammas_betas, trainable=True,
            name='gammas', initializer='random_normal'
        )
        self.betas = self.add_weight(
            shape=shape_gammas_betas, trainable=True,
            name='betas', initializer='random_normal'
        )
        # una para gammas otra para betas?
        self.attention = MultiHeadAttention(length=self.time_frames)

    def prepare_cond(self, conditions):
        conditions = tf.expand_dims(conditions, -1)
        if self.mode == 'time':
            # reshape for deeper layers
            conditions = tf.image.resize(
                conditions, (self.z_dim, self.time_frames), method='nearest'
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
        gammas = matmul_time(conditions, gammas)
        betas = matmul_time(conditions, betas)
        keys = tf.concat([gammas, betas], axis=-1)
        gammas, _ = self.attention([gammas, keys, x])
        betas, _ = self.attention([betas, keys, x])
        return tf.add(betas, tf.multiply(x, gammas))
