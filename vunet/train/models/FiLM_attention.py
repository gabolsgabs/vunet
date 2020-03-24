import tensorflow as tf
from vunet.train.config import config


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
    activations -> [batch, time, freq, channels]
    units -> [batch, 1, time, channels]
    expected matmul -> [freq, 1] x [1, time] = [freq, time]
    """
    # tile time in the 'freq' axis -> [batch, time, time, channels]
    u = tf.tile(units, [1, list(activations.shape)[1], 1, 1])
    # the dimension to broadcast has to be first [batch, channels, freq, time]
    a = tf.transpose(activations, perm=[0, 3, 1, 2])
    u = tf.transpose(u, perm=[0, 3, 1, 2])
    # output tf.matmul -> [batch, channels, time, freq]
    output = tf.matmul(u, a)
    # back to [batch, freq, time, channels], original feature map input
    return tf.transpose(output, perm=[0, 3, 2, 1])


def dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.Model):
    def __init__(
        self, units, length, num_heads=config.NUM_HEADS, **kwargs
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
        # values
        self.wv = tf.keras.layers.Dense(
            units*num_heads, trainable=True, kernel_initializer='random_normal'
        )
        # merge the different heads
        self.dense = tf.keras.layers.Dense(
            units, trainable=True, kernel_initializer='random_normal'
        )

    def split_heads(self, x):
        """Split the last dimension into (num_heads, units).
        Transpose the result such that the shape is
        (batch_size, num_heads, time, units)
        """
        # [batch, time, num_heads, units]
        x = tf.reshape(
            x, (config.BATCH_SIZE, self.length, self.num_heads, self.units)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        # get queries and keys
        q = self.wq(q)  # [batch_size, len_q, units*num_heads]
        k = self.wk(k)  # [batch_size, len_k, units*num_heads]
        v = self.wv(v)  # [batch_size, len_k, units*num_heads]
        # [batch_size, num_heads, len_q, units]
        q = self.split_heads(q)
        # [batch_size, num_heads, len_k, units]
        k = self.split_heads(k)
        # [batch_size, num_heads, len_k, units]
        v = self.split_heads(v)
        # scaled -> [batch_size, num_heads, seq_len_q, units]
        # attention -> [batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = dot_product_attention(q, k, v)
        # [batch_size, seq_len_q, units, channels, num_heads]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (config.BATCH_SIZE, -1, self.units*self.num_heads)
        )
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class FilmAttention(tf.keras.Model):
    def __init__(
        self,
        type_gammas_betas='simple',
        do_time_attention=None,     # simple, complex
        do_freq_attention=None,     # simple, complex
        **kwargs
    ):
        super(FilmAttention, self).__init__(**kwargs)
        self.type_gammas_betas = type_gammas_betas  # simple or complex
        self.do_time_attention = do_time_attention  # simple or complex
        self.do_freq_attention = do_freq_attention  # simple or complex

    def build(self, input_shape):
        """ data_shape -> [batch, freq, time, channles]
            cond_shape -> [batch, z_dim, time]
        """
        data_shape, cond_shape = input_shape
        self.units = cond_shape[1]  # z_dim
        self.time_frames = data_shape[2]
        if self.type_gammas_betas == 'simple':
            shape_gammas_betas = [1, self.units, 1]
        if self.type_gammas_betas == 'complex':
            shape_gammas_betas = [data_shape[-1], self.units, 1]  # channels

        # FiLM parameters
        self.gammas = self.add_weight(
            shape=shape_gammas_betas, trainable=True,
            name='gammas', initializer='random_normal'
        )
        self.betas = self.add_weight(
            shape=shape_gammas_betas, trainable=True,
            name='betas', initializer='random_normal'
        )

        # Attention in time
        if self.do_time_attention:
            self.time_attention = MultiHeadAttention(
                units=self.units, length=self.time_frames
            )
        # Attention in freq
        if self.do_freq_attention:
            # units = n_freqs
            self.freq_attention = MultiHeadAttention(
                units=self.time_frames, length=data_shape[1]
            )

    def title_shape(self, shape):
        # TILE
        shape[0] = 1        # not tile in the batch dimension
        shape[2] = 1        # not tile in the time dimension
        if (self.type_gammas_betas == 'complex' or
           self.do_time_attention or self.do_freq_attention):
            shape[-1] = 1    # not tile in the channels dimension
        if self.do_freq_attention:
            shape[1] = 1    # not tile in the freq dimension
        return shape

    def prepare_cond(self, conditions):
        conditions = tf.expand_dims(conditions, -1)
        # reshape for deeper layers
        conditions = tf.image.resize(
            conditions, (self.units, self.time_frames), method='nearest'
        )
        return tf.nn.softmax(conditions, axis=1)

    def merge_two_last(self, data):
        # merge channels and features
        shape = list(data.shape)
        return tf.reshape(
            data, (config.BATCH_SIZE, shape[1], shape[-1]*shape[-2])
        )

    def call(self, inputs):
        x, conditions = inputs
        # we use the original ones
        conditions = self.prepare_cond(conditions)
        # to have the right dimension [batch, time, cond, channels] for direct mult
        conditions = tf.transpose(conditions, perm=[0, 2, 1, 3])
        # OPERATIONS
        if self.do_time_attention:  # unless do attention in time
            cond_att = self.merge_two_last(conditions)
            # [batch, time, features, channels] for attention
            x_att = self.merge_two_last(tf.transpose(x, perm=[0, 2, 1, 3]))
            # the new conditions overwrite the original ones
            conditions, _ = self.time_attention(cond_att, cond_att, x_att)
            conditions = tf.expand_dims(conditions, -1)

        gammas = matmul_time(conditions, self.gammas)
        betas = matmul_time(conditions, self.betas)

        if self.do_freq_attention:
            # [batch, features, time, channels] for attention
            x_att = self.merge_two_last(x)
            # to use the note info??
            freqs, _ = self.freq_attention(x_att, x_att, x_att)
            gammas = matmul_freq(freqs, gammas)
            betas = matmul_freq(freqs, betas)

        # FiLM
        shape_tile = self.title_shape(list(x.shape))
        gammas = tf.tile(gammas, shape_tile)
        betas = tf.tile(betas, shape_tile)

        return tf.add(betas, tf.multiply(x, gammas))
