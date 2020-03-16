import tensorflow as tf
from tensorflow.keras.layers import Lambda


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
