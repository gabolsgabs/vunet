from tensorflow.keras.layers import (
    Input, Conv1D, Dense, BatchNormalization, Dropout, Reshape
)
import tensorflow as tf
from vunet.train.config import config
from vunet.train.models.autopool import AutoPool1D


def get_input_conditions():
    if config.COND_INPUT == 'autopool':
        input_conditions = Input(shape=(config.Z_DIM, config.N_FRAMES))
        # axis = 2 because input_conditions -> [batch, cond, time]
        x = AutoPool1D(axis=2)(input_conditions)
    if config.CONTROL_TYPE == 'dense':
        if config.COND_INPUT == 'autopool':
            x = Reshape((1, config.Z_DIM))(x)
        else:
            input_conditions = Input(shape=(1, config.Z_DIM))
            x = input_conditions
    if config.CONTROL_TYPE == 'cnn':
        if config.COND_INPUT == 'autopool':
            x = Reshape((config.Z_DIM, 1))(x)
        else:
            input_conditions = Input(shape=(config.Z_DIM, 1))
            x = input_conditions
    return input_conditions, x


def dense_block(
    x, n_neurons, input_dim, initializer, activation='relu'
):
    for i, (n, d) in enumerate(zip(n_neurons, input_dim)):
        extra = i != 0
        x = Dense(n, input_dim=d, activation=activation,
                  kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def dense_control(n_conditions, n_neurons):
    """
    For simple dense control:
        - n_conditions = 6
        - n_neurons = phonemes [32, 64, 128],
                      type [16, 64, 128],
                      vocals [8, 32, 64]
    For complex dense control:
        - n_conditions = 1008
        - n_neurons = phonemes [64, 256, 1024],
                      type [16, 256, 1024],
                      vocals [8, 32, 512]
    """
    input_conditions, x = get_input_conditions()
    input_dim = [config.Z_DIM] + n_neurons[:-1]
    initializer = tf.random_normal_initializer(stddev=0.02)
    dense = dense_block(x, n_neurons, input_dim, initializer)
    gammas = Dense(
        n_conditions, input_dim=n_neurons[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(dense)
    betas = Dense(
        n_conditions, input_dim=n_neurons[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(dense)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas


def cnn_block(
    x, n_filters, kernel_size, padding, initializer, activation='relu'
):
    for i, (f, p) in enumerate(zip(n_filters, padding)):
        extra = i != 0
        x = Conv1D(f, kernel_size, padding=p, activation=activation,
                   kernel_initializer=initializer)(x)
        if extra:
            x = Dropout(0.5)(x)
            x = BatchNormalization(momentum=0.9, scale=True)(x)
    return x


def cnn_control(n_conditions, n_filters):
    """
    For simple dense control:
        - n_conditions = 6
        - n_filters = phonemes [32, 64, 128],
                      type [16, 64, 128],
                      vocals [8, 32, 64]
    For complex dense control:
        - n_conditions = 1008
        - n_filters = phonemes [64, 256, 1024],
                      type [16, 256, 1024],
                      vocals [8, 32, 512]
    """
    input_conditions, x = get_input_conditions()
    initializer = tf.random_normal_initializer(stddev=0.02)
    cnn = cnn_block(
        x, n_filters, config.Z_DIM, config.PADDING, initializer
    )
    gammas = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_G,
        kernel_initializer=initializer
    )(cnn)
    betas = Dense(
        n_conditions, input_dim=n_filters[-1], activation=config.ACT_B,
        kernel_initializer=initializer
    )(cnn)
    # both = Add()([gammas, betas])
    return input_conditions, gammas, betas