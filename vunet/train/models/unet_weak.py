import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam
from vunet.train.models.control_weak import (
    dense_control,
    cnn_control,
    FiLM_simple_layer,
    FiLM_complex_layer,
    slice_tensor,
    slice_tensor_range,
)
from vunet.train.models.unet import get_activation, u_net_deconv_block
from vunet.train.config import config


def u_net_conv_block(
    x,
    n_filters,
    initializer,
    gamma,
    beta,
    activation,
    film_type,
    kernel_size=(5, 5),
    strides=(2, 2),
    padding="same",
):
    x = Conv2D(
        n_filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        kernel_initializer=initializer,
    )(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    if film_type == "simple":
        x = FiLM_simple_layer()([x, gamma, beta])
    if film_type == "complex":
        x = FiLM_complex_layer()([x, gamma, beta])
    x = get_activation(activation)(x)
    return x


def get_control_model():
    n_conditions = config.N_CONDITIONS
    n_neurons = config.N_NEURONS
    n_filters = config.N_FILTERS
    if config.CONTROL_TYPE == "dense":
        input_conditions, gammas, betas = dense_control(
            n_conditions=n_conditions, n_neurons=n_neurons
        )
    if config.CONTROL_TYPE == "cnn":
        input_conditions, gammas, betas = cnn_control(
            n_conditions=n_conditions, n_filters=n_filters
        )
    return input_conditions, gammas, betas


def get_gammas_betas_for_block(gammas, betas, ndx, ndx_range, n_filters):
    # Original architecture - conditions as dict
    if config.FILM_TYPE == "simple":
        gamma, beta = slice_tensor(ndx)(gammas), slice_tensor(ndx)(betas)
    if config.FILM_TYPE == "complex":
        init, end = ndx_range, ndx_range + n_filters
        gamma = slice_tensor_range(init, end)(gammas)
        beta = slice_tensor_range(init, end)(betas)
        ndx_range += n_filters
    return gamma, beta, ndx_range


def unet_weak():
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=config.INPUT_SHAPE)
    n_layers = config.N_LAYERS
    x = inputs
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)
    input_conditions, gammas, betas = get_control_model()

    # Encoder
    ndx_range = 0
    for ndx in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** ndx)
        gamma, beta, ndx_range = get_gammas_betas_for_block(
            gammas, betas, ndx, ndx_range, n_filters
        )
        x = u_net_conv_block(
            x,
            n_filters,
            initializer,
            gamma,
            beta,
            activation=config.ACTIVATION_ENCODER,
            film_type=config.FILM_TYPE,
        )
        encoder_layers.append(x)
    # Decoder
    for i in range(n_layers):
        # parameters each decoder layer
        is_final_block = i == n_layers - 1  # the las layer is different
        # not dropout in the first block and the last two encoder blocks
        dropout = not (i == 0 or i == n_layers - 1 or i == n_layers - 2)
        # for getting the number of filters
        encoder_layer = encoder_layers[n_layers - i - 1]
        skip = i > 0  # not skip in the first encoder block
        if is_final_block:
            n_filters = 1
            activation = config.ACT_LAST
        else:
            n_filters = encoder_layer.get_shape().as_list()[-1] // 2
            activation = config.ACTIVATION_DECODER
        x = u_net_deconv_block(
            x, encoder_layer, n_filters, initializer, activation, dropout, skip
        )
    outputs = multiply([inputs, x])
    model = Model(inputs=[inputs, input_conditions], outputs=outputs)
    model.compile(optimizer=Adam(lr=config.LR, beta_1=0.5), loss=config.LOSS)
    return model
