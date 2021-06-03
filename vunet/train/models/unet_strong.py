import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam
from vunet.train.models.control_strong import FilmStrong
from vunet.train.models.unet import get_activation, u_net_deconv_block
from vunet.train.config import config


def u_net_conv_block(
    x,
    n_filters,
    initializer,
    input_conditions,
    inputs,
    activation,
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
    # n_filters == 512 means we are in the last layer
    if "latent" not in config.NAME or n_filters == 512:
        x = FilmStrong(
            type_gammas_betas=config.FILM_TYPE, in_freq=config.FREQ_ATTENTION
        )([x, input_conditions])
    x = get_activation(activation)(x)
    return x


def unet_strong():
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=config.INPUT_SHAPE)
    n_layers = config.N_LAYERS
    x = inputs
    input_conditions = Input(shape=[config.Z_DIM, config.N_FRAMES])
    conditions = input_conditions
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    # Encoder
    for ndx in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** ndx)
        x = u_net_conv_block(
            x,
            n_filters,
            initializer,
            conditions,
            inputs,
            activation=config.ACTIVATION_ENCODER,
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
