import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, multiply, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
# from vunet.train.models.FiLM_attention_conditions import (
#     FilmAttention, MultiHeadAttention
# )
from vunet.train.models.FiLM_attention_activation import FilmAttention
from vunet.train.models.unet_model import get_activation, u_net_deconv_block
from vunet.train.config import config


def u_net_conv_block(
    x, n_filters, initializer, input_conditions, inputs,
    activation, kernel_size=(5, 5), strides=(2, 2), padding='same'
):
    x = Conv2D(n_filters, kernel_size=kernel_size,  padding=padding,
               strides=strides, kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.9, scale=True)(x)
    # x = FilmAttention(
    #     type_gammas_betas=config.FILM_TYPE,
    #     do_time_attention=config.TIME_ATTENTION,
    #     do_freq_attention=config.FREQ_ATTENTION,
    # )([x, input_conditions])
    x = FilmAttention(
        type_gammas_betas=config.FILM_TYPE
    )([x, input_conditions])
    x = get_activation(activation)(x)
    return x


def vunet_attention_model():
    # axis should be fr, time -> right not it's time freqs
    inputs = Input(shape=config.INPUT_SHAPE)
    n_layers = config.N_LAYERS
    x = inputs
    input_conditions = Input(shape=[config.Z_DIM, config.N_FRAMES])
    conditions = input_conditions
    encoder_layers = []
    initializer = tf.random_normal_initializer(stddev=0.02)

    # # General attention
    # if config.TIME_ATTENTION:
    #     cond = tf.nn.softmax(conditions, axis=1)
    #     # [batch, time, features, channels] for attention
    #     cond = tf.transpose(input_conditions, perm=[0, 2, 1])
    #     x_att = tf.squeeze(tf.transpose(x, perm=[0, 2, 1, 3]), axis=-1)
    #     # the new conditions overwrite the original ones
    #     time_attention = MultiHeadAttention(
    #         units=config.Z_DIM, length=config.N_FRAMES, output_channels=1
    #     )
    #     conditions, _ = time_attention([cond, cond, x_att])
    #     conditions = tf.transpose(tf.squeeze(conditions, axis=-1), perm=[0, 2, 1])

    # Encoder
    for ndx in range(n_layers):
        n_filters = config.FILTERS_LAYER_1 * (2 ** ndx)
        x = u_net_conv_block(
            x, n_filters, initializer, conditions, inputs,
            activation=config.ACTIVATION_ENCODER
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
        skip = i > 0    # not skip in the first encoder block
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
    model.compile(
        optimizer=Adam(lr=config.LR, beta_1=0.5), loss=config.LOSS)
    return model
