import keras
from keras.models import Model
from keras.layers import (Input, Dense, Conv1D, Conv2D, Layer, Reshape,
                          Conv2DTranspose, concatenate, multiply,
                          advanced_activations, Lambda)
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.backend import tf
from .autopool import AutoPool1D
# from keras.engine.topology import Layer

""" CONDITIONED AFTER THE BATCH NORM OF EACH ENCODING BLOCK"""


class FiLM(Layer):
    def __init__(self, **kwargs):
        super(FiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FiLM, self).build(input_shape)

    def call(self, args):
        assert isinstance(args, list)
        x, gamma, beta = args
        return tf.add(beta, tf.multiply(gamma, x))

    def compute_output_shape(self, input_shape):
        return input_shape


def FiLM_lambda(args):
    x, gamma, beta = args
    s = list(K.int_shape(x))
    s[0] = 1
    # avoiding tile with the num of batch -> it is the same for both tensors
    g = tf.tile(tf.expand_dims(tf.expand_dims(gamma, 2), 3), s)
    b = tf.tile(tf.expand_dims(tf.expand_dims(beta, 2), 3), s)
    return tf.add(b, tf.multiply(x, g))


def slice_lambda(position):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, position]
    return Lambda(func)


def slice_input_excerpt():
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, :-1, :]
    return Lambda(func)


def slice_input_conditions(n_conditions):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :n_conditions, -1, :]
    return Lambda(func)


def FiLM_generator_dense(input_shape, n_c_param, act_g='linear',
                         act_b='linear', cond_input='binary'):
    if cond_input in ['binary', 'ponderate', 'energy']:
        input_conditions = Input(shape=(1, input_shape[0]))
    else:
        input_conditions = Input(shape=input_shape)
    first_layer = input_conditions
    if cond_input == 'autopool':
        first_layer = AutoPool1D(axis=2)(first_layer)
        first_layer = Reshape((1, input_shape[0]))(first_layer)

    dense_first = 16
    i = 4
    while input_shape[0] > dense_first:
        i += 1
        dense_first = 2**i

    dense = Dense(dense_first, input_dim=input_shape[0], activation='relu',
                  kernel_initializer='he_normal')(first_layer)

    dense = Dense(128, input_dim=dense_first, activation='relu',
                  kernel_initializer='he_normal')(dense)
    dense = Dropout(0.5)(dense)
    dense = keras.layers.normalization.BatchNormalization()(dense)

    dense = Dense(256, input_dim=128, activation='relu',
                  kernel_initializer='he_normal')(dense)
    dense = Dropout(0.5)(dense)
    dense = keras.layers.normalization.BatchNormalization()(dense)

    gammas = Dense(n_c_param, input_dim=256, activation=act_g,
                   kernel_initializer='he_normal')(dense)   # 'ones'
    betas = Dense(n_c_param, input_dim=256, activation=act_b,
                  kernel_initializer='he_normal')(dense)   # 'zeros'
    # #both = Add()([gammas, betas])
    # return gammas, betas
    return input_conditions, gammas, betas


def FiLM_generator_CNN(input_shape, n_c_param, nb_filter,
                       act_g='linear', act_b='linear', cond_input='binary'):
    if cond_input in ['binary', 'ponderate', 'energy']:
        input_conditions = Input(shape=(input_shape[0], 1))
    else:
        input_conditions = Input(shape=input_shape)
    first_layer = input_conditions
    if cond_input == 'autopool':
        first_layer = AutoPool1D(axis=2)(first_layer)
        first_layer = Reshape((input_shape[0], 1))(first_layer)

    layerConv = Conv1D(nb_filter, input_shape[0], activation='relu',
                       padding='same')(first_layer)

    layerConv = Conv1D(2*nb_filter, input_shape[0], activation='relu',
                       padding='same')(layerConv)
    layerConv = Dropout(0.5)(layerConv)
    layerConv = keras.layers.normalization.BatchNormalization()(layerConv)

    layerConv = Conv1D(4*nb_filter, input_shape[0], activation='relu',
                       padding='valid')(layerConv)
    layerConv = Dropout(0.5)(layerConv)
    layerConv = keras.layers.normalization.BatchNormalization()(layerConv)

    gammas = Dense(n_c_param, activation=act_g,
                   kernel_initializer='he_normal')(layerConv)
    betas = Dense(n_c_param, activation=act_b,
                  kernel_initializer='he_normal')(layerConv)
    # #both = Add()([gammas, betas])
    # return gammas, betas
    return input_conditions, gammas, betas


def uNetConvBlock_filmed(layer, nb_filter, gamma, beta, cond='bn-film'):
    layerEncod = Conv2D(nb_filter, (5, 5),  padding='same',
                        strides=(2, 2))(layer)
    layerEncod = keras.layers.normalization.BatchNormalization()(layerEncod)

    if cond == 'bn-film':
        layerEncod = Lambda(FiLM_lambda)([layerEncod, gamma, beta])
        # layerEncod = FiLM()([layerEncod, gamma, beta])

    layerRelu = advanced_activations.LeakyReLU(alpha=0.2)(layerEncod)
    return layerRelu


def uNetConvBlock(layer, nb_filter):
    layerEncod = Conv2D(nb_filter, (5, 5),  padding='same',
                        strides=(2, 2))(layer)
    layerEncod = keras.layers.normalization.BatchNormalization()(layerEncod)
    layerRelu = advanced_activations.LeakyReLU(alpha=0.2)(layerEncod)
    return layerRelu


def uNetDeconvBlock(layerPrev, layerConcat, nb_filter, shapeOut1, shapeOut2):
    layerMerge = concatenate([layerPrev, layerConcat], axis=3)
    layerDecod = Conv2DTranspose(
        nb_filter, (5, 5),
        padding='same', strides=(2, 2), activation='relu')(layerMerge)
    return layerDecod


def uNetDeconvBlockLast(layerPrev, layerConcat, nb_filter, shapeOut1,
                        shapeOut2, activation='sigmoid'):
    layerMerge = concatenate([layerPrev, layerConcat], axis=3)

    if activation == 'lrelu':
        layerDecod = Conv2DTranspose(
            nb_filter, (5, 5),
            padding='same', strides=(2, 2))(layerMerge)
        layerDecod = advanced_activations.LeakyReLU(alpha=0.2)(layerDecod)
    else:
        layerDecod = Conv2DTranspose(
            nb_filter, (5, 5),
            padding='same', strides=(2, 2), activation=activation)(layerMerge)
    return layerDecod


def ponderateMeanAbsoluteLoss(N, y_true, y_pred):
    import scipy
    diff = K.abs(y_pred - y_true)
    sumDiff = K.sum(diff, axis=2)
    ponderateWindow = K.variable(scipy.signal.gaussian(N, N//2))
    return K.mean(ponderateWindow * sumDiff)


def get_unet_filmed(
    input_shape=(512, 128, 1), input_shape_cond=(40,), n_c_param=6, lr=0.001,
    act_g='linear', act_b='linear', act_last='sigmoid', emb_type='dense',
    nb_filter=16, cond_input='binary', **kargs
):
    nbFreq, nbFrame, _ = input_shape
    input_excerpt = Input(shape=input_shape)

    if emb_type == 'dense':
        input_conditions, gammas, betas = FiLM_generator_dense(
            input_shape_cond, n_c_param, act_g, act_b, cond_input)
    if emb_type == 'cnn':
        input_conditions, gammas, betas = FiLM_generator_CNN(
            input_shape_cond, n_c_param, nb_filter, act_g, act_b, cond_input)

    layer1Relu = uNetConvBlock_filmed(
        input_excerpt, 16, slice_lambda(0)(gammas), slice_lambda(0)(betas))
    layer2Relu = uNetConvBlock_filmed(
        layer1Relu, 32, slice_lambda(1)(gammas), slice_lambda(1)(betas))
    layer3Relu = uNetConvBlock_filmed(
        layer2Relu, 64, slice_lambda(2)(gammas), slice_lambda(2)(betas))
    layer4Relu = uNetConvBlock_filmed(
        layer3Relu, 128, slice_lambda(3)(gammas), slice_lambda(3)(betas))
    layer5Relu = uNetConvBlock_filmed(
        layer4Relu, 256, slice_lambda(4)(gammas), slice_lambda(4)(betas))
    layer6Relu = uNetConvBlock_filmed(
        layer5Relu, 512, slice_lambda(5)(gammas), slice_lambda(5)(betas))

    layer1Decod = Conv2DTranspose(
        256, (5, 5),
        padding='same', activation='relu', strides=(2, 2))(layer6Relu)
    layer1Decod = Dropout(0.5)(layer1Decod)

    layer2Decod = uNetDeconvBlock(
        layer5Relu, layer1Decod, 128, nbFreq//16, nbFrame//16)
    layer2Decod = Dropout(0.5)(layer2Decod)

    layer3Decod = uNetDeconvBlock(
        layer4Relu, layer2Decod, 64, nbFreq//8, nbFrame//8)
    layer3Decod = Dropout(0.5)(layer3Decod)

    layer4Decod = uNetDeconvBlock(
        layer3Relu, layer3Decod, 32, nbFreq//4, nbFrame//4)

    layer5Decod = uNetDeconvBlock(
        layer2Relu, layer4Decod, 16, nbFreq//2, nbFrame//2)

    layer6Decod = uNetDeconvBlockLast(
        layer1Relu, layer5Decod, 1, nbFreq, nbFrame, activation=act_last)
    outputLayer = multiply([input_excerpt, layer6Decod])

    model = Model(inputs=[input_excerpt, input_conditions],
                  outputs=outputLayer)
    # model = Model(inputs=input_net, outputs=outputLayerList)

    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error')
    return model
