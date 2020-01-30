import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, multiply
from keras.layers import advanced_activations
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras import backend as K


def uNetConvBlock(layer, nb_filter):
    layerEncod = Conv2D(nb_filter, (5, 5),  padding='same',
                        strides=(2, 2))(layer)
    layerEncod = keras.layers.normalization.BatchNormalization()(layerEncod)
    layerRelu = advanced_activations.LeakyReLU(alpha=0.2)(layerEncod)
    return layerRelu


def uNetDeconvBlock(layerPrev, layerConcat, nb_filter, shapeOut1, shapeOut2):
    layerMerge = concatenate([layerPrev, layerConcat], axis=3)
    layerDecod = Conv2DTranspose(
        nb_filter, (5, 5), padding='same', strides=(2, 2),
        activation='relu')(layerMerge)
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


def get_unet(input_shape=(512, 128, 1), lr=0.001, act_last='sigmoid', **kargs):
    nbFreq, nbFrame, _ = input_shape

    input_excerpt = Input(shape=input_shape)
    layer1Relu = uNetConvBlock(input_excerpt, 16)
    layer2Relu = uNetConvBlock(layer1Relu, 32)
    layer3Relu = uNetConvBlock(layer2Relu, 64)
    layer4Relu = uNetConvBlock(layer3Relu, 128)
    layer5Relu = uNetConvBlock(layer4Relu, 256)
    layer6Relu = uNetConvBlock(layer5Relu, 512)

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
        layer1Relu, layer5Decod, 1, nbFreq, nbFrame,  activation=act_last)
    outputLayer = multiply([input_excerpt, layer6Decod])

    model = Model(input_excerpt, outputLayer)
    model.compile(optimizer=Adam(lr=lr), loss='mean_absolute_error')

    return model
