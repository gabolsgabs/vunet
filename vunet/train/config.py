# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import tensorflow as tf


class config(Config):
    # Add also notes?
    groups = [
        'standard',
        'sdp', 'sdt', 'sdv',        # simple dense phonemes/types/vocals
        'cdp', 'cdt', 'cdv',        # complex dense phonemes/types/vocals
        'scp', 'sct', 'scv',        # simple cnn phonemes/types/vocals
        'ccp', 'cct', 'ccv'         # complex cnn phonemes/types/vocals
    ]
    # General
    MODE = setting(
        'conditioned', standard='standard',
        sdp='conditioned', sdt='conditioned', sdv='conditioned',
        cdp='conditioned', cdt='conditioned', cdv='conditioned',
        scp='conditioned', sct='conditioned', scv='conditioned',
        ccp='conditioned', cct='conditioned', ccv='conditioned'
    )
    NAME = 'before_checking_if_the_input_is_correct'

    # GENERATOR
    # PATH_BASE = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/'   # guzheng
    PATH_BASE = '/data3/anasynth_nonbp/meseguerbrocal/multitracks/'   # gusli
    INDEXES_TRAIN = PATH_BASE + 'indexes/indexes_1_4.npz'
    INDEXES_VAL = PATH_BASE + 'indexes/indexes_128_4.npz'
    NUM_THREADS = tf.data.experimental.AUTOTUNE
    N_PREFETCH = tf.data.experimental.AUTOTUNE

    # training
    BATCH_SIZE = 64
    N_BATCH = 2048
    N_EPOCH = 1000
    AUG = True

    VAL = .863  #
    TEST = .9   # 73 tracks

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 1e-6
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_PLATEAU_PATIENCE = 5

    # conditions
    CONDITION = setting(
        'phonemes', standard='phonemes',
        sdp='phonemes', sdt='phoneme_types', sdv='chars',
        cdp='phonemes', cdt='phoneme_types', cdv='chars',
        scp='phonemes', sct='phoneme_types', scv='chars',
        ccp='phonemes', cct='phoneme_types', ccv='chars'
    )
    COND_INPUT = 'binary'      # 'binary', 'mean_dur', 'mean_dur_norm', 'vocal_energy', 'autopool'
    COND_MATRIX = 'overlap'    # sequential
    Z_DIM = setting(
        39, standard=0,
        sdp=39, sdt=8, sdv=1,
        cdp=39, cdt=8, cdv=1,
        scp=39, sct=8, scv=1,
        ccp=39, cct=8, ccv=1
    )

    # unet paramters
    N_FRAMES = 128
    INPUT_SHAPE = [512, N_FRAMES, 1]
    FILTERS_LAYER_1 = 16
    N_LAYERS = 6
    LR = 1e-3
    ACTIVATION_ENCODER = 'leaky_relu'
    ACTIVATION_DECODER = 'relu'
    ACT_LAST = 'sigmoid'
    LOSS = 'mean_absolute_error'

    # -------------------------------

    # control parameters
    CONTROL_TYPE = setting(
        'dense', standard='',
        sdp='dense', sdt='dense', sdv='dense',
        cdp='dense', cdt='dense', cdv='dense',
        scp='cnn', sct='cnn', scv='cnn',
        ccp='cnn', cct='cnn', ccv='cnn'
    )
    FILM_TYPE = setting(
        'simple', standard='',
        sdp='simple', sdt='simple', sdv='simple',
        cdp='complex', cdt='complex', cdv='complex',
        scp='simple', sct='simple', scv='simple',
        ccp='complex', cct='complex', ccv='complex'
    )
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        6,
        sdp=6, sdt=6, sdv=6,
        cdp=1008, cdt=1008, cdv=1008,
        scp=6, sct=6, scv=6,
        ccp=1008, cct=1008, ccv=1008
    )

    # cnn control
    N_FILTERS = setting(
        [],  standard=[],
        sdp=[], sdt=[], sdv=[],
        cdp=[], cdt=[], cdv=[],
        scp=[32, 64, 128], sct=[16, 64, 128], scv=[8, 32, 64],
        ccp=[64, 256, 1024], cct=[16, 256, 1024], ccv=[8, 32, 512]
    )
    PADDING = ['same', 'same', 'valid']
    # Dense control
    N_NEURONS = setting(
        [32, 64, 128],  standard=[],
        sdp=[32, 64, 128], sdt=[16, 64, 128], sdv=[8, 32, 64],
        cdp=[64, 256, 1024], cdt=[16, 256, 1024], cdv=[8, 32, 512],
        scp=[], sct=[], scv=[],
        ccp=[], cct=[], ccv=[],
    )
