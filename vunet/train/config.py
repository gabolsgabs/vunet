# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import tensorflow as tf


class config(Config):
    # Add also notes?
    groups = [
        'standard',
        'sdp', 'sdt', 'sdv', 'sdn', 'sdc',        # simple dense phonemes/types/vocals/notes/chars
        'cdp', 'cdt', 'cdv', 'cdn', 'cdc',        # complex dense phonemes/types/vocals/notes/chars
        'scp', 'sct', 'scv', 'scn', 'scc',        # simple cnn phonemes/types/vocals/notes/chars
        'ccp', 'cct', 'ccv', 'ccn', 'ccc'         # complex cnn phonemes/types/vocals/notes/chars
    ]
    # General
    MODE = setting(
        default='conditioned', standard='standard',
    )
    NAME = 'with_aug'

    # GENERATOR
    # PATH_BASE = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/'   # guzheng
    PATH_BASE = '/data3/anasynth_nonbp/meseguerbrocal/multitracks/'   # gusli
    INDEXES_TRAIN = PATH_BASE + 'indexes/indexes_1_4.npz'
    INDEXES_VAL = PATH_BASE + 'indexes/indexes_128_4.npz'
    NUM_THREADS = tf.data.experimental.AUTOTUNE
    N_PREFETCH = tf.data.experimental.AUTOTUNE

    # training
    BATCH_SIZE = 128
    N_BATCH = 1024
    N_EPOCH = 1000
    AUG = False

    # Total = 515
    TEST = .89      # 101 tracks
    VAL = .863      # Val < x < Test -> 98
    TRAIN = .6      # Train < x Val -> 309

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 1e-5
    EARLY_STOPPING_PATIENCE = 30
    REDUCE_PLATEAU_PATIENCE = 15

    # conditions
    CONDITION = setting(
        'phonemes', standard='phonemes',
        sdp='phonemes', sdt='phoneme_types', sdv='vocals', sdn='notes', sdc='chars',
        cdp='phonemes', cdt='phoneme_types', cdv='vocals', cdn='notes', cdc='chars',
        scp='phonemes', sct='phoneme_types', scv='vocals', scn='notes', scc='chars',
        ccp='phonemes', cct='phoneme_types', ccv='vocals', ccn='notes', ccc='chars'
    )
    COND_INPUT = 'binary'      # 'binary', 'mean_dur', 'mean_dur_norm', 'vocal_energy', 'autopool'
    COND_MATRIX = 'overlap'    # sequential
    Z_DIM = setting(
        40, standard=0,
        sdp=40, sdt=9, sdv=1, sdn=97, sdc=29,
        cdp=40, cdt=9, cdv=1, cdn=97, cdc=29,
        scp=40, sct=9, scv=1, scn=97, scc=29,
        ccp=40, cct=9, ccv=1, ccn=97, ccc=29,
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
        sdp='dense', sdt='dense', sdv='dense', sdn='dense', sdc='dense',
        cdp='dense', cdt='dense', cdv='dense', cdn='dense', cdc='dense',
        scp='cnn', sct='cnn', scv='cnn', scn='cnn', scc='cnn',
        ccp='cnn', cct='cnn', ccv='cnn',  ccn='cnn',  ccc='cnn'
    )
    FILM_TYPE = setting(
        'simple', standard='',
        sdp='simple', sdt='simple', sdv='simple', sdn='simple', sdc='simple',
        cdp='complex', cdt='complex', cdv='complex', cdn='complex', cdc='complex',
        scp='simple', sct='simple', scv='simple', scn='simple', scc='simple',
        ccp='complex', cct='complex', ccv='complex', ccn='complex', ccc='complex'
    )
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        6,
        sdp=6, sdt=6, sdv=6, sdn=6, sdc=6,
        cdp=1008, cdt=1008, cdv=1008, cdn=1008, cdc=1008,
        scp=6, sct=6, scv=6, scn=6, scc=6,
        ccp=1008, cct=1008, ccv=1008, ccn=1008, ccc=1008
    )

    # cnn control
    N_FILTERS = setting(
        [],  standard=[],
        sdp=[], sdt=[], sdv=[], sdn=[], sdc=[],
        cdp=[], cdt=[], cdv=[], cdn=[], cdc=[],
        scp=[32, 64, 128], sct=[16, 64, 128], scv=[8, 32, 64], scn=[64, 128, 256], scc=[32, 64, 128],
        ccp=[64, 256, 1024], cct=[16, 256, 1024], ccv=[8, 32, 512], ccn=[128, 256, 1024], ccc=[64, 256, 1024],
    )
    PADDING = ['same', 'same', 'valid']
    # Dense control
    N_NEURONS = setting(
        [32, 64, 128],  standard=[],
        sdp=[32, 64, 128], sdt=[16, 64, 128], sdv=[8, 32, 64], sdn=[64, 128, 256], sdc=[32, 64, 128],
        cdp=[64, 256, 1024], cdt=[16, 256, 1024], cdv=[8, 32, 512], cdn=[128, 256, 1024], cdc=[64, 256, 1024],
        scp=[], sct=[], scv=[], scn=[], scc=[],
        ccp=[], cct=[], ccv=[], ccn=[], ccc=[],
    )
