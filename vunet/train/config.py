# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import tensorflow as tf


class config(Config):
    # Add also notes?
    groups = [
        'standard',
        'sdp', 'sdt', 'sdn', 'sdc',        # simple dense phonemes/types/notes/chars
        'cdp', 'cdt', 'cdn', 'cdc',        # complex dense phonemes/types/notes/chars
        'scp', 'sct', 'scn', 'scc',        # simple cnn phonemes/types/notes/chars
        'ccp', 'cct', 'ccn', 'ccc'         # complex cnn phonemes/types/notes/chars
    ]
    # General
    MODE = ''   # conditioned, standard, attention
    NAME = ''
    # FOR OLD VERSION
    COND_MATRIX = 'overlap'    # sequential
    COND_INPUT = ''  # 'binary', 'mean_dur', 'mean_dur_norm', 'vocal_energy', 'autopool'

    # NEW FILM ATTENTION LAYER
    TIME_ATTENTION = None
    FREQ_ATTENTION = None
    WITH_SOFTMAX = False

    # GENERATOR
    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/'   # guzheng
    # PATH_BASE = '/data3/anasynth_nonbp/meseguerbrocal/multitracks/'   # gusli
    INDEXES_TRAIN = PATH_BASE + 'indexes/indexes_1_4.npz'
    INDEXES_VAL = PATH_BASE + 'indexes/indexes_128_4.npz'
    NUM_THREADS = tf.data.experimental.AUTOTUNE
    N_PREFETCH = tf.data.experimental.AUTOTUNE

    # training
    BATCH_SIZE = 128
    N_BATCH = 1024
    N_EPOCH = 1000
    AUG = True

    # Total = 515
    TEST = .89      # 101 tracks
    VAL = .863      # Val < x < Test -> 98
    TRAIN = .6      # Train < x Val -> 309

    # checkpoints
    EARLY_STOPPING_MIN_DELTA = 1e-5
    EARLY_STOPPING_PATIENCE = 20
    REDUCE_PLATEAU_PATIENCE = 10

    # conditions

    CONDITION = setting(
        'phonemes', standard='phonemes',
        sdp='phonemes', sdt='phoneme_types', sdn='notes', sdc='chars',
        cdp='phonemes', cdt='phoneme_types', cdn='notes', cdc='chars',
        scp='phonemes', sct='phoneme_types', scn='notes', scc='chars',
        ccp='phonemes', cct='phoneme_types', ccn='notes', ccc='chars'
    )
    Z_DIM = setting(
        40, standard=0,
        sdp=40, sdt=9, sdn=97, sdc=29,
        cdp=40, cdt=9, cdn=97, cdc=29,
        scp=40, sct=9, scn=97, scc=29,
        ccp=40, cct=9, ccn=97, ccc=29,
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
        sdp='dense', sdt='dense', sdn='dense', sdc='dense',
        cdp='dense', cdt='dense', cdn='dense', cdc='dense',
        scp='cnn', sct='cnn', scn='cnn', scc='cnn',
        ccp='cnn', cct='cnn', ccn='cnn',  ccc='cnn'
    )
    FILM_TYPE = setting(
        'simple', standard='',
        sdp='simple', sdt='simple', sdn='simple', sdc='simple',
        cdp='complex', cdt='complex', cdn='complex', cdc='complex',
        scp='simple', sct='simple', scn='simple', scc='simple',
        ccp='complex', cct='complex', ccn='complex', ccc='complex'
    )
    ACT_G = 'linear'
    ACT_B = 'linear'
    N_CONDITIONS = setting(
        6,
        sdp=6, sdt=6, sdn=6, sdc=6,
        cdp=1008, cdt=1008, cdn=1008, cdc=1008,
        scp=6, sct=6, scn=6, scc=6,
        ccp=1008, cct=1008, ccn=1008, ccc=1008
    )

    # cnn control
    N_FILTERS = setting(
        [],  standard=[],
        sdp=[], sdt=[], sdn=[], sdc=[],
        cdp=[], cdt=[], cdn=[], cdc=[],
        scp=[32, 64, 128], sct=[16, 64, 128], scn=[64, 128, 256], scc=[32, 64, 128],
        ccp=[64, 256, 1024], cct=[16, 256, 1024], ccn=[128, 256, 1024], ccc=[64, 256, 1024],
    )
    PADDING = ['same', 'same', 'valid']
    # Dense control
    N_NEURONS = setting(
        [32, 64, 128],  standard=[],
        sdp=[32, 64, 128], sdt=[16, 64, 128], sdn=[64, 128, 256], sdc=[32, 64, 128],
        cdp=[64, 256, 1024], cdt=[16, 256, 1024], cdn=[128, 256, 1024], cdc=[64, 256, 1024],
        scp=[], sct=[], scn=[], scc=[],
        ccp=[], cct=[], ccn=[], ccc=[],
    )
