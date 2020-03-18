# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):
    groups = ['standard', 'conditioned', 'attention']

    MODEL_NAME = ''
    RESULTS_NAME = 'results_dali.pkl'
    CONFIG = 'sdp'
    CONDITION = 'phonemes'
    PER_FEATURE = False
    COND_MATRIX = 'overlap'
    FILM_TYPE = 'simple'
    # original conditioned
    COND_INPUT = '' # 'binary', 'mean_dur', 'mean_dur_norm', 'vocal_energy', 'autopool'
    CONTROL_TYPE = 'dense'
    # attention
    WITH_SOFTMAX = False     # False or True
    TIME_ATTENTION = None
    FREQ_ATTENTION = None

    OVERLAP = 0
    MODE = setting('conditioned', standard='standard', attention='attention')

    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/models'

    PATH_MODEL = setting(
        os.path.join(PATH_BASE, 'conditioned'),
        standard=os.path.join(PATH_BASE, 'standard'),
        attention=os.path.join(PATH_BASE, 'attention')
    )
