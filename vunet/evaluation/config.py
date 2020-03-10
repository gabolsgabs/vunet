# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):
    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']

    RESULTS_NAME = 'results_dali.pkl'
    CONFIG = 'sdp'
    CONDITION = 'phonemes'
    COND_INPUT = 'binary'
    COND_MATRIX = 'overlap'
    MODEL_NAME = ''
    OVERLAP = 0
    MODE = setting('conditioned', standard='standard')

    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/models'

    PATH_MODEL = setting(
        os.path.join(PATH_BASE, 'conditioned'),
        standard=os.path.join(PATH_BASE, 'standard'),
    )

    CONTROL_TYPE = setting(
        default='dense', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
    FILM_TYPE = setting(
        default='simple', simple_dense='simple', complex_dense='complex',
        simple_cnn='simple', complex_cnn='complex'
    )
