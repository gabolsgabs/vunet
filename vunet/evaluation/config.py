# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):
    # groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
    #           'complex_cnn']

    RESULTS_NAME = 'results_dali.pkl'
    MODEL_NAME = 'with_aug'
    OVERLAP = 0
    MODE = 'standard'

    # GROUP = setting(
    #     default='simple_dense', simple_dense='simple_dense',
    #     complex_dense='complex_dense',
    #     simple_cnn='simple_cnn', complex_cnn='complex_cnn',
    #     standar='standard'
    # )

    PATH_BASE = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/models'
    PATH_MODEL = os.path.join(PATH_BASE, 'standard')

    # PATH_MODEL = setting(
    #     # os.path.join(PATH_BASE, 'conditioned/simple_dense'),
    #     standard=os.path.join(PATH_BASE, 'standard'),
    #     # simple_dense=os.path.join(PATH_BASE, 'conditioned/simple_dense'),
    #     # complex_dense=os.path.join(PATH_BASE, 'conditioned/complex_dense'),
    #     # simple_cnn=os.path.join(PATH_BASE, 'conditioned/simple_cnn'),
    #     # complex_cnn=os.path.join(PATH_BASE, 'conditioned/complex_cnn')
    # )

    # Order path is created config.CONDITION, config.FILM_TYPE, config.CONTROL_TYPE

    # MODE = setting(default='conditioned', standard='standard')
    # EMB_TYPE = setting(
    #     default='dense', simple_dense='dense', complex_dense='dense',
    #     simple_cnn='cnn', complex_cnn='cnn'
    # )
