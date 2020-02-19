# -*- coding: utf-8 -*-
from effortless_config import Config, setting
import os


class config(Config):
    groups = ['standard', 'simple_dense', 'complex_dense', 'simple_cnn',
              'complex_cnn']

    GROUP = setting(
        default='simple_dense', simple_dense='simple_dense',
        complex_dense='complex_dense',
        simple_cnn='simple_cnn', complex_cnn='complex_cnn',
    )
    PATH_BASE = '/net/gusli/data3/anasynth_nonbp/meseguerbrocal/multitracks/models/standard/'
    NAME = 'before_checking_if_the_input_is_correct'

    PATH_MODEL = setting(
        os.path.join(PATH_BASE, 'conditioned/simple_dense'),
        standard=os.path.join(PATH_BASE, 'standard'),
        simple_dense=os.path.join(PATH_BASE, 'conditioned/simple_dense'),
        complex_dense=os.path.join(PATH_BASE, 'conditioned/complex_dense'),
        simple_cnn=os.path.join(PATH_BASE, 'conditioned/simple_cnn'),
        complex_cnn=os.path.join(PATH_BASE, 'conditioned/complex_cnn')
    )
    PATH_AUDIO = '/net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/musdb18/test/complex'
    OVERLAP = 0
    MODE = setting(default='conditioned', standard='standard')
    EMB_TYPE = setting(
        default='dense', simple_dense='dense', complex_dense='dense',
        simple_cnn='cnn', complex_cnn='cnn'
    )
