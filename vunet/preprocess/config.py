# -*- coding: utf-8 -*-
from effortless_config import Config


class config(Config):
    PATH_BASE = '/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks/'   # /net/guzheng/
    PATH_DALI = '/u/anasynth/meseguerbrocal/dataset/multitracks/v0/dali_ready/'
    PATH_INDEXES = PATH_BASE + 'indexes/'
    FR = 8192
    FFT_SIZE = 1024
    HOP = 256
    TARGETS = ['phonemes', 'phoneme_types', 'chars', 'notes']
    STEP = 1                # step in frames for creating the input data
    CHUNK_SIZE = 4          # chunking the indexes before mixing -> define the number of data points of the same track
    TIME_R = HOP/FR
    N_FRAMES = 128
