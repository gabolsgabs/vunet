# How to use this package

Tools for training a new model.

###### 1. Precompute the data.

See the preprocess [folder](https://github.com/gabolsgabs/cunet/tree/master/cunet/preprocess)

###### 2. Train a new model.

> python -m cunet.train.main        # check cunet/config.py for the configuration


PHONEMES_PAIRS = {
    'AA': 'vowel', 'AE': 'vowel', 'AH': 'vowel', 'AO': 'vowel', 'AW': 'vowel',
    'AY': 'vowel', 'B': 'stop', 'CH': 'affricate', 'D': 'stop',
    'DH': 'fricative', 'EH': 'vowel', 'ER': 'vowel', 'EY': 'vowel',
    'F': 'fricative', 'G': 'stop', 'HH': 'aspirate', 'IH': 'vowel',
    'IY': 'vowel', 'JH': 'affricate', 'K': 'stop', 'L': 'liquid', 'M': 'nasal',
    'N': 'nasal', 'NG': 'nasal', 'OW': 'vowel', 'OY': 'vowel', 'P': 'stop',
    'R': 'liquid', 'S': 'fricative', 'SH': 'fricative', 'T': 'stop',
    'TH': 'fricative', 'UH': 'vowel', 'UW': 'vowel', 'V': 'fricative',
    'W': 'semivowel', 'Y': 'semivowel', 'Z': 'fricative', 'ZH': 'fricative'
}

PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']


PHONEMES_TYPE = ['vowel', 'stop', 'affricate', 'fricative', 'aspirate',
                 'liquid', 'nasal', 'semivowel']
