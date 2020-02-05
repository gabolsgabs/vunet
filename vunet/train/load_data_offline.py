import numpy as np
from vunet.train.config import config
import logging
from pathlib import Path


logger = logging.getLogger('tensorflow')


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data):
    return np.divide((data - complex_min(data)),
                     (complex_max(data) - complex_min(data)))


def load_data(files):
    """The data is loaded in memory just once for the generator to have direct
    access to it"""
    data = {}
    for i, v in enumerate(files):
        print(v)
        name = v.split('/')[-2]
        logger.info('Loading the file %s %i out of %i' % (name, i, len(files)))
        tmp = np.load(v)
        data.setdefault(name, {})
        data[name]['vocals'] = normlize_complex(tmp['vocals'])
        data[name]['mix'] = normlize_complex(tmp['mixture'])
        data[name]['acc'] = normlize_complex(tmp['accompaniment'])
        data[name]['cond'] = tmp[config.CONDITION]
        data[name]['ncc'] = tmp['ncc']
    return data


def get_data():
    return load_data(
        [str(i) for i in Path(config.PATH_BASE).rglob('*features.npz')]
    )
