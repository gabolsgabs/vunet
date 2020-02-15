import numpy as np
from vunet.train.config import config
import logging
import gc
from pathlib import Path
from joblib import Parallel, delayed


logger = logging.getLogger('tensorflow')


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data):
    return np.divide((data - complex_min(data)),
                     (complex_max(data) - complex_min(data)))


def load_a_file(v, i, end):
    name = v.split('/')[-2]
    print('Loading the file %s %i out of %i' % (name, i, end))
    tmp = np.load(v)
    data = {}
    # data.setdefault(name, {})
    data['vocals'] = normlize_complex(tmp['vocals'])
    data['mix'] = normlize_complex(tmp['mixture'])
    data['acc'] = normlize_complex(tmp['accompaniment'])
    data['cond'] = tmp[config.CONDITION]
    data['ncc'] = tmp['ncc']
    return (name, data)


def load_data(files):
    """The data is loaded in memory just once for the generator to have direct
    access to it"""
    # for i, v in enumerate(files):
    #     data = load_a_file(v=v, i=i, end=len(files))
    data = {
        k: v for k, v in Parallel(n_jobs=16, verbose=5)(
                delayed(load_a_file)(v=v, i=i, end=len(files))
                for i, v in enumerate(files)
            )
    }
    _ = gc.collect()
    return data


def get_data():
    return load_data(
        [str(i) for i in Path(config.PATH_BASE).rglob('*features.npz')]
    )
