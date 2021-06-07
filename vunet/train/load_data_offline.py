import numpy as np
from vunet.train.config import config
import logging
import gc
from pathlib import Path


logger = logging.getLogger("tensorflow")


def get_max_complex(data, keys=["vocals", "mixture", "accompaniment"]):
    # sometimes the max is not the mixture
    pos = np.argmax([np.abs(complex_max(data[i])) for i in keys])
    return np.array([complex_max(data[i]) for i in keys])[pos]


def visualization(spec, features, mask=False):
    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(features, aspect="auto", origin="lower")
    if mask:
        features = np.ma.masked_where(features == 0, features)
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.imshow(spec, aspect="auto", origin="lower")
    if mask and np.max(features.data) > 0:
        ax2.imshow(
            features,
            origin="lower",
            interpolation="none",
            alpha=0.9,
            cmap="Blues",
            aspect="auto",
        )
    plt.show()
    return


def complex_max(d):
    return d[np.unravel_index(np.argmax(np.abs(d), axis=None), d.shape)]


def complex_min(d):
    return d[np.unravel_index(np.argmin(np.abs(d), axis=None), d.shape)]


def normlize_complex(data, c_max=1):
    if c_max != 1:
        factor = np.divide(complex_max(data), c_max)
    else:
        factor = 1
    # normalize between 0-1
    output = np.divide(
        (data - complex_min(data)), (complex_max(data) - complex_min(data))
    )
    return np.multiply(output, factor)  # scale to the original range


def split_overlapped(data):
    output = np.zeros(data.shape)
    ndx = [i for i in np.argsort(data[:, 0])[::-1] if data[:, 0][i] > 0][::-1]
    if len(ndx) > 1:
        s = np.round(data.shape[1] / len(ndx)).astype(np.int)
        for i, v in enumerate(ndx):
            output[v, i * s : i * s + s] = 1
        output[v, i * s :] = 1
    else:
        output = data
    return output


def as_categorical(data):
    data_binary = data.__copy__()
    data_binary[data > 0] = 1
    init = np.sum(np.diff(data_binary, axis=1), axis=0)
    init = np.hstack((1, init))
    ndx = np.where(init != 0)[0]
    for i in range(len(ndx)):
        if i + 1 < len(ndx):
            e = ndx[i + 1]
        else:
            e = len(init)
        data_binary[:, ndx[i] : e] = split_overlapped(data[:, ndx[i] : e])
    data_binary[data_binary > 0] = 1
    return np.expand_dims(data_binary, axis=-1).astype(np.float32)


def binarize(data):
    data[data > 0] = 1
    # data[-1, :] = 0  # blank to 0
    return np.expand_dims(data, axis=-1).astype(np.float32)


def load_a_file(v, i, end, condition):
    name = v.split("/")[-2]
    print("Loading the file %s %i out of %i" % (name, i, end))
    tmp = np.load(v)
    data = {}
    # data.setdefault(name, {})
    data["normalization"] = {
        "vocals": [complex_max(tmp["vocals"]), complex_min(tmp["vocals"])],
        "accompaniment": [
            complex_max(tmp["accompaniment"]),
            complex_min(tmp["accompaniment"]),
        ],
        "mixture": [complex_max(tmp["mixture"]), complex_min(tmp["mixture"])],
    }
    c_max = get_max_complex(tmp)
    data["mix"] = normlize_complex(tmp["mixture"], c_max)
    data["vocals"] = normlize_complex(tmp["vocals"], c_max)
    data["acc"] = normlize_complex(tmp["accompaniment"], c_max)
    data["cond"] = np.concatenate(
        [binarize(tmp[condition]), as_categorical(tmp[condition])], axis=-1
    )
    data["ncc"] = tmp["ncc"]
    return (name, data)


def load_data(files):
    from joblib import Parallel, delayed

    """The data is loaded in memory just once for the generator to have direct
    access to it"""
    # for i, v in enumerate(files):
    #     data = load_a_file(v=v, i=i, end=len(files))
    data = {
        k: v
        for k, v in Parallel(n_jobs=16, verbose=5)(
            delayed(load_a_file)(v=v, i=i, end=len(files), condition=config.CONDITION)
            for i, v in enumerate(files)
        )
    }
    _ = gc.collect()
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)
    return data


def get_data(ids=None):
    files = [str(i) for i in Path(config.PATH_BASE).rglob("*features.npz")]
    if ids:
        files = [i for i in files if i.split("/")[-2] in ids]
    return load_data(files)
