import itertools
import copy
import manage_gpus as gpl
import numpy as np
import os
import sys


def get_epochs(batch_size, n_batch, size_training_data, r=3):
    # batch_size * n_batch * n_epoch = 3 * size_training_data
    return np.ceil((size_training_data*r)/(batch_size*n_batch)).astype(int)


def set_split(dali, b=0, n=50, errors=False):
    output = {}
    if not errors:
        # keep only the songs without errors
        for k, entry in dali.items():
            e = []
            for i in entry.info['errors'].values():
                e += i
            if len(e) == 0:
                output[k] = copy.deepcopy(entry)
    else:
        output = dali
    for i in list(output.keys())[b:b+n]:
        output[i].info['ground-truth'] = True
    return output


def chunks(l, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def get_indexes(files, N, step, chunk_size, **args):
    indexes = []
    np.random.shuffle(files)
    for n in files:
        tmp = np.load(os.path.join(n, 'mixture_spec_mag.npy'))
        s = []
        for j in np.arange(0, tmp.shape[0]-N, step):
            s.append([n, j])
        s = list(np.asarray(s, dtype=object)[np.random.permutation(len(s))])
        indexes += s
    indexes = list(chunks(indexes, chunk_size))
    # indexes = list(zip(*[iter(indexes)] * chunk_size)) # chunk a list
    np.random.shuffle(indexes)
    return list(itertools.chain.from_iterable(indexes))


def get_indexes_val(files, N):
    indexes = []
    for n in files:
        tmp = np.load(os.path.join(n, 'mixture_spec_mag.npy'))
        s = []
        for j in np.arange(0, tmp.shape[0]-N, N):
            s.append([n, j])
        indexes += s
    return indexes


def get_lock():
    gpu_id_locked = -1
    try:
        gpu_id_locked = gpl.get_gpu_lock(gpu_device_id=-1, soft=False)
    except gpl.NoGpuManager:
        print("no gpu manager available - will use all available GPUs",
              file=sys.stderr)
    except gpl.NoGpuAvailable:
        # there is no GPU available for locking, continue with CPU
        comp_device = "/cpu:0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return gpu_id_locked
