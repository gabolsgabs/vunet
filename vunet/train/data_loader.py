import numpy as np
import tensorflow as tf
from vunet.train.config import config
from itertools import groupby
import random
from vunet.train.load_data_offline import get_data

DATA = get_data()


def check_shape(data):
    n = data.shape[0]
    if n % 2 != 0:
        n = data.shape[0] - 1
    return np.expand_dims(data[:n, :], axis=2)


def apply_to_keys(keys, func, *args, **kwargs):
    def wrapped(d):
        return {
            k: func(v, *args, **kwargs)
            if k in keys else v for k, v in d.items()
        }
    return wrapped


def split_overlapped(data):
    output = np.zeros(data.shape)
    ndx = [i for i in np.argsort(data[:, 0])[::-1] if data[:, 0][i] > 0][::-1]
    if len(ndx) > 1:
        s = np.round(data.shape[1] / len(ndx)).astype(np.int)
        for i, v in enumerate(ndx):
            output[v, i*s:i*s+s] = 1
        output[v, i*s:] = 1
    else:
        output = data
    return output


@tf.function(autograph=False)
def as_categorical(data):
    def py_target_as_categorical(data):
        data = data.numpy()
        data_binary = data.__copy__()
        data_binary[data > 0] = 1
        init = np.sum(np.diff(data_binary, axis=1), axis=0)
        init = np.hstack((1, init))
        ndx = np.where(init != 0)[0]
        for i in range(len(ndx)):
            if i+1 < len(ndx):
                e = ndx[i+1]
            else:
                e = len(init)
            data_binary[:, ndx[i]:e] = split_overlapped(data[:, ndx[i]:e])
        data_binary[data_binary > 0] = 1
        return np.expand_dims(data_binary, axis=-1).astype(np.float32)
    return tf.py_function(py_target_as_categorical, [data], (tf.float32))


@tf.function(autograph=False)
def binarize(data):
    def py_binarize(data):
        data = data.numpy()
        data[data > 0] = 1
        # data[-1, :] = 0  # blank to 0
        return np.expand_dims(data, axis=-1).astype(np.float32)
    return tf.py_function(py_binarize, [data], (tf.float32))


@tf.function(autograph=False)
def get_sequence(data):
    def py_get_sequence(data):
        # the length is in the last element
        output = np.zeros(config.INPUT_SHAPE[1]+1)
        # silence as element in the sequence
        seq = np.where(data >= 1)
        if len(seq[1]) > 0:
            if config.DUPLICATE:
                # seq = np.array(seq[1])
                tmp = []
                for i in groupby(seq[1]):
                    if i[0] != 39:
                        tmp += [j for j in i[1]]
                    else:
                        tmp.append(i[0])
                seq = np.array(tmp)
            else:
                seq = np.array([i[0] for i in groupby(seq[1])])
            output[:len(seq)] = seq
            output[-1] = len(seq)
        return output.astype(np.float32)
    return tf.py_function(py_get_sequence, [data], (tf.float32))


def process_target(data):
    b = binarize(data)
    c = as_categorical(data)
    output = tf.concat([b, c], axis=-1)
    return output


def get_frame(data, frame):
    return data[:, frame:frame+config.INPUT_SHAPE[1]]


def get_input_frame(target, data, frame, val_set):
    output = get_frame(data['mix'], frame)
    # 25% of doing it
    if not val_set and random.sample(range(0, 4), 1)[0] == 0 and config.AUG:
        # just pick another point of the same track
        if random.sample(range(0, 2), 1)[0] == 0:
            # or pick a random point of another track
            uid = random.choice([i for i in DATA.keys()])
            data = DATA[uid]
        frame = random.choice(
            [i for i in range(len(data['acc'])-config.INPUT_SHAPE[1])]
        )
        output = np.sum([target, get_frame(data['acc'], frame)], axis=0)
    return check_shape(np.abs(output))


def yield_data(indexes, files, val_set):
    for i in indexes:
        if i[0] in files:
            target = get_frame(DATA[i[0]]['vocals'], i[1])
            yield {
                'target': check_shape(np.abs(target)),
                # abs and check_shape not before for doing the sum next
                'input': get_input_frame(target, DATA[i[0]], i[1], val_set),
                'conditions': get_frame(DATA[i[0]]['cond'], i[1])
            }


def load_indexes_file(val_set=False):
    if not val_set:
        indexes = np.load(config.INDEXES_TRAIN, allow_pickle=True)['indexes']
        r = list(range(len(indexes)))
        random.shuffle(r)
        indexes = indexes[r]
        files = [k for k, v in DATA.items() if v['ncc'] < config.TRAIN]
    else:
        indexes = np.load(config.INDEXES_VAL, allow_pickle=True)['indexes']
        files = [
            k for k, v in DATA.items()
            if v['ncc'] > config.VAL and v['ncc'] < config.TEST
        ]
    return yield_data(indexes, files, val_set)


@tf.function(autograph=False)
def prepare_condition(data):
    def py_prepare_condition(data):
        reshape = False
        c_shape = (config.Z_DIM, config.N_FRAMES)
        if config.COND_MATRIX == 'overlap':
            cond = data[:, :, 0].numpy()
        if config.COND_MATRIX == 'sequential':
            cond = data[:, :, 1].numpy()
        output = cond
        if config.COND_INPUT == 'binary':
            output = np.max(cond[:-1, :], axis=1)   # -1 remove silence
            reshape = True
        if config.COND_INPUT == 'mean_dur':
            output = np.mean(cond[:-1, :], axis=1)
            reshape = True
        if config.COND_INPUT == 'mean_dur_norm':
            output = np.mean(cond[:-1, :], axis=1)
            output = output / np.max(output)
            reshape = True
        if config.COND_INPUT == 'vocal_energy':
            # just an scalar
            output = np.mean(np.max(cond[:-1, :], axis=1))
            reshape = True
        if config.CONTROL_TYPE == 'dense' and reshape:
            c_shape = (1, config.Z_DIM)
        if config.CONTROL_TYPE == 'cnn' and reshape:
            c_shape = (config.Z_DIM, 1)
        output = tf.ensure_shape(tf.reshape(output, c_shape), c_shape)
        return output
    return tf.py_function(py_prepare_condition, [data], (tf.float32))


def convert_to_estimator_input(d):
    inputs = tf.ensure_shape(d['input'], config.INPUT_SHAPE)
    outputs = tf.ensure_shape(d["target"], config.INPUT_SHAPE)
    cond = prepare_condition(d['conditions'])
    inputs = (inputs, cond)
    return (inputs, outputs)


def dataset_generator(val_set=False):
    ds = tf.data.Dataset.from_generator(
        load_indexes_file,
        {'target': tf.float32, 'input': tf.float32, 'conditions': tf.int32},
        args=[val_set]
    ).map(
        apply_to_keys(["conditions"], process_target),
        num_parallel_calls=config.NUM_THREADS
    ).map(
        convert_to_estimator_input, num_parallel_calls=config.NUM_THREADS
    ).batch(
        config.BATCH_SIZE, drop_remainder=True
    ).prefetch(
        buffer_size=config.N_PREFETCH
    )
    if not val_set:
        ds = ds.repeat()
    return ds
