import gc
import numpy as np
import os
from pipelines.pipeline_keras import prepare_condition


def get_val_data(val_data_files, data_index, batch_size, dim, emb_type,
                 net_type, num_cond, N, cond_input, dim_cond):
    mixtures = {}
    targets = {}
    conditions = {}
    for c, i in enumerate(val_data_files):
        print("Reading "+str(c)+" out of "+str(len(val_data_files))+": "+i)
        mixtures[i] = np.load(os.path.join(i, 'mixture_spec_mag.npy'))
        targets[i] = np.load(os.path.join(i, 'vocals_spec_mag.npy'))
        if net_type == 'cond':
            conditions[i] = np.load(os.path.join(i, 'phonemes_matrix.npy'))

    X = np.empty((len(data_index), *dim))
    Y = np.empty((len(data_index), *dim))
    if net_type == 'cond':
        C = np.empty((len(data_index), *dim_cond))
    # Generate data
    for i, (name, pos) in enumerate(data_index):
        # print(name, pos, cond)
        X_tmp = mixtures[name][pos:pos+N]
        Y_tmp = targets[name][pos:pos+N]
        X[i, ] = X_tmp
        if net_type == 'cond':
            C_tmp = conditions[name][pos:pos+N]
            C[i, ] = prepare_condition(C_tmp, cond_input)
        # Store target
        Y[i, ] = Y_tmp

    X = np.expand_dims(np.transpose(X, (0, 2, 1)), axis=3)
    Y = np.expand_dims(np.transpose(Y, (0, 2, 1)), axis=3)
    if net_type == 'cond':
        if cond_input in ['binary', 'ponderate', 'energy']:
            if emb_type == 'dense':
                C = np.expand_dims(C, axis=1)
            if emb_type == 'cnn':
                C = np.expand_dims(C, axis=2)
        else:
            C = np.transpose(C, (0, 2, 1))
        output = [X, C], Y
    else:
        output = X, Y
    return output
