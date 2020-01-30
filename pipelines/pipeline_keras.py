import keras
import numpy as np
import os
import gc

"""
tutorial at
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

Complex numbers and normalized magnitude log as entry

    mag_log = np.log10(np.abs(complex_number[:, 0:512])+1)
    mx_log = np.max(mag_log)
    mn_log = np.min(mag_log)
    norm_mag_log = (mag_log - mn_log)/(mx_log - mn_log)

For obtaining the original magnitude spectrum:
1- keep the max and mib log value (of the whole song):
    mag_log = norm_mag_log*(mx_log - mn_log) + mn_log
2- inverse log:
    mag = np.power(10, mag_log)-1
"""


def prepare_condition(cond, t):
    output = None
    if t == 'binary':
        output = np.max(cond, axis=0)
        output[np.where(output > 0)] = 1
    if t == 'energy':
        output = np.max(cond, axis=1)
        output[np.where(output > 0)] = 1
        output = np.mean(output)
    if t == 'ponderate':
        output = np.mean(cond, axis=0)
        tmp = np.max(output)
        if tmp > 1:
            output = output / tmp
    if t == 'autopool':
        output = cond
        output[np.where(output > 0)] = 1
    return output


class ValidationError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_index, config):
        'Initialization'
        # Train parameters
        for key in config:
            setattr(self, key, config[key])

        # self.files = files -> crear antes usando las id de DALI

        # Data info
        self.data_index = data_index
        self.targets = {}
        self.mixtures = {}
        self.conditions = {}
        # Control
        self.checked = 0
        self.size_epoch = self.batch_size*self.n_batch

        self.__check()
        self.__load_data()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.data_index) / self.batch_size))
        return self.n_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes)
        return X, Y

    def __load_data(self):
        for c, i in enumerate(self.files):
            print("Reading " + str(c) + " :" + i)
            self.mixtures[i] = np.load(os.path.join(i, 'mixture_spec_mag.npy'))
            self.targets[i] = np.load(os.path.join(i, 'vocals_spec_mag.npy'))
            if self.net_type == 'cond':
                self.conditions[i] = np.load(
                    os.path.join(i, 'phonemes_matrix.npy'))
        gc.collect()

    def __check(self):
        if len(self.data_index) < self.size_epoch:
            message = ("You are demanding more information in an epoch",
                       " than data points you have")
            raise ValidationError(message)
        return

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        le = self.batch_size*self.n_batch
        self.indexes = self.data_index[self.checked:self.checked+le]
        if le == len(self.indexes):
            self.checked += le
        else:
            # Last chunk of data that might be smalller than the epoch size
            d = le - len(self.indexes)
            self.indexes += self.data_index[0:d]
            self.checked = d
        _ = gc.collect()

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # (n_samples, *dim, n_channels)
        # Initialization
        output = None
        X = np.empty((self.batch_size, *self.dim))
        if self.net_type == 'cond':
            C = np.empty((self.batch_size, *self.dim_cond))
        Y = np.empty((self.batch_size, *self.dim))

        for i, (name, pos) in enumerate(indexes):
            X_tmp = self.mixtures[name][pos:pos+self.N]
            Y_tmp = self.targets[name][pos:pos+self.N]
            X[i, ] = X_tmp
            if self.net_type == 'cond':
                C_tmp = self.conditions[name][pos:pos+self.N]
                C[i, ] = prepare_condition(C_tmp, self.cond_input)
            # Store target
            Y[i, ] = Y_tmp

        X = np.expand_dims(np.transpose(X, (0, 2, 1)), axis=3)
        Y = np.expand_dims(np.transpose(Y, (0, 2, 1)), axis=3)
        r = np.random.permutation(X.shape[0])

        if self.net_type == 'cond':
            if self.cond_input in ['binary', 'ponderate', 'energy']:
                if self.emb_type == 'dense':
                    C = np.expand_dims(C, axis=1)
                if self.emb_type == 'cnn':
                    C = np.expand_dims(C, axis=2)
            else:
                C = np.transpose(C, (0, 2, 1))
            output = [X[r], C[r]], Y[r]
        else:
            output = X[r], Y[r]
        return output
