from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import numpy as np


BATCH_SIZE = 512
DATA_SHAPE = [40, 128]


@tf.function
def my_operation0(data, my_units):
    def py_my_operation(d, u):
        u = u.numpy()
        d = d.numpy()
        for ndx, value in enumerate(d):
            factor = tf.tile(
                tf.expand_dims(tf.expand_dims(u[np.random.randint(40)], 0), 1),
                DATA_SHAPE
            )
            d[ndx, :, :] = tf.multiply(d[ndx, :, :], factor)
        return d
    return tf.py_function(py_my_operation, [data, my_units], (tf.float32))


@tf.function
def my_operation1(data, my_units):
    def py_my_operation(d, u):
        return tf.multiply(
            d, tf.tile(tf.expand_dims(my_units, 1), [1, DATA_SHAPE[1]])
        )
    return tf.py_function(py_my_operation, [data, my_units], (tf.float32))


def my_operation2(data, my_units):
    return tf.multiply(
        data, tf.tile(tf.expand_dims(my_units, 1), [1, DATA_SHAPE[1]])
    )


@tf.function
def my_operation3(data, my_units):
    for ndx, value in enumerate(data):
        tmp = np.random.randint(40)
        test = tf.tile(
            tf.expand_dims(tf.expand_dims(my_units[tmp], 0), 1), DATA_SHAPE
        )
        data[ndx, :, :] = tf.multiply(data[ndx, :, :], test)
    return data


class Toy_layer(tf.keras.layers.Layer):
    def __init__(self, units, config='one'):
        super(Toy_layer, self).__init__()
        self.units = units
        self.config = config

    def build(self, input_shape):
        self.my_units = self.add_weight(
            shape=(self.units,), initializer='random_normal', trainable=True
        )

    def call(self, data):
        if self.config == 'zero':
            output = my_operation0(data, self.my_units)
            output = tf.ensure_shape(output, (BATCH_SIZE, *DATA_SHAPE))
        if self.config == 'one':
            output = my_operation1(data, self.my_units)
            output = tf.ensure_shape(output, (BATCH_SIZE, *DATA_SHAPE))
        if self.config == 'two':
            output = my_operation2(data, self.my_units)
        if self.config == 'three':
            output = my_operation3(data, self.my_units)
        return tf.reduce_sum(output, axis=[1, 2])


def train_model(name):
    inputs = Input(shape=[40, 128])
    data = np.random.rand(BATCH_SIZE, 40, 128).astype(np.float32)
    model = Model(inputs=inputs, outputs=Toy_layer(40, name)(inputs))
    output = model(data)
    model.compile(optimizer=Adam(lr=1e-3), loss='mean_absolute_error')
    target = tf.convert_to_tensor(np.zeros((BATCH_SIZE)), dtype=tf.float32)
    model.fit(x=data, y=target, epochs=100, steps_per_epoch=1)
    return model


if __name__ == '__main__':
    # data -> [batch_size, features, time_frames]
    BATCH_SIZE = 512
    FEATURES = 40
    TIME_FRAMES = 128
    CHANNELS = 32
    WHATEVER = 10

    # random data
    def get_random_data():
        data = np.zeros([BATCH_SIZE, FEATURES, TIME_FRAMES]).astype(np.float32)
        for i in data:
            for j in np.random.randint(FEATURES, size=20):
                b = np.random.randint(TIME_FRAMES, size=1)[0].astype(np.int)
                e = np.random.randint(b, b+25, size=1)[0].astype(np.int)
                i[j, b:e] = 1
        return data

    def what_to_expand(example):
        # THIS CAN BE DONE IN THE GENERATOR
        output = []
        active_index = np.max(example, axis=1).astype(bool)
        for j, a in enumerate(example[active_index, :]):
            frames_repetition = np.where(a == 1)[0]
            frames_repetition = np.split(
                frames_repetition,
                np.where(np.diff(frames_repetition) > 1)[0] + 1
            )
            for frames in frames_repetition:
                output.append([j, frames[0], frames[-1]])
        return output

    def to_sparse(batch):
        values = []
        indices = []
        for i in range(BATCH_SIZE):
            # THIS CAN BE DONE IN THE GENERATOR
            example = batch[i, :, :]
            active_index = np.max(example, axis=1).astype(bool)
            for j, a in enumerate(example[active_index, :]):
                frames_repetition = np.where(a == 1)[0]
                frames_repetition = np.split(
                    frames_repetition,
                    np.where(np.diff(frames_repetition) > 1)[0] + 1
                )
                for frames in frames_repetition:
                    for f in frames:
                        values.append(j)
                        indices.append([i, f])
        return indices, values

    batch_orig = get_random_data()
    indices, values = to_sparse(batch_orig)
    batch = [what_to_expand(batch_orig[i, :, :]) for i in range(BATCH_SIZE)]
    neuros = np.random.rand(FEATURES)   # to be learnt
    output_tensor = np.zeros([BATCH_SIZE, WHATEVER, TIME_FRAMES, CHANNELS])   # the bin dimension is not important

    # tf.sparse.SparseTensor
    # tf.sparse.sparse_dense_matmul

    # MAIN COMPUTATION I WANNA DO INSIDE THE LAYER
    for i in range(BATCH_SIZE):
        for v in batch[i, :, :]:
            output_tensor[i, :, v[1]:v[2], :] += np.tile(
                neuros[v[0]], [WHATEVER, v[2]-v[1], CHANNELS]
            )   # using this because in tf has to tile a trainable variable
    # The output_tensor will be then multiply for something that has the same dimension
