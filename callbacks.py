from keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard)


def make_earlystopping(patience):
    return EarlyStopping(monitor='val_loss', min_delta=0, mode='min',
                         patience=patience, verbose=1,
                         restore_best_weights=True)


def make_checkpoit(name_ckpt):
    return ModelCheckpoint(name_ckpt, verbose=1, mode='min',
                           save_best_only=True, monitor='val_loss')


def make_tensorboard():
    return TensorBoard(log_dir='/u/anasynth/meseguerbrocal/tmp/logs',
                       write_graph=True)
