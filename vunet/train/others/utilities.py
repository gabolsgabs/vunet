from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau,
)
from vunet.train.config import config
import os


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return


def write_config(folder):
    fl = open(os.path.join(folder, "config.txt"), "w")
    fl.write(str(config))
    fl.close()
    return


def save_dir(t, name):
    folder = os.path.join(config.PATH_BASE, t, config.MODE)
    if config.MODE == "conditioned":
        folder = os.path.join(
            folder, "_".join((config.CONDITION, config.FILM_TYPE, config.CONTROL_TYPE))
        )
        name = "_".join([config.COND_INPUT, name])
    if config.MODE == "attention":
        folder = os.path.join(folder, "_".join((config.CONDITION, config.FILM_TYPE)))
        name = "_".join(
            [
                config.COND_MATRIX,
                str(config.TIME_ATTENTION),
                str(config.FREQ_ATTENTION),
                name,
            ]
        )
    name = name.rstrip("_")
    config.NAME = name
    folder = os.path.join(folder, name)
    create_folder(folder)
    return folder


def make_earlystopping():
    return EarlyStopping(
        monitor="val_loss",
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="min",
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True,
    )


def make_checkpoint(folder):
    folder = os.path.join(folder, "checkpoint")
    create_folder(folder)
    return ModelCheckpoint(
        # filepath=os.path.join(folder, 'ckpt_{epoch:02d}-{val_loss:.5f}'),
        filepath=os.path.join(folder, "ckpt"),
        verbose=1,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
    )


def make_reduce_lr():
    return ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        min_lr=1e-5,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        patience=config.REDUCE_PLATEAU_PATIENCE,
        verbose=1,
    )


def make_tensorboard(folder):
    folder = os.path.join(folder, "tensorboard")
    create_folder(folder)
    return TensorBoard(log_dir=folder, write_graph=True)


# visualize_callback = VisualizeCallback(
#     model, train_ds, test_ds, tensorboard_dir)
