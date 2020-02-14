""" TO BE ADAPTED"""
import logging
import tensorflow as tf
from vunet.train.others.utilities import (
    make_earlystopping, make_reduce_lr, make_tensorboard, make_checkpoint,
    save_dir, write_config
)
from vunet.train.config import config
from vunet.train.models.vunet_model import vunet_model
from vunet.train.models.unet_model import unet_model
import os

from vunet.train.others.lock import get_lock


logger = tf.get_logger()
logger.setLevel(logging.INFO)


def main():
    config.parse_args()
    save_path = save_dir('models', config.NAME)
    write_config(save_path)
    _ = get_lock()
    logger.info('Starting the computation')

    logger.info('Running training with config %s' % str(config))
    logger.info('Getting the model')
    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = vunet_model()
    latest = tf.train.latest_checkpoint(
        os.path.join(save_path, 'checkpoint'))
    if latest:
        model.load_weights(latest)
        logger.info("Restored from {}".format(latest))
    else:
        logger.info("Initializing from scratch.")

    logger.info('Preparing the genrators')
    # Here to be sure that has the same config
    from vunet.train.data_loader import dataset_generator
    ds_train = dataset_generator()
    ds_val = dataset_generator(val_set=True)

    logger.info('Starting training for %s' % config.NAME)

    # USE VAL_STEPS!!
    # https://www.tensorflow.org/tutorials/images/classification

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        steps_per_epoch=config.N_BATCH,
        epochs=config.N_EPOCH,
        callbacks=[
            make_earlystopping(),
            make_reduce_lr(),
            make_tensorboard(save_path),
            make_checkpoint(save_path)
        ])

    logger.info('Saving model %s' % config.NAME)
    model.save(os.path.join(save_path, config.NAME+'.h5'))
    logger.info('Done!')
    return


if __name__ == '__main__':
    main()
