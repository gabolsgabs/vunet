from callbacks import make_earlystopping, make_checkpoit
import DALI
from get_config import Train
from get_val import get_val_data
from helpers import (get_lock, get_indexes, get_indexes_val, set_split,
                     get_epochs)
import manage_gpus as gpl
from models.uNetModel import get_unet
from models.uNetModel_filmed_simple import get_unet_filmed as get_simple
from models.uNetModel_filmed_complex import get_unet_filmed as get_complex
import os
from pipelines.pipeline_keras import DataGenerator
import PhDUtilities as ut
import sys


# Metrics -> https://github.com/schufo/wiass


def run(config_file=None, dali=None, config=None, gpu_id_locked=None):
    if not config:
        config = Train(config_file)
    print(config.pipeline)
    print(config.net)

    if not dali:
        dali = DALI.get_the_DALI_dataset(config.general['path_dali'])
        dali = set_split(dali)

    print('PREPROCESSING: Creating train generator')
    config.pipeline['files'] = [dali[i].info['audio']['path'] for i in dali
                                if not dali[i].info['ground-truth']]

    indexes = get_indexes(**config.pipeline)
    g_train = DataGenerator(indexes, config.pipeline)

    # num of batchs for one epoch -> each bath has batch_size points
    n_epoch = get_epochs(config.pipeline['batch_size'],
                         config.pipeline['n_batch'], len(indexes), 10)
    patience = int(n_epoch*.5)

    print('PREPROCESSING: Creating val indexes')
    config.pipeline['files_val'] = [
        dali[i].info['audio']['path'] for i in dali
        if dali[i].info['ground-truth']]

    val_index = get_indexes_val(
        config.pipeline['files_val'], config.pipeline['N'])

    val_data = get_val_data(
        config.pipeline['files_val'], val_index, config.pipeline['batch_size'],
        config.pipeline['dim'], config.net['emb_type'],
        config.pipeline['net_type'], config.pipeline['num_cond'],
        config.pipeline['N'], config.pipeline['cond_input'],
        config.pipeline['dim_cond'])

    if gpu_id_locked is None:
        gpu_id_locked = get_lock()

    if config.net['net_type'] == 'cond':
        if config.net['film_type'] == 'simple':
            model = get_simple(**config.net)
        if config.net['film_type'] == 'complex':
            model = get_complex(**config.net)
    if config.net['net_type'] == 'no_cond':
        model = get_unet(**config.net)

    model.summary()

    from keras import backend as K
    print(K.get_value(model.optimizer.lr))
    try:
        print(model.layers[-2].activation)
    except Exception as e:
        print(model.layers[-2].get_config())

    config.general['path_output'] = os.path.join(
        config.general['path_output'],
        config.general['name'].split('_kfold_')[0])

    if not ut.general.check_directory(config.general['path_output'],
                                      print_error=False):
        ut.general.create_directory(config.general['path_output'])

    name = os.path.join(config.general['path_output'], config.general['name'])
    name_ckpt = name + '_{epoch:02d}-{val_loss:.5f}.h5'
    name_final = name + '.h5'

    earlystopping = make_earlystopping(patience)
    checkpoint = make_checkpoit(name_ckpt)
    # tensorbard = make_tensorboard()

    ut.general.write_in_gzip(
        config.general['path_output'], config.general['name'] + '_config',
        config)

    model.fit_generator(
        generator=g_train, steps_per_epoch=config.pipeline['n_batch'],
        epochs=n_epoch, validation_data=val_data, workers=1,
        use_multiprocessing=False, callbacks=[earlystopping, checkpoint])

    model.save(name_final)

    if gpu_id_locked >= 0:
        gpl.free_lock(gpu_id_locked)


if __name__ == "__main__":
    configFile = sys.argv[1]
    run(config_file=configFile)
