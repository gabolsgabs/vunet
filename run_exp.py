import DALI
import gc
from get_config import Test
# from helpers import set_split
from post.results import get_results
import os
import PhDUtilities as ut
import sys


def run(config_file):
    config = Test(config_file)
    dali = DALI.get_the_DALI_dataset(config.general['path_dali'])
    """
    dali = set_split(dali)
    files = [dali[i].info['audio']['path'] for i in dali
             if dali[i].info['ground-truth']]
    """

    print(config.general['path_model'])
    models = ut.general.list_files_from_folder_recursive(
        config.general['path_model'], '.h5')

    for model in models:
        print("RUNING: " + model)
        m_config = os.path.join("/".join(model.split('/')[:-1]), 'config.gz')
        train_config = ut.general.read_gzip(m_config)
        if not ut.general.check_directory(config.general['path_results']):
            ut.general.create_directory(config.general['path_results'])

        # name = model.split('vunet')[1].replace('.h5', '').split('/')[2:]
        # tmp = model.split('vunet')[1].split('/')[1]
        # path_save = os.path.join(config.general['path_results'], tmp)+'/'
        # name = "_".join(name)
        path_save = config.general['path_results']
        name = model.split('/')[-1].replace('.h5', '')
        if not ut.general.check_directory(path_save):
            ut.general.create_directory(path_save)

        if not ut.general.check_file(os.path.join(path_save, name)+'.json'):
            results = get_results(
                m_config.pipeline['files_val'], dali, model, train_config.net)
            results['config'] = m_config
            # create_panda
            # write panda
            ut.general.write_in_json(path_save, name, results)

        _ = gc.collect()
    return


if __name__ == "__main__":
    configFile = sys.argv[1]
    run(configFile)
