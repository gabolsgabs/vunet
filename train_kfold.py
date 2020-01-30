import copy
import DALI
from get_config import Train
from helpers import (set_split, get_lock)
import numpy as np
from train import run as run_train
import sys

KFOLD = 10


def run(config_file):
    config = Train(config_file)
    print(config.pipeline)
    print(config.net)

    dali = DALI.get_the_DALI_dataset(config.general['path_dali'])
    dali = set_split(dali, b=0, n=0)

    n = int(len(dali)/KFOLD)
    name = config.general['name']
    k = 0
    gpu_id_locked = get_lock()
    for b in np.arange(0, len(dali), n):
        for i in list(dali.keys()):
            dali[i].info['ground-truth'] = False
        for i in list(dali.keys())[b:b+n]:
            dali[i].info['ground-truth'] = True
        config.general['name'] = name + '_kfold_' + str(k)
        run_train(config=copy.deepcopy(config), dali=dali,
                  gpu_id_locked=gpu_id_locked)
        k += 1


if __name__ == "__main__":
    configFile = sys.argv[1]
    run(configFile)
