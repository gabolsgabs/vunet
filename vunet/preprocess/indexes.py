import numpy as np
import itertools
import os
from vunet.preprocess.config import config
from pathlib import Path
import logging


def chunks(l, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def get_indexes():
    logger = logging.getLogger('getting_indexes')
    logger.info('Computing the indexes')
    indexes = []
    try:
        files = [str(i) for i in Path(config.PATH_BASE).rglob('*features.npz')]
        for f in np.random.choice(files, len(files), replace=False):
            logger.info('Input points for track %s' % f)
            file_length = np.load(f)['vocals'].shape[1]  # in frames
            s = []
            name = f.split('/')[-2]
            for j in np.arange(0, file_length-config.N_FRAMES, config.STEP):
                s.append([name, j])
            s = list(
                np.asarray(s, dtype=object)[np.random.permutation(len(s))]
            )
            indexes += s
        logger.info('Chunking the data points')
        # chunking the indexes before mixing -> create groups of CHUNK_SIZE
        indexes = list(chunks(indexes, config.CHUNK_SIZE))
        logger.info('Shuffling')
        # mixing these groups
        np.random.shuffle(indexes)
        # joining the groups in a single vector
        logger.info('Shuffling')
        indexes = list(itertools.chain.from_iterable(indexes))
    except Exception as error:
        logger.error(error)
    return indexes


def main():
    logging.basicConfig(
        filename=os.path.join(config.PATH_INDEXES, 'getting_indexes.log'),
        level=logging.INFO
    )
    logger = logging.getLogger('getting_indexes')
    logger.info('Starting the computation')
    name = "_".join([
        'indexes', str(config.STEP), str(config.CHUNK_SIZE)
    ])
    indexes = get_indexes()
    logger.info('Saving')
    np.savez(
        os.path.join(config.PATH_INDEXES, name),
        indexes=indexes, config=str(config)
    )
    logger.info('Done!')
    return


if __name__ == "__main__":
    config.parse_args()
    main()
