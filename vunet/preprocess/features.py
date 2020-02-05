import DALI
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from glob import glob
from vunet.preprocess.config import config
import logging
import os
import librosa


# ADD RACHELS -> medley /net/assdb/data/mir2/2017_MedleyDB_ALL
# my audio -> /net/guzheng/data2/anasynth_nonbp/meseguerbrocal/source_separation/multitracks
# Look at /Users/meseguerbrocal/Documents/PhD/datasets/multitracks/


def get_config_as_str():
    return {
        'FR': config.FR, 'FFT_SIZE': config.FFT_SIZE,
        'HOP': config.HOP
    }


def compute_target(entry, dur, time_r, t):
    g = 'phonemes'
    # t = 'chars', 'phonemes', 'phoneme_types', 'notes'
    if t == 'chars' or t == 'notes':
        g = 'notes'
    return entry.get_annot_as_matrix(
        time_r, dur=dur, t=t, g=g).astype(np.int16)


def spec_complex(audio_file):
    """Compute the complex spectrum"""
    output = {'type': 'complex'}
    logger = logging.getLogger('computing_spec')
    try:
        logger.info('Computing complex spec for %s' % audio_file)
        audio, fe = librosa.load(audio_file, sr=config.FR)
        output['spec'] = librosa.stft(
            audio, n_fft=config.FFT_SIZE, hop_length=config.HOP)
    except Exception as my_error:
        logger.error(my_error)
    return output


def spec_mag(audio_file, norm=True):
    """Compute the normalized mag spec and the phase of an audio file"""
    output = {}
    logger = logging.getLogger('computing_spec')
    try:
        spec = spec_complex(audio_file)
        spec = spec['spec']
        logger.info('Computing mag and phase for %s' % audio_file)
        # n_freq_bins -> connected with fft_size with 1024 -> 513 bins
        # the number of band is odd -> removing the last band
        n = spec.shape[0] - 1
        mag = np.abs(spec[:n, :])
        #  mag = mag / np.max(mag)
        if norm:
            mx = np.max(mag)
            mn = np.min(mag)
            #  betweens 0 and 1 (x - min(x)) / (max(x) - min(x))
            mag = ((mag - mn) / (mx-mn))
            output['norm_param'] = np.array([mx, mn])
        output['phase'] = np.angle(spec)
        output['magnitude'] = mag
    except Exception as my_error:
        logger.error(my_error)
    return output


def spec_mag_log(audio_file):
    """Compute the normalized log mag spec and the phase of an audio file"""
    output = {}
    logger = logging.getLogger('computing_spec')
    try:
        spec = spec_mag(audio_file, False)    # mag without norm
        mag = spec['magnitude']
        output['phase'] = spec['phase']
        spec_log = np.log1p(mag)
        mx = np.max(spec_log)
        mn = np.min(spec_log)
        output['norm_param'] = np.array([mx, mn])
        output['log_magnitude'] = (spec_log - mn) / (mx - mn)
    except Exception as my_error:
        logger.error(my_error)
    return output


def compute_one_song(dali, folder):
    logger = logging.getLogger('computing_spec')
    name = folder.split('/')[-3]
    if name in dali:
        entry = dali[name]
        logger.info('Computing spec for %s' % name)
        data_input = {
            i.split('/')[-1].replace('.wav', ''): spec_complex(i)['spec']
            for i in glob(folder+'/*.wav')
        }
        time_r = config.TIME_R
        dur = data_input[list(data_input.keys())[0]].shape[1]*time_r - time_r
        data_target = {
            key: compute_target(entry, dur, time_r, key)
            for key in config.TARGETS
        }
        data_target['ncc'] = entry.info['scores']['NCC']
        data = {**data_input, **data_target}
        np.savez(
            os.path.join(config.PATH_BASE, name+'/features.npz'),
            config=get_config_as_str(), **data
        )
    return


def main():
    dali = DALI.get_the_DALI_dataset(config.PATH_DALI)
    logging.basicConfig(
        filename=os.path.join(config.PATH_BASE, 'computing_spec.log'),
        level=logging.INFO
    )
    logger = logging.getLogger('computing_spec')
    logger.info('Starting the computation')

    Parallel(n_jobs=16, verbose=5)(
        delayed(compute_one_song)(dali=dali, folder=str(i))
        for i in Path(config.PATH_BASE).rglob('*merged*'))
    return


if __name__ == '__main__':
    config.parse_args()
    main()