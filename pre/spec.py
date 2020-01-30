import DALI
import numpy as np
import PhDUtilities as ut
from PhDUtilities.audio import read_MP3
from PhDUtilities.audio import stft_source_separation as stft
import os


class Param(object):
    """docstring for ."""

    def __init__(self, sr_hz=8192, fft_size=1024, hop=256):
        self.sr_hz = sr_hz
        self.fft_size = fft_size
        self.hop = hop

    def set_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def spec_complex(audio_file, p=None):
    """Compute the complex spectrum
        -> time_r = hop/sr_hz
        -> n_freq_bins -> connected with fftsize with 1024 -> 513 bins
    """
    error = False
    output = {'type': 'complex'}
    if not p:
        p = Param()
    try:
        audio, fe = read_MP3(audio_file, stereo2mono=True, sr_hz=p.sr_hz)
        output['spec_complex'] = stft(audio, fftsize=p.fft_size, hop=p.hop)
    except ValueError:
        print('error on ' + audio_file)
        error = True
    return output, error


def spec_mag(audio_file, p=None, norm=True):
    """Compute the normalize (by the max) mag spec and remove the last band"""
    error = False
    output = {}
    if not p:
        p = Param()
    try:
        tmp, error = spec_complex(audio_file, p)
        spec = tmp['spec_complex']
        n = spec.shape[1] - 1
        mag = np.abs(spec[:, :n])
        #  mag = mag / np.max(mag)
        #  betweens 0 and 1 (x - min(x)) / (max(x) - min(x))
        if norm:
            mx = np.max(mag)
            mn = np.min(mag)
            mag = ((mag - mn) / (mx-mn))
            output['norm_param'] = np.array([mx, mn])
        output['phase'] = np.angle(spec)
        output['spec_mag'] = mag
    except ValueError:
        print('error on ' + audio_file)
        error = True
    return output, error


def spec_mag_log(audio_file, p=None):
    """Compute the complex spectrum
        -> time_r = hop/sr_hz
    """
    error = False
    output = {}
    if not p:
        p = Param()
    try:
        tmp, error = spec_mag(audio_file, p, False)
        mag = tmp['spec']
        output['phase'] = tmp['phase']
        spec_log = np.log10(mag+1)
        mx = np.max(spec_log)
        mn = np.min(spec_log)
        output['norm_param'] = np.array([mx, mn])
        output['spec_log'] = (spec_log - mn) / (mx - mn)
        #  spec_log = spec_log / mx
    except ValueError:
        print('error on ' + audio_file)
        error = True
    # return error, spec_log, phase, mx, mn
    return error, output


def compute_spec(path_dali, function, **kwargs):
    p = Param()
    if kwargs:
        p.set_kwargs(kwargs)
    dali = DALI.get_the_DALI_dataset(path_dali)
    for uid, entry in dali.items():
        print(uid)
        path_audio = entry.info['audio']['path']
        files = ut.general.list_files_from_folder(path_audio, 'wav')
        for f in files:
            output, _ = function(f, p)
            for i in output:
                name = os.path.join(
                    path_audio, "_".join(
                        [f.split('/')[-1].replace(".wav", ""), i]))
                np.save(name, output[i].astype(np.float32))
    return


if __name__ == '__main__':
    path_dali = '/u/anasynth/meseguerbrocal/dataset/multitracks/v0/dali_ready/'
    compute_spec(path_dali, spec_mag)
