from .helpers import istft, stft
import copy
import numpy as np


def griffin_lim(spec_mag, spec_phase, iterations=250, momentum=False):
    # momentum (boolean) paper recent (mais marche pas aussi bien)
    # Now Griffin-Lim dat
    p = copy.deepcopy(spec_phase)
    for i in range(iterations):
        S = spec_mag * np.exp(1j * p)
        inv_a = np.array(istft(S, 256))
        new_a = np.array(stft(inv_a, hop=256))
        new_p = np.angle(new_a)
        # Momentum-modified Griffin-Lim
        if momentum:
            p = new_p + ((i > 0) * (0.99 * (new_p - p)))
        else:
            p = new_p
    return p


def reconstruct_mag(mix_mag, mix_phase, pred_mag, iterations=0):
    recont_mag = (pred_mag * (np.max(mix_mag) - np.min(mix_mag)) +
                  np.min(mix_mag))
    # recont_mag = pred_mag*np.max(mix_mag)
    if iterations > 0:
        mix_phase = griffin_lim(recont_mag, mix_phase, iterations)
    recont_spec = recont_mag * np.exp(1j * mix_phase)
    recont_audio = istft(recont_spec, 256)
    recont_audio /= abs(recont_audio).max()
    return recont_audio


def reconstruct_log(mix_log, mix_phase, pred_log, mix_log_max, mix_log_min,
                    iterations=0):
    pred_log = pred_log*(mix_log_max - mix_log_min) + mix_log_min
    # pred_log = pred_log*mix_log_max
    pred_mag = np.power(10, pred_log)-1
    # mix_log = mix_log*(mix_log_max - mix_log_min) + mix_log_min
    mix_log = mix_log*mix_log_max
    mix_mag = np.power(10, mix_log)-1
    return reconstruct_mag(mix_mag, mix_phase, pred_mag, iterations)
