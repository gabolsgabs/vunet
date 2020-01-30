from keras.backend import tf
from .reconstruct import reconstruct_mag
from keras.models import load_model
import mir_eval
import numpy as np
import os
from PhDUtilities.audio import read_MP3
from .helpers import wav_write

import sys
sys.path.append('../models/')
sys.path.append('../pipeline/')
from models.autopool import AutoPool1D
from pipelines.pipeline_keras import prepare_condition


def prepare_song_phoneme(spec, phonemes, config):
    N = config['N']
    size = spec.shape[0]
    x = np.zeros((size//N+1, *config['input_shape']), dtype=np.float32)
    c = np.zeros((size//N+1, *config['input_shape_cond'][::-1]), dtype=np.float32)
    for index, i in enumerate(np.arange(0, size, N)):
        x_ = spec[i:i+N].T
        c_ = phonemes[i:i+N]
        s = spec[i:i+N].T.shape[1]
        if s != N:
            x_ = np.zeros((config['input_shape'][:2]), dtype=np.float32)
            x_[:, :s] = spec[i:i+s].T
            c_ = np.zeros((config['N'], config['num_cond']), dtype=np.float32)
            c_[:s, :] = phonemes[i:i+s]
        x[index] = np.expand_dims(x_, axis=2)
        c[index] = prepare_condition(c_, config['cond_input'])

    if config['cond_input'] in ['binary', 'ponderate', 'energy']:
        if config['emb_type'] == 'dense':
            c = np.expand_dims(c, axis=1)
        if config['emb_type'] == 'cnn':
            c = np.expand_dims(c, axis=2)
    else:
        c = np.transpose(c, (0, 2, 1))
    return x, c


def prepare_song(spec, config):
    N = config['N']
    num_bands = config['num_bands']
    size = spec.shape[0]
    x = np.zeros((size//N+1, num_bands, N, 1), dtype=np.float32)
    # for i in range(0, (size//N+1) * N, N):
    for index, i in enumerate(np.arange(0, size, N)):
        x_ = spec[i:i+N].T
        if spec[i:i+N].T.shape[1] != N:
            x_ = np.zeros((num_bands, N), dtype=np.float32)
            x_[:, :spec[i:i+N].T.shape[1]] = spec[i:i+N].T
        x[index] = np.expand_dims(x_, axis=2)
    return x


def separate_audio(path_audio, model, params):
    pred_audio, pred_spec = isolate_spec(path_audio, model, params)
    if 'name' not in params:
        name = path_audio.split('/')[-1].replace('.mp3', '.wav')
    name = os.path.join(params['path_output'], params['name'])
    wav_write(pred_audio, name)
    return


def concatenate(data):
    output = np.array([])
    output = np.concatenate(data, axis=1)
    return output


def isolate_spec(path_audio, model, config):
    spec = np.load(os.path.join(path_audio, 'mixture_spec_mag.npy'))
    phase = np.load(os.path.join(path_audio, 'mixture_phase.npy'))
    phone = np.load(os.path.join(path_audio, 'phonemes_matrix.npy'))
    pred_audio = np.array([])
    pred_spec = np.array([])

    if config['net_type'] == 'cond':
        x, p = prepare_song_phoneme(spec, phone, config)
        pred_spec = model.predict([x, p])
    if config['net_type'] == 'no_cond':
        x = prepare_song(spec, config)
        pred_spec = model.predict(x)

    s = spec.shape[0]
    pred_spec = np.transpose(np.squeeze(concatenate(pred_spec), axis=-1))[:s, :]
    pred_audio = reconstruct_mag(
        spec, phase, np.hstack((pred_spec, pred_spec[:, -1:])))
    return pred_audio, pred_spec


def do_an_exp(path_audio, model, config):
    mix, _ = read_MP3(os.path.join(path_audio, 'mixture.wav'),
                      stereo2mono=True, sr_hz=8192)
    orig, _ = read_MP3(os.path.join(path_audio, 'vocals.wav'),
                       stereo2mono=True, sr_hz=8192)
    rest, _ = read_MP3(os.path.join(path_audio, 'accompaniment.wav'),
                       stereo2mono=True, sr_hz=8192)

    pred_audio, pred_spec = isolate_spec(path_audio, model, config)
    tmp = min(pred_audio.shape[0], mix.shape[0], orig.shape[0], rest.shape[0])
    orig = orig[:tmp]   # original isolate source
    rest = rest[:tmp]   # accompaniment (sum of all apart from the original)
    mix = mix[:tmp]     # original mix
    pred_audio = pred_audio[:tmp]   # predicted separation
    rest_pred = mix - pred_audio
    estimated = np.array([pred_audio, rest_pred])
    ground_truth = np.array([orig, rest])

    """
    path_save = '/u/anasynth/meseguerbrocal/tmp/audio/'
    wav_write(pred_audio, path_save + 'pred_audio.wav')
    wav_write(rest_pred, path_save + 'rest_pred.wav')
    wav_write(orig, path_save + 'orig.wav')
    wav_write(rest, path_save + 'rest.wav')
    """

    return mir_eval.separation.bss_eval_sources(
        reference_sources=ground_truth, estimated_sources=estimated,
        compute_permutation=False)


def get_stats(dict, stat):
    values = np.fromiter(dict.values(), dtype=float)
    results = {'mean': np.mean(values), 'std': np.std(values)}
    print(stat+" : mean {}, std {}".format(results['mean'], results['std']))
    return results


def get_results(files, dali, path_model, config):
    results = {'sdr': {}, 'sir': {}, 'sar': {}}
    try:
        model = load_model(path_model)
    except Exception as e:
        model = load_model(
            path_model, custom_objects={"tf": tf, 'AutoPool1D': AutoPool1D})
    for i, f in enumerate(files):
        uid = f.split('/')[-3].replace(" ", "_")
        name = dali[uid].info['artist'] + ' - ' + dali[uid].info['title']
        print('Analyzing ' + name)
        print('Song num: ' + str(i+1) + ' out of ' + str(len(files)))
        sdr, sir, sar, perm = do_an_exp(f, model, config)
        results['sdr'][name] = sdr[perm[0]]
        results['sir'][name] = sir[perm[0]]
        results['sar'][name] = sar[perm[0]]
        print(sdr[perm[0]], sir[perm[0]], sar[perm[0]])
    results['sdr'].update(get_stats(results['sdr'], 'SDR'))
    results['sir'].update(get_stats(results['sir'], 'SIR'))
    results['sar'].update(get_stats(results['sar'], 'SAR'))
    return results
