import librosa
import tensorflow as tf
import mir_eval
import numpy as np
import os
import logging
import pandas as pd
import gc
from vunet.evaluation.config import config
from vunet.preprocess.config import config as config_prepro
from vunet.train.config import config as config_train
from vunet.preprocess.features import spec_complex
from vunet.evaluation.test_ids import ids
from vunet.train.load_data_offline import get_data
from vunet.train.load_data_offline import normlize_complex
from vunet.train.others.lock import get_lock


logging.basicConfig(level=logging.INFO)


def update_config():
    config_train.COND_INPUT = config.COND_INPUT
    config_train.FILM_TYPE = config.FILM_TYPE
    config_train.TIME_ATTENTION = config.TIME_ATTENTION
    config_train.FREQ_ATTENTION = config.FREQ_ATTENTION
    

def update_cond_shape(num_frames):
    reshape = False
    conditions_shape = {
        'phonemes': 40, 'phoneme_types': 9, 'notes': 97, 'chars': 29
    }
    config.Z_DIM = conditions_shape[config.CONDITION]
    if config.COND_INPUT == 'vocal_energy':
        config.Z_DIM = 1
    shape = (config.Z_DIM, num_frames)
    reshape_list = ['binary', 'mean_dur', 'mean_dur_norm', 'vocal_energy']
    if config.COND_INPUT in reshape_list:
        reshape = True
    if config.CONTROL_TYPE == 'dense' and reshape:
        shape = (1, config.Z_DIM)
    if config.CONTROL_TYPE == 'cnn' and reshape:
        shape = (config.Z_DIM, 1)
    config.COND_SHAPE = shape
    return


def prepare_cond(data):
    if config.COND_MATRIX == 'overlap':
        cond = data[:, :, 0]
    if config.COND_MATRIX == 'sequential':
        cond = data[:, :, 1]
    output = cond
    if config.COND_INPUT == 'binary':
        output = np.max(cond, axis=1)   # silence is not removed
    if config.COND_INPUT == 'mean_dur':
        output = np.mean(cond, axis=1)
    if config.COND_INPUT == 'mean_dur_norm':
        output = np.mean(cond, axis=1)
        output = output / np.max(output)
    if config.COND_INPUT == 'vocal_energy':
        # just an scalar
        output = np.mean(np.max(cond, axis=1))
    output = np.reshape(output, config.COND_SHAPE)
    return output


def istft(data):
    return librosa.istft(
        data, hop_length=config_prepro.HOP,
        win_length=config_prepro.FFT_SIZE
    )


def adapt_pred(pred, target):
    # normalization between target range
    pred = (
        (np.max(target) - np.min(target))
        * ((pred - np.min(pred))/(np.max(pred) - np.min(pred)))
        + np.min(target)
    )
    pred += np.mean(target) - np.mean(pred)  # center in center target
    return pred


def reconstruct(pred_mag, orig_mix_phase):
    pred_mag = pred_mag[:, :orig_mix_phase.shape[1]]
    pred_mag /= np.max(pred_mag)
    pred_spec = pred_mag * np.exp(1j * orig_mix_phase)
    return istft(pred_spec)


def prepare_a_song(spec, num_frames, num_bands):
    size = spec.shape[1]
    segments = np.zeros(
        (size//(num_frames-config.OVERLAP)+1, num_bands, num_frames, 1),
        dtype=np.float32
    )
    for index, i in enumerate(np.arange(0, size, num_frames-config.OVERLAP)):
        segment = spec[:num_bands, i:i+num_frames]
        tmp = segment.shape[1]
        if tmp != num_frames:
            segment = np.zeros((num_bands, num_frames), dtype=np.float32)
            segment[:, :tmp] = spec[:num_bands, i:i+num_frames]
        segments[index] = np.expand_dims(np.abs(segment), axis=2)
    return segments


def prepare_a_condition(cond, num_frames):
    size = cond.shape[1]
    num_bands = cond.shape[0]
    update_cond_shape(num_frames)
    segments = np.zeros(
        (size//(num_frames-config.OVERLAP)+1, *config.COND_SHAPE),
        dtype=np.float32
    )
    for ndx, i in enumerate(np.arange(0, size, num_frames-config.OVERLAP)):
        segment = np.zeros(
            (num_bands, num_frames, cond.shape[-1]), dtype=np.float32
        )
        lenght = cond[:, i:i+num_frames, :].shape[1]
        segment[:, :lenght, :] = cond[:, i:i+num_frames, :]
        segment = prepare_cond(segment)
        segments[ndx] = segment
    return segments


def separate_audio(path_audio, path_output, model, cond):
    y, _ = analize_spec(
        spec_complex(path_audio)['spec'], model, cond)
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    name = path_audio.split('/')[-1].replace('.mp3', '.wav')
    name = os.path.join(path_output, name)
    librosa.output.write_wav(name, y, sr=config_prepro.FR)
    return


def concatenate(data, shape):
    output = np.array([])
    if config.OVERLAP == 0:
        output = np.concatenate(data, axis=1)
    else:
        output = data[0]
        o = int(config.OVERLAP/2)
        f = 0
        if config.OVERLAP % 2 != 0:
            f = 1
        for i in range(1, data.shape[0]):
            output = np.concatenate(
                (output[:, :-(o+f), :], data[i][:, o:, :]), axis=1)
    if shape[0] % 2 != 0:
        # duplicationg the last bin for odd input mag
        output = np.vstack((output, output[-1:, :]))
    return output


def analize_spec(model, orig_mix_spec, cond):
    logger = logging.getLogger('results')
    pred_audio = np.array([])
    orig_mix_spec = normlize_complex(orig_mix_spec)
    orig_mix_mag = np.abs(orig_mix_spec)
    orig_mix_phase = np.angle(orig_mix_spec)
    pred_audio, pred_mag = None, None
    try:
        if config.MODE == 'standard':
            num_bands, num_frames = model.input_shape[1:3]
            x = prepare_a_song(orig_mix_mag, num_frames, num_bands)
            pred_mag = model(x)
        if config.MODE in ['conditioned', 'attention']:
            num_bands, num_frames = model.input_shape[0][1:3]
            x = prepare_a_song(orig_mix_mag, num_frames, num_bands)
            cond = prepare_a_condition(cond, num_frames)
            pred_mag = model.predict([x, cond])
        pred_mag = np.squeeze(
            concatenate(pred_mag, orig_mix_spec.shape), axis=-1)
        pred_audio = reconstruct(pred_mag, orig_mix_phase)
    except Exception as my_error:
        logger.error(my_error)
    return pred_audio, pred_mag


def compute_a_segment(time, value,  orig, pred):
    init, end = [int(i) for i in time.split('_')]
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        reference_sources=orig[:, init:end],
        estimated_sources=pred[:, init:end],
        compute_permutation=False
    )
    value.update({
        'sdr': sdr[perm[0]], 'sir': sir[perm[0]],
        'sar': sar[perm[0]],
    })
    return value


def metrics_per_segments(cond, orig, pred):
    metrics = ['sdr', 'sir', 'sar', 'dur']
    if config.COND_MATRIX == 'overlap':
        cond = cond[:, :, 0]
    if config.COND_MATRIX == 'sequential':
        cond = cond[:, :, 1]
    # Get audio segments
    segments = {}
    for t, i in enumerate(cond):
        pos = np.where(i == 1)[0]
        if len(pos) > 0:
            init = int(pos[0]*config_prepro.TIME_R*config_prepro.FR)
            for j in np.where(np.diff(pos) != 1)[0]:
                end = int(pos[j]*config_prepro.TIME_R*config_prepro.FR)
                tmp = "_".join((str(init), str(end)))
                segments.setdefault(tmp, {})
                segments[tmp].setdefault('features', [])
                segments[tmp]['features'].append(t)
                segments[tmp]['dur'] = (
                    end/config_prepro.FR - init/config_prepro.FR
                )
                init = int((pos[j]+1)*config_prepro.TIME_R*config_prepro.FR)
    # compute metric of each segment
    from joblib import Parallel, delayed
    tmp = Parallel(n_jobs=16, verbose=5)(
        delayed(compute_a_segment)(time, value, orig, pred)
        for time, value in segments.items()
    )
    # group the info per feature
    output = {}
    output.setdefault(40, {m: [] for m in metrics})
    for value in tmp:
        for i, f in enumerate(value['features']):
            output.setdefault(f, {m: [] for m in metrics})
            for m in metrics:
                output[f][m].append(value[m])
            if f != 39 and i == 0:     # silence features -> just once
                for m in metrics:
                    output[40][m].append(value[m])
    # mean per feature
    return {i: {
        m: np.mean(output[i][m]) if m != 'dur' else np.sum(output[i][m])
        for m in metrics
    } for i in output}


def do_an_exp(model, data):
    # audio original sources
    # target = istft(data['vocals'])
    # mix = istft(data['mix'])
    # acc = istft(data['acc'])
    target = istft(
        np.multiply(data['vocals'], data['normalization']['vocals'][0])
    )
    mix = istft(
        np.multiply(data['mix'], data['normalization']['mixture'][0])
    )
    acc = istft(
        np.multiply(data['acc'], data['normalization']['accompaniment'][0])
    )
    # predicted separation
    pred_audio, pred_mag = analize_spec(model, data['mix'], data['cond'])
    # to go back to the range of values of the original target
    pred_audio = adapt_pred(pred_audio, target)
    # size
    s = min(pred_audio.shape[0], target.shape[0], mix.shape[0], acc.shape[0])
    pred_acc = mix[:s] - pred_audio[:s]
    pred = np.array([pred_audio[:s], pred_acc])
    orig = np.array([target[:s], acc[:s]])
    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
        reference_sources=orig, estimated_sources=pred,
        compute_permutation=False
    )
    general = {'sdr': sdr[perm[0]], 'sir': sir[perm[0]], 'sar': sar[perm[0]]}
    segments = {}
    if config.PER_FEATURE:
        segments = metrics_per_segments(data['cond'], orig, pred)
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    gc.collect()
    return general, segments


def get_stats(dict, stat):
    logger = logging.getLogger('computing_spec')
    values = np.fromiter(dict.values(), dtype=float)
    r = {'mean': np.mean(values), 'std': np.std(values)}
    logger.info(stat + " : mean {}, std {}".format(r['mean'], r['std']))
    return r


def create_pandas(files, features):
    columns = ['name', 'sdr', 'sir', 'sar']
    columns += [
        j for i in range(features+1)    # for no silence feature
        for j in ['sdr_'+str(i), 'sir_'+str(i), 'sar_'+str(i), 'dur_'+str(i)]
    ]
    if config.MODE == 'standard':
        data = np.zeros((len(files), len(columns)))
    else:
        data = np.zeros((len(files), len(columns)))
    df = pd.DataFrame(data, columns=columns)
    df['name'] = df['name'].astype('str')
    df = df.replace(0, np.NaN)
    return df


def load_checkpoint(path_results):
    from vunet.train.models.unet_model import unet_model
    from vunet.train.models.vunet_model import vunet_model
    from vunet.train.models.vunet_attention_model import vunet_attention_model
    path_results = os.path.join(path_results, 'checkpoint')
    latest = tf.train.latest_checkpoint(path_results)
    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = vunet_model()
    if config.MODE == 'attention':
        model = vunet_attention_model()
    model.load_weights(latest)
    return model


def load_a_unet(target=None):
    from tensorflow.keras.models import load_model
    model = None
    if config.MODE == 'standard':
        name = config.MODEL_NAME
        path_results = os.path.join(config.PATH_MODEL, config.MODEL_NAME)
    elif config.MODE == 'conditioned':
        name = ''.join([config.COND_INPUT, config.MODEL_NAME]).rstrip('_')
        model_type = "_".join(
            [config.CONDITION, config.FILM_TYPE, config.CONTROL_TYPE]
        )
        path_results = os.path.join(config.PATH_MODEL, model_type, name)
    elif config.MODE == 'attention':
        name = "_".join([
            config.COND_MATRIX, str(config.TIME_ATTENTION),
            str(config.FREQ_ATTENTION), config.MODEL_NAME
        ]).rstrip('_')
        model_type = "_".join((config.CONDITION, config.FILM_TYPE))
        path_results = os.path.join(config.PATH_MODEL, model_type, name)
    path_model = os.path.join(path_results, name+'.h5')
    if os.path.exists(path_model):
        if config.MODE == 'standard':
            model = load_model(path_model)
        if config.MODE == 'conditioned':
            from vunet.train.models.autopool import AutoPool1D
            model = load_model(
                path_model, custom_objects={"tf": tf, 'AutoPool1D': AutoPool1D}
            )
        if config.MODE == 'attention':
            model = load_model(path_model, custom_objects={"tf": tf})
    else:
        model = load_checkpoint(path_results)
    return model, path_results


def store_data_in_pandas(data, df, mode, i=None):
    if mode == 'general':
        for metric in data:
            df.at[i, metric] = data[metric]
    if mode == 'features':
        for feature, metrics in data.items():
            for metric, value in metrics.items():
                if not np.isnan(value):
                    df.at[i, "_".join((metric, str(feature)))] = value
    return df


def main():
    # _ = get_lock()
    config.parse_args()
    # config_train.set_group(config.CONFIG)
    update_config()
    model, path_results = load_a_unet()
    songs = get_data(ids)
    results = create_pandas(
        list(songs.keys()), features=list(songs.values())[0]['cond'].shape[0]
    )
    file_handler = logging.FileHandler(
        os.path.join(path_results, 'results.log'),  mode='w')
    file_handler.setLevel(logging.INFO)
    logger = logging.getLogger('results')
    logger.addHandler(file_handler)
    logger.info('Starting the computation for model ' + path_results)
    for i, (name, data) in enumerate(songs.items()):
        logger.info('Song num: ' + str(i+1) + ' out of ' + str(len(results)))
        results.at[i, 'name'] = name
        logger.info('Analyzing ' + name)
        general, features = do_an_exp(model, data)
        results = store_data_in_pandas(general, results, 'general', i)
        if config.PER_FEATURE:
            results = store_data_in_pandas(features, results, 'features', i)
        logger.info(results.iloc[i])
    results.to_pickle(os.path.join(path_results, config.RESULTS_NAME))
    logger.removeHandler(file_handler)
    return


if __name__ == '__main__':
    main()
