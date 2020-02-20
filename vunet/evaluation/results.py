import librosa
from tensorflow.keras.models import load_model
import mir_eval
import numpy as np
import os
import logging
import pandas as pd
from vunet.evaluation.config import config
from vunet.preprocess.config import config as config_prepro
from vunet.train.load_data_offline import normlize_complex
from vunet.preprocess.features import spec_complex
from vunet.evaluation.test_ids import ids
from vunet.train.load_data_offline import get_data


logging.basicConfig(level=logging.INFO)


def istft(data):
    return librosa.istft(
        data, hop_length=config_prepro.HOP,
        win_length=config_prepro.FFT_SIZE)


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
        # if config.MODE == 'conditioned':
        #     num_bands, num_frames = model.input_shape[0][1:3]
        #     x = prepare_a_song(orig_mix_mag, num_frames, num_bands)
        #     if config.EMB_TYPE == 'dense':
        #         cond = cond.reshape(1, -1)
        #     if config.EMB_TYPE == 'cnn':
        #         cond = cond.reshape(-1, 1)
        #     tmp = np.zeros((x.shape[0], *cond.shape))
        #     tmp[:] = cond
        #     pred_mag = model.predict([x, tmp])
        pred_mag = np.squeeze(
            concatenate(pred_mag, orig_mix_spec.shape), axis=-1)
        pred_audio = reconstruct(pred_mag, orig_mix_phase)
    except Exception as my_error:
        logger.error(my_error)
    return pred_audio, pred_mag


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
        compute_permutation=False)
    return sdr[perm[0]], sir[perm[0]], sar[perm[0]]


def get_stats(dict, stat):
    logger = logging.getLogger('computing_spec')
    values = np.fromiter(dict.values(), dtype=float)
    r = {'mean': np.mean(values), 'std': np.std(values)}
    logger.info(stat + " : mean {}, std {}".format(r['mean'], r['std']))
    return r


def create_pandas(files):
    columns = ['name', 'sdr', 'sir', 'sar']
    if config.MODE == 'standard':
        data = np.zeros((len(files), len(columns)))
    else:
        data = np.zeros((len(files), len(columns)))
    df = pd.DataFrame(data, columns=columns)
    df['name'] = df['name'].astype('str')
    return df


def load_checkpoint(path_results):
    import tensorflow as tf
    from vunet.train.models.unet_model import unet_model
    from vunet.train.models.vunet_model import vunet_model
    path_results = os.path.join(path_results, 'checkpoint')
    latest = tf.train.latest_checkpoint(path_results)
    if config.MODE == 'standard':
        model = unet_model()
    if config.MODE == 'conditioned':
        model = vunet_model()
    model.load_weights(latest)
    return model


def load_a_unet(target=None):
    model = None
    if config.MODE == 'standard':
        path_results = os.path.join(config.PATH_MODEL, config.MODEL_NAME)
        path_model = os.path.join(path_results, config.MODEL_NAME+'.h5')
        if os.path.exists(path_model):
            model = load_model(path_model)
        else:
            model = load_checkpoint(path_results)
    else:
        import tensorflow as tf
        path_results = os.path.join(config.PATH_MODEL, config.MODEL_NAME)
        path_model = os.path.join(path_results, config.MODEL_NAME+'.h5')
        if os.path.exists(path_model):
            model = load_model(path_model,  custom_objects={"tf": tf})
        else:
            model = load_checkpoint(path_results)
    return model, path_results


def main():
    config.parse_args()
    songs = get_data(ids)
    results = create_pandas(list(songs.keys()))
    model, path_results = load_a_unet()
    file_handler = logging.FileHandler(
        os.path.join(path_results, 'results.log'),  mode='w')
    file_handler.setLevel(logging.INFO)
    logger = logging.getLogger('results')
    logger.addHandler(file_handler)
    logger.info('Starting the computation')
    for i, (name, data) in enumerate(songs.items()):
        logger.info('Song num: ' + str(i+1) + ' out of ' + str(len(results)))
        results.at[i, 'name'] = name
        logger.info('Analyzing ' + name)
        (results.at[i, 'sdr'], results.at[i, 'sir'],
         results.at[i, 'sar']) = do_an_exp(model, data)
        logger.info(results.iloc[i])
    results.to_pickle(os.path.join(path_results, config.RESULTS_NAME))
    logger.removeHandler(file_handler)
    return


if __name__ == '__main__':
    main()
