execfile('./setUp.py')

from keras.models import load_model
import configparser as cp
import json
import pickle
import ffmpeg
import objectPretrait as Obj
import os
import glob
import stftUtil as s
import mir_eval
import ffmpeg
import sys
import pickle
import numpy as np


def resultsfromAudioPath(audioPath, model, N=128, numBands=512,
                         sr_hz=8192, db='multi', compress=False):
    if db == 'multi':
        trackName = audioPath.split('/')[-1].split('.')[0]
    elif db == 'musdb':
        trackName = audioPath.split('/')[-2]
    print('doing file {}'.format(audioPath))
    signal_v, fe = ffmpeg.readMP3(audioPath, True, sr_hz)
    spec_m = s.stft(signal_v, hop=256)
    spec = np.abs(spec_m)[:, 0:numBands]
    spec = (spec - np.min(spec))/np.max(spec)
    if compress:
        spec = np.sqrt(spec)
    nbFrames = spec.shape[0]
    newSpec = np.zeros((nbFrames, numBands))
    cpt = 0
    for i in range(0, nbFrames - N+1, N):
        im = spec[i:i+N, :]
        im = (np.transpose(im)).reshape((1, 1, numBands, N))
        netPredict = model.predict(im)
        newSpec[i:i+N, :] = np.transpose((netPredict)[0, 0, :, :])
    if compress:
        newSpec = newSpec**2
    phase_m = (np.angle(spec_m))
    ampl_m = np.abs(spec_m)
    newSpec2 = np.empty((newSpec.shape[0], numBands+1))
    newSpec2[:, :numBands] = newSpec
    newSpec2[:, numBands] = newSpec[:, numBands-1]
    spectRecontruct_m = (
        ampl_m.max() * (newSpec2 + ampl_m.min())) * np.exp(1j * phase_m)
    audioReconstruct = s.istft(spectRecontruct_m, 256)
    audioReconstruct /= abs(audioReconstruct).max()

    return newSpec, audioReconstruct


def computeRes(model, descDataBase, dataType, pathData, N, numBands, srHz):
    res = {}
    for track in descDataBase:
        audioPath = descDataBase[track]['path']
        newSpec, audioRecons = resultsfromAudioPath(
            audioPath, model, N, numBands, srHz)
        d = dict()
        d['newSpec'] = newSpec
        d['audioRecons'] = audioRecons
        res[audioPath] = d
    return res


def evalOneSep(estimatedVoice, track):
    mix, _ = ffmpeg.readMP3(track, True, 8192)

    rest, _ = ffmpeg.readMP3(track.replace('mix', 'rest'), True, 8192)
    voice, _ = ffmpeg.readMP3(track.replace('mix', 'voice'), True, 8192)
    lenTot = min(
        estimatedVoice.shape[0], mix.shape[0], voice.shape[0], rest.shape[0])
    voice = voice[:lenTot]
    rest = rest[:lenTot]
    mix = mix[:lenTot]
    estimatedVoice = estimatedVoice[:lenTot]
    restEstimated = mix - estimatedVoice

    estimated = np.array([estimatedVoice, restEstimated])
    groundTruth = np.array([rest, voice])

    resSep = mir_eval.separation.bss_eval_sources(groundTruth, estimated)

    return resSep


def evalTestSep(descDataBase, results):

    sdrL = []
    sirL = []
    sarL = []
    for t in descDataBase:
        groundTruthPath = descDataBase[t]['voice']
        estimated = results[t+'/mix.wav']['audioRecons']
        sdr, sir, sar, perm = evalOneSep(estimated, t+'/mix.wav')
        sdrL.append(sdr[perm[0]])
        sirL.append(sir[perm[0]])
        sarL.append(sar[perm[0]])

    print("SAR : mean {}, std {}".format(np.mean(sarL), np.std(sarL)))
    print("SIR : mean {}, std {}".format(np.mean(sirL), np.std(sirL)))
    print("SDR : mean {}, std {}".format(np.mean(sdrL), np.std(sdrL)))
    return {"sir": sirL, "sar": sarL, "sdr": sdrL}


def compute(configFile):
    # Parse arguments
    p = cp.ConfigParser()
    p.read(configFile)

    general = p['GENERAL']
    database = general['database']
    storeName = general['storeName']
    dataType = general['dataType']
    N = general.getint('N')
    numBands = general.getint('numBands')
    srHz = general.getint('srHz')

    path = p['PATH']
    pathDesc = path['pathDesc']
    pathModel = path['pathModel']
    rootStore = path['rootStore']

    expe = p['EXPE']
    store = expe.getboolean('store')
    metric = expe['computeMetrics']

    pathData = path['pathData']
    if database == "musdb":
        allTracks = glob.glob(os.path.join(pathData, "*"))
        descDataBase = dict()
        for e in allTracks:
            descDataBase[e] = {'path': os.path.join(e, "mix.wav"),
                               'voice': os.path.join(e, "voice.wav")}
    else:
        with open(pathDesc) as desc:
            descDataBase = pickle.load(desc)

    model = load_model(pathModel)

    results = computeRes(model, descDataBase, dataType, pathData,
                         N, numBands, srHz)

    metrics = evalTestSep(descDataBase, results)

    if store:
        dirStore = os.path.join(rootStore, database+'_'+storeName)
        if not os.path.isdir(dirStore):
            os.mkdir(dirStore)
        pickle.dump(results, open(os.path.join(dirStore, 'results'), 'w'))
        pickle.dump(metrics, open(os.path.join(dirStore, 'metrics'), 'w'))


if __name__ == "__main__":
    configFile = sys.argv[1]
    compute(configFile)
