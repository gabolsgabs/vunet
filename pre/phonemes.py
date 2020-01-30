import DALI
import numpy as np
import PhDUtilities as ut
import os


PHONEMES_PAIRS = {
    'AA': 'vowel', 'AE': 'vowel', 'AH': 'vowel', 'AO': 'vowel', 'AW': 'vowel',
    'AY': 'vowel', 'B': 'stop', 'CH': 'affricate', 'D': 'stop',
    'DH': 'fricative', 'EH': 'vowel', 'ER': 'vowel', 'EY': 'vowel',
    'F': 'fricative', 'G': 'stop', 'HH': 'aspirate', 'IH': 'vowel',
    'IY': 'vowel', 'JH': 'affricate', 'K': 'stop', 'L': 'liquid', 'M': 'nasal',
    'N': 'nasal', 'NG': 'nasal', 'OW': 'vowel', 'OY': 'vowel', 'P': 'stop',
    'R': 'liquid', 'S': 'fricative', 'SH': 'fricative', 'T': 'stop',
    'TH': 'fricative', 'UH': 'vowel', 'UW': 'vowel', 'V': 'fricative',
    'W': 'semivowel', 'Y': 'semivowel', 'Z': 'fricative', 'ZH': 'fricative'
}

PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']


PHONEMES_TYPE = ['vowel', 'stop', 'affricate', 'fricative', 'aspirate',
                 'liquid', 'nasal', 'semivowel']


def for_guillaume(path_dali, path_save, thd=.8):
    import csv
    output = []
    dali = DALI.get_the_DALI_dataset(path_dali)
    for uid, entry in dali.items():
        pth = entry.info['audio']['path']
        path_audio = os.path.join(pth, 'vocals.wav')
        path_phonemes = os.path.join(pth, 'phonemes_annot.npy')
        annot = entry.annotations['annot']['phonemes']
        phonemes = [[i['time'], tuple(i['text'])] for i in annot]
        np.save(path_phonemes, phonemes)
        errors = []
        # CHANGE IF NEEDED!
        for i in entry.info['errors'].values():
            errors += i
        if entry.info['scores']['NCC'] >= thd and len(errors) == 0:
            output.append([path_audio, path_phonemes])

    with open(path_save+'for_guillaume.csv', mode='w') as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in output:
            csv_writer.writerow(i)
    return


def p2v(phonemes, t='raw'):
    v = []
    if t == 'raw':
        v = PHONEMES
    if t == 'type':
        v = PHONEMES_TYPE
    output = np.zeros(len(v)+1)
    for p in phonemes:
        if t == 'type':
            p = PHONEMES_PAIRS[p]
        i = v.index(p)
        output[i] += 1
    return output


def phonemes2vector(phonemes, duration, time_r, errors_v, errors_s, t='raw'):
    """Transforms the annotations into frame vector wrt a time resolution.

    Parameters
    ----------
        annot : list
            annotations only horizontal level
            (for example: annotations['annot']['lines'])
        dur : float
            duration of the vector (for adding zeros).
        time_r : float
            time resolution for discriticing the time.
        type : str
            'voice': each frame has a value 1 or 0 for voice or not voice.
            'notes': each frame has the freq value of the main vocal melody.
    """
    if t == 'raw':
        s = len(PHONEMES) + 1
    if t == 'type':
        s = len(PHONEMES_TYPE) + 1
    vector = np.zeros((int(duration / time_r), s))
    for p in phonemes:
        if p['index'] not in errors_v:
            b, e = p['time']
            b = np.round(b/time_r).astype(int)
            e = np.round(e/time_r).astype(int)
            vector[b:e+1] = p2v(p['text'], t)
    for error in errors_s:
        v = np.zeros(s)
        v[-1] = 1
        b = np.round(error[0]/time_r).astype(int)
        e = np.round(error[1]/time_r).astype(int)
        vector[b:e+1] = v
    return vector


def compute_phonemes(path_dali, time_r=0.03125):
    #  time_r = hop/sr_hz -> 256/8192
    dali = DALI.get_the_DALI_dataset(path_dali)
    for uid, entry in dali.items():
        print(uid)
        path_audio = entry.info['audio']['path']
        path_phonemes = os.path.join(path_audio, 'phonemes_matrix.npy')
        end = ut.audio.end_song(os.path.join(path_audio, 'vocals.wav'))
        annot = entry.annotations['annot']['phonemes']
        errors_s = entry.info['errors']['silence_segments']
        errors_v = entry.info['errors']['vocals_lines']
        phonemes = phonemes2vector(annot, end, time_r, errors_v, errors_s)
        np.save(path_phonemes, phonemes.astype(np.float32))
    return


if __name__ == '__main__':
    path_dali = '/u/anasynth/meseguerbrocal/dataset/multitracks/v0/dali_ready/'
    compute_phonemes(path_dali)
