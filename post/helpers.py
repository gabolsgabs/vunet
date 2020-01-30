import scipy
import numpy as np


def wav_write(x, file_name, Fs=8192):
    """
    Write the `x` array into a wave file.

    Parameters
    ----------
    x : 1D-array
      Data array.
    Fs : integer
      Sampling frequency.
    filename : string
      File name.
    """

    import wave
    import struct

    output = wave.open(file_name, 'w')
    output.setparams((1, 2, Fs, 0, 'NONE', 'not compressed'))

    values = []

    for i in range(0, len(x)):
        value = int(32767*x[i])
        packed_value = struct.pack('h', value)
        values.append(packed_value)

    value_str = b''.join(values)
    output.writeframes(value_str)
    output.close()
    return


def stft(x, fftsize=1024, hop=768, nbFft=None):
    w = scipy.hanning(fftsize+1)[:-1]
    size_spec = int((x.shape[0]-fftsize)/hop + 1)
    return np.array([np.fft.rfft(w*x[i*hop:i*hop+fftsize], nbFft)
                     for i in range(0, size_spec)])


def istft(S, hop_size):
    win_len = (S.shape[1]-1)*2
    num_frames = S.shape[0]
    N = (num_frames - 1) * hop_size + win_len
    s = np.zeros(N)
    for i in range(num_frames):
        s[i * hop_size:i * hop_size + win_len] += np.fft.irfft(S[i, :])
    return s
