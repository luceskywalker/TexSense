from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def butter_2nd(sig, fs, f_cut):
    """
    applies zero-lag 2nd order butterworth filter to a signal
    :param sig: signal to be filtered - np.array
    :param fs: sampling frequency of sig - integer
    :param f_cut: cut-off frequency (low-pass) - integer
    :return sig_filt: filtered signal - np array
    """
    # design filter
    nf = f_cut/(fs/2)
    b, a = butter(2, nf, btype='low')

    # apply filter
    sig_filt = filtfilt(b, a, sig, axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))
    return sig_filt

def low_pass_filter(sig, fs, f_cut, order):
    """
    applies zero-lag butterworth filter to a signal
    :param sig: signal to be filtered - np.array
    :param fs: sampling frequency of sig - integer
    :param f_cut: cut-off frequency (low-pass) - integer
    :param order: order of filter - integer
    :return sig_filt: filtered signal - np array
    """
    # design filter
    nf = f_cut/(fs/2)
    b, a = butter(order, nf, btype='low')

    # apply filter
    sig_filt = filtfilt(b, a, sig, axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))
    return sig_filt

def downsample(input_data_df, original_sampling_rate_hz, target_sampling_rate_hz):
    """Downsample the data in the input_data_df from the original sampling rate to a given target sampling rate."""
    data_array = input_data_df.values
    len_data = data_array.shape[0]
    current_x = np.linspace(0, len_data, len_data)
    data_array_downsampled = interp1d(current_x, data_array, axis=0)(
        np.linspace(0, len_data, int(len_data * target_sampling_rate_hz / original_sampling_rate_hz))
    )

    output_data_df = pd.DataFrame(data_array_downsampled, columns=input_data_df.columns)

    return output_data_df