from scipy.signal import butter, filtfilt


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