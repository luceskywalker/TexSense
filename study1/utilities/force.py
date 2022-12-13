import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from study1.utilities.signal import low_pass_filter

def segments_steps(force_df, fs):
    f_vert = pd.DataFrame(-force_df["Fz"])
    # get highly smoothed signal for peak detection
    highly_smoothed_sig = low_pass_filter(f_vert, fs, 5, 4)

    # grab all smooth peaks above 500 Newtons
    peaks, _ = find_peaks(highly_smoothed_sig.squeeze(), height=500)
    # remove first and last peaks to avoid partial GRFs
    # peaks = np.delete(peaks, [0, -1])

    # squeeze to np array for further processing
    slightly_smoothed_sig = f_vert.values.squeeze()

    # extract HS (search from peak to the left while GRF > 20)
    HS_list = []
    # extract TO (search from peak to the right while GRF > 20)
    TO_list = []

    for peak in peaks:
        peak_hs_copy = peak
        while (peak_hs_copy > 0) & (slightly_smoothed_sig[peak_hs_copy] > 20):
            peak_hs_copy = peak_hs_copy - 1
        HS_list.append(peak_hs_copy + 1)  # go one back

        peak_to_copy = peak
        while (peak_to_copy < len(slightly_smoothed_sig)) & (slightly_smoothed_sig[peak_to_copy] > 20):
            peak_to_copy = peak_to_copy + 1
            if peak_to_copy == len(slightly_smoothed_sig):
                break
        TO_list.append(peak_to_copy - 1)  # go on back

    HS_list = list(set(HS_list))  # removes possible duplicates
    HS_list.sort()

    TO_list = list(set(TO_list))  # removes possible duplicates
    TO_list.sort()

    # make sure to start with HS and end with TO
    if HS_list[0] > TO_list[0]:
        HS_list = HS_list[:-1]
        TO_list = TO_list[1:]
    elif HS_list[-1] > TO_list[-1]:
        HS_list = HS_list[:-1]
        TO_list = TO_list[1:]
    try:
        len(TO_list) == len(HS_list)
    except:
        print("different number of HS and TO detected, something is wrong here..")

    IC = np.array(HS_list)
    TO = np.array(TO_list)
    return IC, TO