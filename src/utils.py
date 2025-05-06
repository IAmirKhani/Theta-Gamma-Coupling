from scipy.stats import binned_statistic
import scipy
import numpy as np
import matplotlib.pyplot as plt

import emd as emd
import emd.sift as sift
import emd.spectra as spectra
import scipy.stats
import pynapple as nap
import sails

from scipy.io import loadmat
import numpy as np
from neurodsp.filt import filter_signal
import copy
import emd
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.io import loadmat
from scipy.stats import entropy
import os
import re
from detect_pt import detect_phasic, preprocess, get_start_end


def extract_frequency_sampling(lfp, hypno):
    fs = len(lfp)/len(hypno)

    return int(fs)


def get_data(lfp_path, state_path):

    data = scipy.io.loadmat(lfp_path)
    states = scipy.io.loadmat(state_path)

    lfp = np.squeeze(data['HPC'])
    hypno = np.squeeze(states['states'])

    fs = extract_frequency_sampling(lfp, hypno)

    unique = np.unique(hypno)
    if unique[0] == 0:
        print('There was 0 in the dataset')
        lfp = lfp[7*fs:-11*fs]
        hypno = hypno[7:-11]
    else:
        None

    return lfp, hypno, fs


def plot_hypnogram(hypno):
    labels = {1: "Wake", 3: "NREM", 4: "Intermediate", 5: "REM"}
    plt.figure(figsize=(12, 6))
    time = np.arange(len(hypno)) / 60
    plt.step(time, hypno)
    plt.xlabel('Time (m)')
    plt.yticks(list(labels.keys()), list(labels.values()))
    plt.ylabel('States')
    plt.title('Hypnogram of sleep')
    plt.show()


def imf_freq(imf, sample_rate, mode='nht'):
    _, IF, _ = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
    freq_vec = np.mean(IF, axis=0)
    return freq_vec


def extract_imfs_by_pt_intervals(lfp, fs, interval, config, return_imfs_freqs=False):

    all_imfs = []
    all_imf_freqs = []
    rem_lfp = []
    all_masked_freqs = []
    for ii in range(len(interval)):
        start_idx = int(interval.loc[ii, 'start'] * fs)
        end_idx = int(interval.loc[ii, 'end'] * fs)
        sig_part = lfp[start_idx:end_idx]
        sig = np.array(sig_part)

        rem_lfp.append(sig)

        try:
            imf, mask_freq = sift.mask_sift(sig, **config)
        except Exception as e:
            print(f"EMD Sift failed: {e}. Skipping this interval.")
            continue
        all_imfs.append(imf)
        all_masked_freqs.append(mask_freq)

        imf_frequencies = imf_freq(imf, fs)
        all_imf_freqs.append(imf_frequencies)

    if return_imfs_freqs:
        return all_imfs, all_imf_freqs, rem_lfp
    else:
        return all_imfs


def tg_split(mask_freq, theta_range=(5, 12)):
    """
        Split a frequency vector into sub-theta, theta, and supra-theta components.

        Parameters:
        mask_freq (numpy.ndarray): A frequency vector or array of frequency values.
        theta_range (tuple, optional): A tuple defining the theta frequency range (lower, upper).
            Default is (5, 12).

        Returns:
        tuple: A tuple containing boolean masks for sub-theta, theta, and supra-theta frequency components.

        Notes: - This function splits a frequency mask into three components based on a specified theta frequency
        range. - The theta frequency range is defined by the 'theta_range' parameter. - The resulting masks 'sub',
        'theta', and 'supra' represent sub-theta, theta, and supra-theta frequency components.
    """
    lower = np.min(theta_range)
    upper = np.max(theta_range)
    mask_index = np.logical_and(mask_freq >= lower, mask_freq < upper)
    sub_mask_index = mask_freq < lower
    supra_mask_index = mask_freq > upper
    sub = sub_mask_index
    theta = mask_index
    supra = supra_mask_index

    return sub, theta, supra


def compute_range(x):
    return x.max() - x.min()


def asc2desc(x):
    pt = emd.cycles.cf_peak_sample(x, interp=True)
    tt = emd.cycles.cf_trough_sample(x, interp=True)
    if (pt is None) or (tt is None):
        return np.nan
    asc = pt + (len(x) - tt)
    desc = tt - pt
    return asc / len(x)


def peak2trough(x):
    des = emd.cycles.cf_descending_zero_sample(x, interp=True)
    if des is None:
        return np.nan
    return des / len(x)


def extract_subsets(arr, max_size):
    subsets = []
    current_subset = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1 and len(current_subset) < max_size:
            current_subset.append(arr[i])
        else:
            subsets.append(current_subset)
            current_subset = [arr[i]]

    # Add the last subset
    if current_subset:
        subsets.append(current_subset)

    return subsets


def bin_tf_to_fpp(x, power, bin_count):
    """
       Bin time-frequency power data into Frequency Phase Power (FPP) plots using specified time intervals of cycles.

       Parameters:
       x (numpy.ndarray): A 1D or 2D array specifying time intervals of cycles for binning.
           - If 1D, it represents a single time interval [start, end].
           - If 2D, it represents multiple time intervals, where each row is [start, end].
       power (numpy.ndarray): The time-frequency power spectrum data to be binned.
       bin_count (int): The number of bins to divide the time intervals into.

       Returns:
       fpp(numpy.ndarray): Returns FPP plots

       Notes:
       - This function takes time-frequency power data and divides it into FPP plots based on specified
         time intervals.
       - The 'x' parameter defines the time intervals, which can be a single interval or multiple intervals.
       - The 'power' parameter is the time-frequency power data to be binned.
       - The 'bin_count' parameter determines the number of bins within each time interval.
       """

    if x.ndim == 1:  # Handle the case when x is of size (2)
        bin_ranges = np.arange(x[0], x[1], 1)
        fpp = binned_statistic(
            bin_ranges, power[:, x[0]:x[1]], 'mean', bins=bin_count)[0]
        # Add an extra dimension to match the desired output shape
        fpp = np.expand_dims(fpp, axis=0)
    elif x.ndim == 2:  # Handle the case when x is of size (n, 2)
        fpp = []
        for i in range(x.shape[0]):
            bin_ranges = np.arange(x[i, 0], x[i, 1], 1)
            fpp_row = binned_statistic(
                bin_ranges, power[:, x[i, 0]:x[i, 1]], 'mean', bins=bin_count)[0]
            fpp.append(fpp_row)
        fpp = np.array(fpp)
    else:
        raise ValueError("Invalid size for x")

    return fpp


def plot_cycles(imf, sig, ctrl, inds):
    xinds = np.arange(len(inds))
    plt.figure(figsize=(8, 6))
    plt.plot(xinds, sig[inds], color=[0.8, 0.8, 0.8], label="Raw LFP")
    theta_part = imf[inds, 5]
    plt.plot(xinds, theta_part, label="IMF-6")

    plt.scatter(ctrl, theta_part[ctrl], color='red',
                marker='o', label='Control Points')
    plt.ylim([-800, 800])
    plt.legend()
    plt.show()


def load_mat_data(path_to_data, file_name, states_file):
    data = loadmat(path_to_data + file_name)
    data = data['PFClfpCleaned'].flatten()

    states = loadmat(path_to_data + states_file)
    states = states['states'].flatten()
    return data, states


def get_first_NREM_epoch(arr, start):
    start_index = None
    for i in range(start, len(arr)):
        if arr[i] == 3:
            if start_index is None:
                start_index = i
        elif arr[i] != 3 and start_index is not None:
            return (start_index, i - 1, i)

    return (start_index, len(arr) - 1, len(arr)) if start_index is not None else None


def get_all_NREM_epochs(arr):
    nrem_epochs = []
    next_start = 0
    while next_start < len(arr)-1:
        indices = get_first_NREM_epoch(arr, next_start)
        if indices == None:
            break
        start, end, next_start = indices
        if end-start <= 30:
            continue
        nrem_epochs.append([start, end])
    return nrem_epochs


def get_filtered_epoch_data(data, epochs, band=(0.1, 4), fs=2500):
    epoch_data = []
    for start, end in epochs:
        data_part = data[start*fs:end*fs]
        epoch_data.extend(data_part)
    epoch_data = np.array(epoch_data)
    filtered_epoch_data = filter_signal(
        epoch_data, fs, 'bandpass', band, n_cycles=3, filter_type='iir', butterworth_order=6, remove_edges=False)
    return filtered_epoch_data, epoch_data


def get_cycles_with_conditions(cycles, conditions):
    C = copy.deepcopy(cycles)
    try:
        C.pick_cycle_subset(conditions)
    except ValueError as e:
        print(f"No cycles satisfy the conditions: {e}")
        return None
    return C


def peak_before_trough(arr):
    trough_val = np.min(arr)
    trough_pos = np.argmin(arr)
    for i in range(trough_pos - 1, 0, -1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= 0:
            return arr[i]
    return -1


def peak_before_trough_pos(arr):
    trough_val = np.min(arr)
    trough_pos = np.argmin(arr)
    for i in range(trough_pos - 1, 0, -1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= 0:
            return i
    return -1


def peak_to_trough_duration(arr):
    trough_val = np.min(arr)
    trough_pos = np.argmin(arr)
    for i in range(trough_pos - 20, 0, -1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] >= 0:
            return trough_pos-i
    return -1


def num_inflection_points(arr):
    sign_changes = np.diff(np.sign(np.diff(arr, 2)))
    num_inflection_points = np.sum(sign_changes != 0)
    return num_inflection_points


def get_cycles_with_metrics(cycles, data, IA, IF, conditions=None):
    C = copy.deepcopy(cycles)

    C.compute_cycle_metric('duration_samples', data,
                           func=len, mode='augmented')
    C.compute_cycle_metric('peak2trough', data,
                           func=peak2trough, mode='augmented')
    C.compute_cycle_metric('asc2desc', data, func=asc2desc, mode='augmented')
    C.compute_cycle_metric('max_amp', IA, func=np.max, mode='augmented')
    C.compute_cycle_metric('trough_values', data,
                           func=np.min, mode='augmented')
    C.compute_cycle_metric('peak_values', data, func=np.max, mode='augmented')
    C.compute_cycle_metric('mean_if', IF, func=np.mean, mode='augmented')
    C.compute_cycle_metric('max_if', IF, func=np.max, mode='augmented')
    C.compute_cycle_metric(
        'range_if', IF, func=compute_range, mode='augmented')
    C.compute_cycle_metric('trough_position', data,
                           func=np.argmin, mode='augmented')
    C.compute_cycle_metric('peak_position', data,
                           func=np.argmax, mode='augmented')

    return C


def get_cycle_inds(cycles, subset_indices):

    all_cycles_inds = []
    for idx in subset_indices:
        if idx != -1:
            inds = cycles.get_inds_of_cycle(idx, mode='augmented')
            all_cycles_inds.append(inds)
    return all_cycles_inds


def get_cycle_ctrl(ctrl, subset_indices):
    all_cycles_ctrl = []
    for idx in subset_indices:
        if idx != -1:
            ctrl_inds = np.array(ctrl[idx], dtype=int)
            all_cycles_ctrl.append(ctrl_inds)
    return all_cycles_ctrl


def arrange_cycle_inds(all_cycles_inds):
    cycles_inds = []
    for ii in range(len(all_cycles_inds)):
        cycle = all_cycles_inds[ii]
        start = cycle[0]
        end = cycle[-1]
        cycles_inds.append([start, end])

    cycles_inds = np.array(cycles_inds)

    return cycles_inds


def compute_mode_frequency_and_entropy(FPP, frequencies, angles):
    mode_frequencies = []
    enropy_values = []

    for fpp in FPP:

        fpp = np.abs(fpp)

        fpp2_sum = np.sum(fpp)
        normalized_fpp2 = fpp / fpp2_sum

        max_index = np.unravel_index(
            np.argmax(normalized_fpp2, axis=None), fpp.shape)
        mode_frequency = frequencies[max_index[0]]

        avg_fpp2 = np.sum(normalized_fpp2, axis=1)
        window_size = 5
        smoothed_avg_fpp = np.convolve(avg_fpp2, np.ones(
            window_size)/window_size, mode='same')

        smoothed_avg_fpp_norm = (smoothed_avg_fpp - np.min(smoothed_avg_fpp)) / \
            (np.max(smoothed_avg_fpp) - np.min(smoothed_avg_fpp))
        dist_smoothed_avg_fpp_norm = smoothed_avg_fpp_norm / \
            np.sum(smoothed_avg_fpp_norm)

        shannon_entropy = entropy(dist_smoothed_avg_fpp_norm, base=2)
        enropy_values.append(shannon_entropy)

        mode_frequencies.append(mode_frequency)

    return np.array(mode_frequencies), np.array(enropy_values)


def abids(X, k):
    search_struct = cKDTree(X)
    return np.array([abid(X, k, x, search_struct) for x in X])


def abid(X, k, x, search_struct, offset=1):
    neighbor_norms, neighbors = search_struct.query(x, k+offset)
    neighbors = X[neighbors[offset:]] - x
    normed_neighbors = neighbors / neighbor_norms[offset:, None]
    # Original publication version that computes all cosines
    # coss = normed_neighbors.dot(normed_neighbors.T)
    # return np.mean(np.square(coss))**-1
    # Using another product to get the same values with less effort
    para_coss = normed_neighbors.T.dot(normed_neighbors)
    return k**2 / np.sum(np.square(para_coss))


def extract_experiment_info(path_to_hpc):

    path_parts = os.path.normpath(path_to_hpc).split(os.sep)

    try:
        idx_for_abdel = path_parts.index('for Abdel')
    except ValueError:
        raise ValueError("The path does not contain 'for Abdel' directory.")

    dataset_type = path_parts[idx_for_abdel + 1]

    rat_number = path_parts[idx_for_abdel + 2]

    treatment_part = path_parts[idx_for_abdel + 3]
    if '_' not in treatment_part and ' ' not in treatment_part:

        treatment = treatment_part
    else:

        tokens = re.split(r'[_\-]', treatment_part)

        tokens = [t for t in tokens if not re.match(
            r'Rat\d*|SD\d*|Rat|Ephys|OS', t, re.IGNORECASE)]

        non_numeric_tokens = [t for t in tokens if not t.isdigit()]
        if non_numeric_tokens:

            treatment = non_numeric_tokens[-1]
        else:
            treatment = 'Unknown'

    post_trial_folder = path_parts[-2]
    post_trial_match = re.search(
        r'post_trial(\d+)', post_trial_folder, re.IGNORECASE)
    if post_trial_match:
        post_trial = post_trial_match.group(1)
    else:
        post_trial = 'Unknown'

    return {
        'dataset_type': dataset_type,
        'rat_number': rat_number,
        'treatment': treatment,
        'post_trial': post_trial
    }

def extract_pt_intervals(lfpHPC, hypno, fs=2500):
    targetFs = 500
    n_down = fs / targetFs
    start, end = get_start_end(hypno=hypno, sleep_state_id=5)
    rem_interval = nap.IntervalSet(start=start, end=end)
    fs = int(n_down * targetFs)
    t = np.arange(0, len(lfpHPC) / fs, 1 / fs)
    lfp = nap.TsdFrame(t=t, d=lfpHPC, columns=['HPC'])

    # Detect phasic intervals
    lfpHPC_down = preprocess(lfpHPC, n_down)
    phREM = detect_phasic(lfpHPC_down, hypno, targetFs)

    # Create phasic REM IntervalSet
    start, end = [], []
    for rem_idx in phREM:
        for s, e in phREM[rem_idx]:
            start.append(s / targetFs)
            end.append(e / targetFs)
    phasic_interval = nap.IntervalSet(start, end)

    # Calculate tonic intervals
    tonic_interval = rem_interval.set_diff(phasic_interval)
    print(f'Number of detected Tonic intrevals:{len(tonic_interval)}')
    # Apply a 100 ms duration threshold to tonic intervals
    min_duration = 0.1  # 100 ms in seconds
    durations = tonic_interval['end'] - tonic_interval['start']
    valid_intervals = durations >= min_duration
    tonic_interval = nap.IntervalSet(tonic_interval['start'][valid_intervals], tonic_interval['end'][valid_intervals])
    print(f'Number of detected Tonic intrevals after threshold:{len(tonic_interval)}')
    return phasic_interval, tonic_interval, lfp


def get_cycle_data(imf5, fs=2500):
    cycle_data = {"fs": None, 'theta_imf': None,
                       "IP": None, "IF": None, "IP": None, "cycles": None}


    # Get cycles using IP
    IP, IF, IA = emd.spectra.frequency_transform(imf5, fs, 'hilbert')
    C = emd.cycles.Cycles(IP)
    cycles = get_cycles_with_metrics(C, imf5, IA, IF)

    cycle_data['fs'] = fs
    cycle_data['theta_imf'] = imf5
    cycle_data['IP'] = IP
    cycle_data['IF'] = IF
    cycle_data['IA'] = IA
    cycle_data['cycles'] = cycles
    return cycle_data

def extract_cycle_info(imfs, imf_frequencies):

  all_FPPs = []
  all_cycles_se =[]
  all_cycles_ctrl = []

  theta_range = [5, 12]
  frequencies = np.arange(15, 141, 1)
  angles=np.linspace(-180,180,19)
  fs = 2500

  for idx, imf in enumerate(imfs):
    cycle_data = get_cycle_data(imf[:, 5], fs=2500)

    amp_thresh = np.percentile(cycle_data['IA'], 25) # higher than 25th percentile of the data
    lo_freq_duration = fs/5  # restrict the analysis to 5-12 Hz
    hi_freq_duration = fs/17

    conditions = ['is_good==1',
                        f'duration_samples<{lo_freq_duration}',
                        f'duration_samples>{hi_freq_duration}',
                        f'max_amp>{amp_thresh}']

    all_cycles = get_cycles_with_conditions(cycle_data['cycles'], conditions)

    subset_cycles_df = all_cycles.get_metric_dataframe(subset=True)
    subset_indices = subset_cycles_df['index'].values

    ctrl = emd.cycles.get_control_points(imf[:, 5], cycle_data['cycles'], mode='augmented')
    cycle_ctrls = get_cycle_ctrl(ctrl, subset_indices)
    all_cycles_ctrl.append(cycle_ctrls)
    all_cycles_inds = get_cycle_inds(all_cycles, subset_indices)
    cycles_inds = arrange_cycle_inds(all_cycles_inds)

    all_cycles_se.append(all_cycles_inds)

    freqs = imf_frequencies[idx]
    sub_theta, theta, supra_theta = tg_split(freqs, theta_range)
    supra_theta_sig = np.sum(imf.T[supra_theta], axis=0)

    # # Corrected Wavelet Transform Computation
    raw_data=sails.wavelet.morlet(supra_theta_sig, freqs=frequencies, sample_rate=fs, ncycles=5,ret_mode='power', normalise=None)
    supraPlot = scipy.stats.zscore(raw_data, axis=1)
    FPP = bin_tf_to_fpp(cycles_inds, supraPlot, bin_count=19)
    all_FPPs.append(FPP)

  return all_cycles_ctrl, all_cycles_se, all_FPPs

def extract_spectural_signiture(FPP):
  smoothed_avg_fpp_all = []

  # Process each cycle to extract its normalized smoothed average FPP
  for fpp in FPP:
      fpp = np.abs(fpp)
      fpp_sum = np.sum(fpp)
      normalized_fpp = fpp / fpp_sum if fpp_sum != 0 else fpp
      avg_fpp = np.sum(normalized_fpp, axis=1)

      # Smooth and normalize the avg_fpp
      window_size = 5  # 5 Hz window
      smoothed_avg_fpp = np.convolve(avg_fpp, np.ones(window_size)/window_size, mode='same')
      smoothed_avg_fpp_norm = (smoothed_avg_fpp - np.min(smoothed_avg_fpp)) / (np.max(smoothed_avg_fpp) - np.min(smoothed_avg_fpp))

      smoothed_avg_fpp_all.append(smoothed_avg_fpp_norm)

  return smoothed_avg_fpp_all

# version 1

def prepare_data_for_sa(imfs, imf_frequencies):

  all_FPPs = []

  theta_range = [5, 12]
  frequencies = np.arange(15, 141, 1)
  angles=np.linspace(-180,180,19)
  fs = 2500

  for idx, imf in enumerate(imfs):
    cycle_data = get_cycle_data(imf[:, 5], fs=2500)

    amp_thresh = np.percentile(cycle_data['IA'], 25) # higher than 25th percentile of the data
    lo_freq_duration = fs/5  # restrict the analysis to 5-12 Hz
    hi_freq_duration = fs/12

    conditions = ['is_good==1',
                        f'duration_samples<{lo_freq_duration}',
                        f'duration_samples>{hi_freq_duration}',
                        f'max_amp>{amp_thresh}']
    print(len(cycle_data['theta_imf']))
    all_cycles = get_cycles_with_conditions(cycle_data['cycles'], conditions)
    if all_cycles is None or all_cycles.chain_vect.size == 0:
        print(f"No valid cycles found for the current interval. Skipping...")
        continue
    
    # Check if any cycles satisfy the conditions - HERE CAN BE CHANGED
    if all_cycles is None or all_cycles.chain_vect.size == 0:
        print("No cycles satisfy the conditions.")
        return pd.DataFrame(), pd.DataFrame(), []
    
    subset_cycles_df = all_cycles.get_metric_dataframe(subset=True)
    subset_indices = subset_cycles_df['index'].values

    all_cycles_inds = get_cycle_inds(all_cycles, subset_indices)
    cycles_inds = arrange_cycle_inds(all_cycles_inds)

    freqs = imf_frequencies[idx]
    _, _, supra_theta = tg_split(freqs, theta_range)
    supra_theta_sig = np.sum(imf.T[supra_theta], axis=0)

    # # Corrected Wavelet Transform Computation
    raw_data=sails.wavelet.morlet(supra_theta_sig, freqs=frequencies, sample_rate=fs, ncycles=5,ret_mode='power', normalise=None)
    supraPlot = scipy.stats.zscore(raw_data, axis=1)
    FPP = bin_tf_to_fpp(cycles_inds, supraPlot, bin_count=19)
    all_FPPs.append(FPP)

  return all_FPPs

def extract_spectral_signatures_for_rat(rat_id):

    # Define the base path to OS Basic datasets
    base_path = '/Users/amir/Desktop/for Abdel/OS Basic'
    treatments = ['CN', 'HC', 'OD', 'OR']
    fs = 2500  # Sample frequency

    # Initialize lists to collect all FPPs
    all_phasic_FPPs = []
    all_tonic_FPPs = []

    rat_path = os.path.join(base_path, str(rat_id))

    # Check if the specified rat folder exists
    if not os.path.isdir(rat_path):
        print(f"Rat folder {rat_id} does not exist.")
        return None, None, None

    # Loop over each treatment for the specified rat
    counts = 0
    for treatment in treatments:
        treatment_path = os.path.join(rat_path, treatment)

        # Check if the treatment folder exists
        if not os.path.isdir(treatment_path):
            print(f"Treatment folder '{treatment}' for Rat {rat_id} does not exist. Skipping...")
            continue

        # Detect all trial folders in the treatment directory and filter for Post_Trial2 to Post_Trial5
        trial_folders = [
            f for f in os.listdir(treatment_path)
            if os.path.isdir(os.path.join(treatment_path, f)) and
            any(f"Post_Trial{num}" in f for num in range(2, 6))
        ]

        for trial_folder in trial_folders:
            trial_path = os.path.join(treatment_path, trial_folder)

            # Search for LFP and state files in the trial folder
            lfp_file = None
            state_file = None

            for file_name in os.listdir(trial_path):
                if 'HPC' in file_name and file_name.endswith('.mat'):
                    lfp_file = os.path.join(trial_path, file_name)
                elif 'states' in file_name and file_name.endswith('.mat'):
                    state_file = os.path.join(trial_path, file_name)

            # Ensure both LFP and state files were found
            if not lfp_file or not state_file:
                print(f"Missing LFP or state file in {trial_path}. Skipping...")
                continue

            # Extract trial number from folder name (assuming format "Post_TrialX")
            trial_number = int(trial_folder.split('Trial')[-1])

            # Load data using your custom functions
            try:
                lfpHPC, hypno, _ = get_data(lfp_file, state_file)

                # Extract phasic and tonic intervals, handling cases with no REM sleep
                try:
                    phasic_interval, tonic_interval, lfp_processed = extract_pt_intervals(lfpHPC, hypno, fs)
                    
                except ValueError as e:
                    print(f"No REM sleep found in {trial_folder} for Rat {rat_id}, Treatment {treatment}. Filling with empty intervals.")
                    phasic_interval, tonic_interval, lfp_processed = [[], [], []]

                # Extract IMFs and frequencies for phasic and tonic intervals if intervals are not empty
                if phasic_interval and tonic_interval:
                    # Ensure 'config' is defined
                    tonic_imfs, tonic_freqs, tonic_lpf = extract_imfs_by_pt_intervals(
                        lfp_processed, fs, tonic_interval, config, return_imfs_freqs=True)
                    phasic_imfs, phasic_freqs, phasic_lpf = extract_imfs_by_pt_intervals(
                        lfp_processed, fs, phasic_interval, config, return_imfs_freqs=True)

                    # Prepare FPP data for both phasic and tonic
                    phasic_fpps = prepare_data_for_sa(
                        phasic_imfs, phasic_freqs)
                    tonic_fpps = prepare_data_for_sa(
                        tonic_imfs, tonic_freqs)

                    # Collect FPPs
                    if phasic_fpps is not None:
                        all_phasic_FPPs.extend(phasic_fpps)
                    if tonic_fpps is not None:
                        all_tonic_FPPs.extend(tonic_fpps)
                else:
                    print(f"No phasic or tonic intervals found in {trial_folder}. Skipping...")

            except FileNotFoundError:
                print(f"Data not found in {trial_path}. Skipping...")

    return all_phasic_FPPs, all_tonic_FPPs

def extract_and_flatten_signatures(fpps):
    """Extracts and flattens spectral signatures from a list of FPPs."""
    spectral_signatures = [extract_spectural_signiture(fpp) for fpp in fpps]
    # Flatten the nested list of spectral signatures
    flattened_signatures = np.array([ss for sublist in spectral_signatures for ss in sublist])
    return flattened_signatures


def extract_data_for_rat(rat_id, config):
    # Define the base path to OS Basic datasets
    base_path = '/Users/amir/Desktop/for Abdel/OS Basic'
    fs = 2500  # Sample frequency

    # Initialize empty DataFrames for concatenation across all recordings and trials for the specified rat
    all_phasic_FPPs = []
    all_tonic_FPPs = []

    rat_path = os.path.join(base_path, str(rat_id))

    # Check if the specified rat folder exists
    if not os.path.isdir(rat_path):
        print(f"Rat folder {rat_id} does not exist.")
        return None, None

    # List all recording folders in the rat directory
    recording_folders = [
        f for f in os.listdir(rat_path)
        if os.path.isdir(os.path.join(rat_path, f))
    ]

    if not recording_folders:
        print(f"No recording folders found for Rat {rat_id}.")
        return None, None

    # Loop over each recording folder
    for recording_folder in recording_folders:
        print(f"Processing recording folder: {recording_folder}")
        recording_path = os.path.join(rat_path, recording_folder)

        # Use regular expressions to parse the folder name
        match = re.match(r'^Rat-OS-Ephys_(Rat\d+)_SD(\d+)_([\w-]+)_([\d-]+)$', recording_folder)
        if not match:
            print(f"Unexpected folder name format: {recording_folder}. Skipping...")
            continue

        rat_id_part = match.group(1)       # e.g., 'Rat6'
        sd_number = match.group(2)         # e.g., '4'
        condition = match.group(3)         # e.g., 'CON'
        date_part = match.group(4)         # e.g., '22-02-2018'

        rat_id_from_folder = ''.join(filter(str.isdigit, rat_id_part))

        # Check if rat_id_from_folder matches rat_id
        if rat_id_from_folder != str(rat_id):
            print(f"Rat ID mismatch in folder {recording_folder}. Expected Rat{rat_id}, found Rat{rat_id_from_folder}. Skipping...")
            continue

        # Detect all trial folders and filter for post_trial2 to post_trial5, considering various folder name formats
        trial_folders = [
            f for f in os.listdir(recording_path)
            if os.path.isdir(os.path.join(recording_path, f)) and
            re.search(r'(?i)post[\-_]?trial[\-_]?([2-5])', f)
        ]

        if not trial_folders:
            print(f"No trial folders found in {recording_folder}.")
            continue

        for trial_folder in trial_folders:
            print(f"Processing trial folder: {trial_folder}")
            trial_path = os.path.join(recording_path, trial_folder)

            # Search for LFP and state files in the trial folder
            lfp_file = None
            state_file = None

            for file_name in os.listdir(trial_path):
                if 'HPC' in file_name and file_name.endswith('.mat'):
                    lfp_file = os.path.join(trial_path, file_name)
                elif 'states' in file_name and file_name.endswith('.mat'):
                    state_file = os.path.join(trial_path, file_name)
                elif 'States' in file_name and file_name.endswith('.mat'):
                    state_file = os.path.join(trial_path, file_name)

            # Ensure both LFP and state files were found
            if not lfp_file or not state_file:
                print(f"Missing LFP or state file in {trial_path}. Skipping...")
                continue

            # Extract trial number from folder name
            trial_number_match = re.search(r'(?i)post[\-_]?trial[\-_]?([2-5])', trial_folder)
            if trial_number_match:
                trial_number = int(trial_number_match.group(1))
            else:
                print(f"Unable to extract trial number from folder name: {trial_folder}. Skipping...")
                continue

            # Load data using custom functions
            try:
                lfpHPC, hypno, _ = get_data(lfp_file, state_file)

                # Extract phasic and tonic intervals, handling cases with no REM sleep
                try:
                    phasic_interval, tonic_interval, lfp = extract_pt_intervals(lfpHPC, hypno)
                except ValueError as e:
                    print(f"No REM sleep found in {trial_folder} for Rat {rat_id}, Condition {condition}. Filling with empty intervals.")
                    phasic_interval, tonic_interval, lfp = [[], [], []]

                # Extract IMFs and frequencies for phasic and tonic intervals if intervals are not empty
                if phasic_interval and tonic_interval:
                    # Assume 'config' is defined elsewhere in your code
                    tonic_imfs, tonic_freqs, tonic_lpf = extract_imfs_by_pt_intervals(
                        lfp, fs, tonic_interval, config, return_imfs_freqs=True)
                    phasic_imfs, phasic_freqs, phasic_lpf = extract_imfs_by_pt_intervals(
                        lfp, fs, phasic_interval, config, return_imfs_freqs=True)

                    # Prepare UMAP data for both phasic and tonic
                    phasic_FPPs = prepare_data_for_sa(phasic_imfs, phasic_freqs)
                    tonic_FPPs = prepare_data_for_sa(tonic_imfs, tonic_freqs)

                    # Concatenate into combined DataFrames
                    all_phasic_FPPs.extend(phasic_FPPs)
                    all_tonic_FPPs.extend(tonic_FPPs)

            except FileNotFoundError:
                print(f"Data not found in {trial_path}. Skipping...")

    return all_phasic_FPPs, all_tonic_FPPs