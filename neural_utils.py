from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os 
import mat73
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import correlate, correlation_lags

def create_gauss_kernel(winsize):
    dt = 0.01
    winlen = np.round(winsize / dt).astype(np.int16)
    k_gauss = np.empty((winlen,), dtype=np.float32)

    mu = (winlen - 1) / 2
    sigma = 2

    for i in range(winlen):
        k_gauss[i] = np.exp(-0.5 * (i - mu) ** 2 / sigma**2) / (sigma * np.sqrt(2 * np.pi))

    return k_gauss


def SVC_score_manual(X, Y, W, b):
    '''
    manually score the accuracy of a support vector classifier based on coefficients W and intercept b
    '''
    Y_labels = np.copy(Y)
    Y_labels[Y_labels == 0] = -1
    f = Y_labels.T * (W.T @ X.T + b) - 0.0  #what is this last value supposed to be? actually 0 is probably right
    N = Y.shape[0]
    return np.sum(f > 0) / N

def cross_validate_trigger(model, X, Y, kf, permutation):
    '''
    specific cross validation for a linear classifier (SVM)
    returns accuracies rather than MSE
    '''
    X = X[permutation]
    Y = Y[permutation]

    scores = []
    coeffs = []
    intercepts = []

    for i, (train, test) in enumerate(kf.split(X)):
        model.fit(X[train], Y[train])
        # scores.append(model.score(X[test], Y[test]))
        scores.append(SVC_score_manual(X[test], Y[test], model.coef_.T, model.intercept_))
        coeffs.append(model.coef_)
        intercepts.append(model.intercept_)

    return np.mean(scores), np.array(coeffs).mean(0), np.array(intercepts).mean(0)

def correlation_helper_fun(X, Y, t0, tf, T):
    '''
    handles demeaning everything and returns lags rescaled to fit T
    '''
    start_idx = np.argmin(np.abs(T - t0))
    end_idx = np.argmin(np.abs(T - tf))
    dt = np.mean(np.diff(T))

    assert np.min(np.abs(T - t0)) < dt, "could not find appropriate starting time for {}".format(t0)
    assert np.min(np.abs(T - tf)) < dt, "could not find appropriate ending time"

    X_corr = X[start_idx:end_idx]
    Y_corr = Y[start_idx:end_idx]

    lags = correlation_lags(X_corr.shape[0], Y_corr.shape[0])
    xcorr = correlate(X_corr - np.mean(X_corr), Y_corr - np.mean(Y_corr))

    lags = lags * dt

    return xcorr, lags

def center_scale_neural_data(X):
    '''
    X -> (trials, neurons)
    '''
    nonzero_neurons = np.argwhere(np.std(X, axis=0) > 0).squeeze()
    X = X[:, nonzero_neurons]
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def scale_neural_data(X):
    nonzero_neurons = np.std(X, axis=0) > 0
    X = X[:, nonzero_neurons]
    return X/np.std(X, axis=0), nonzero_neurons, np.std(X, axis=0, keepdims=True)

def select_behavioral_data_flexible(X, T, tf, winsize, rescale=True):
    '''
    modified from the other version to support a different time axis for each trial
    '''
    n_time, n_behavioral_dims, n_trials = X.shape

    if len(T.shape) == 1:
        T = np.tile(T, (n_trials, 1))

    t_idxs = np.full(T.shape, False)

    idx_tf = np.argmin(np.abs(T - tf), axis=1)

    assert np.max(np.min(np.abs(T - tf), axis=1)) < 1e-3, "Did not find a time bin within the correct tolerance of time t={}".format(tf)

    for n in range(n_trials):
        for i in range(winsize):
            t_idxs[n, idx_tf[n] - i] = True

    X_select = np.empty((winsize, n_behavioral_dims, n_trials), dtype=np.float32)
    X_select[:] = np.nan

    for n in range(n_trials):
        X_select[:,:,n] = X[t_idxs[n], :, n]

    X_select = np.nanmean(X_select, axis=0).T

    if rescale:
        X_select = (X_select - np.nanmean(X_select, axis=0, keepdims=True)) / np.nanstd(X_select, axis=0, keepdims=True)

    return X_select

def select_neural_data_flexible(X, T, tf, winsize):
    '''
    purpose is to handle more general tf (that does not necessarily align)
    along with T \in {trials, time} so a different time vector applies to each trial
    '''
    n_time, n_neurons, n_trials = X.shape

    if len(T.shape) == 1:
        T = np.tile(T, (n_trials, 1))

    t_idxs = np.full(T.shape, False)
    idx_tf = np.argmin(np.abs(T - tf), axis=1)  #compute the closest index for each trial

    assert np.max(np.min(np.abs(T-tf), axis=1)) < 1e-2, ("For some reason the greatest difference in time is {}".format(np.max(np.min(np.abs(T-tf), axis=0))))

    for n in range(n_trials):
        for i in range(winsize):
            t_idxs[n, idx_tf[n] - i] = True

    OUT = np.empty((winsize, n_neurons, n_trials), dtype=np.float32)

    for n in range(n_trials):
        OUT[:, :, n] = X[t_idxs[n], :, n]

    return OUT


# ====================================================
# New Helper Function: Compute Windows with Two Separate Thresholds
# ====================================================
def compute_windows_new(T_behavior, early_avg_grip_ap, late_avg_grip_ap, 
                        window_length=100, threshold_pct1=0.75, threshold_pct2=0.5):
    """
    Compute two windows using new definitions.
    
    Window 1 (Early Trials – Drop/Recovery):
      - Start: Set to the index corresponding to time 0.
      - End: Starting from that index, over the next 'window_length' indices, the early_avg_grip_ap 
             signal is normalized (subtracting the value at time 0) and the end is defined as the 
             first index where the normalized signal reaches threshold_pct1 of its maximum.
    
    Window 2 (Later Trials – Original):
      - Start: Exactly one index after Window 1 ends.
      - End: A candidate end is first determined as the last index within a fixed search range 
             where the late_avg_grip_ap (normalized relative to its start) is above threshold_pct2 
             times its maximum in that range. In addition, Window 2 end must be at least 30 time 
             steps after its start. Then, starting from this candidate, we search forward for the 
             first point after the candidate where the grip aperture value is within a small 
             tolerance of the candidate's value.
    
    Parameters:
      T_behavior         : array-like, time vector.
      early_avg_grip_ap  : array-like, average early trial grip aperture.
      late_avg_grip_ap   : array-like, average late trial grip aperture.
      window_length      : int, number of indices to search for the Window 1 end (default 100).
      threshold_pct1     : float, proportion (e.g., 0.75) for Window 1 end (default 0.75).
      threshold_pct2     : float, proportion (e.g., 0.5) for Window 2 candidate end (default 0.5).
      
    Returns:
      t1_start, t1_end, t2_start, t2_end : floats, time values corresponding to the boundaries.
      indices : dict with keys 'window1_start_idx', 'window1_end_idx', 'window2_start_idx', 'window2_end_idx'.
      lowest_grip_ap_value : float, the value of late_avg_grip_ap at Window 2 end.
    """
    # --- Window 1 Start: set to the index corresponding to time 0 ---
    window1_start_idx = np.argmin(np.abs(np.array(T_behavior)))
    t1_start = T_behavior[window1_start_idx]
    
    # --- Window 1 End ---
    end_window = min(window1_start_idx + window_length, len(early_avg_grip_ap))
    early_segment = early_avg_grip_ap[window1_start_idx:end_window]
    # Normalize relative to the value at time 0.
    early_norm = early_segment - early_avg_grip_ap[window1_start_idx]
    max_norm = np.max(early_norm)
    threshold_val1 = threshold_pct1 * max_norm
    rel_idx = np.where(early_norm >= threshold_val1)[0]
    if rel_idx.size > 0:
        window1_end_idx = window1_start_idx + rel_idx[0]
    else:
        window1_end_idx = end_window - 1
    t1_end = T_behavior[window1_end_idx]
    
    # --- Window 2 Start: exactly one index after Window 1 ends ---
    window2_start_idx = window1_end_idx + 1
    if window2_start_idx >= len(T_behavior):
        window2_start_idx = len(T_behavior) - 1
    t2_start = T_behavior[window2_start_idx]
    
    # --- Window 2 End ---
    # Restrict the search to indices from window2_start up to index 500 (or end if 500 exceeds length)

    window2_end_idx = 449
    t2_end = T_behavior[window2_end_idx]
    lowest_grip_ap_value = late_avg_grip_ap[window2_end_idx]
    
    indices = {
        'window1_start_idx': window1_start_idx,
        'window1_end_idx': window1_end_idx,
        'window2_start_idx': window2_start_idx,
        'window2_end_idx': window2_end_idx
    }
    
    return t1_start, t1_end, t2_start, t2_end, indices, lowest_grip_ap_value




def format_session_key(session_key):
    return str(float(session_key))


# ====================================================
def compute_windows_new(T_behavior, early_avg_grip_ap, late_avg_grip_ap, 
                        window_length=100, threshold_pct1=0.75, threshold_pct2=0.5):
    """
    Compute two windows using new definitions.
    
    Window 1 (Early Trials – Drop/Recovery):
      - Start: Set to the index corresponding to time 0.
      - End: Starting from that index, over the next 'window_length' indices, the early_avg_grip_ap 
             signal is normalized (subtracting the value at time 0) and the end is defined as the 
             first index where the normalized signal reaches threshold_pct1 of its maximum.
    
    Window 2 (Later Trials – Original):
      - Start: Exactly one index after Window 1 ends.
      - End: A candidate end is first determined as the last index within a fixed search range 
             where the late_avg_grip_ap (normalized relative to its start) is above threshold_pct2 
             times its maximum in that range. In addition, Window 2 end must be at least 30 time 
             steps after its start. Then, starting from this candidate, we search forward for the 
             first point after the candidate where the grip aperture value is within a small 
             tolerance of the candidate's value.
    
    Parameters:
      T_behavior         : array-like, time vector.
      early_avg_grip_ap  : array-like, average early trial grip aperture.
      late_avg_grip_ap   : array-like, average late trial grip aperture.
      window_length      : int, number of indices to search for the Window 1 end (default 100).
      threshold_pct1     : float, proportion (e.g., 0.75) for Window 1 end (default 0.75).
      threshold_pct2     : float, proportion (e.g., 0.5) for Window 2 candidate end (default 0.5).
      
    Returns:
      t1_start, t1_end, t2_start, t2_end : floats, time values corresponding to the boundaries.
      indices : dict with keys 'window1_start_idx', 'window1_end_idx', 'window2_start_idx', 'window2_end_idx'.
      lowest_grip_ap_value : float, the value of late_avg_grip_ap at Window 2 end.
    """
    # --- Window 1 Start: set to the index corresponding to time 0 ---
    window1_start_idx = np.argmin(np.abs(np.array(T_behavior)))
    t1_start = T_behavior[window1_start_idx]
    
    # --- Window 1 End ---
    end_window = min(window1_start_idx + window_length, len(early_avg_grip_ap))
    early_segment = early_avg_grip_ap[window1_start_idx:end_window]
    # Normalize relative to the value at time 0.
    early_norm = early_segment - early_avg_grip_ap[window1_start_idx]
    max_norm = np.max(early_norm)
    threshold_val1 = threshold_pct1 * max_norm
    rel_idx = np.where(early_norm >= threshold_val1)[0]
    if rel_idx.size > 0:
        window1_end_idx = window1_start_idx + rel_idx[0]
    else:
        window1_end_idx = end_window - 1
    t1_end = T_behavior[window1_end_idx]
    
    # --- Window 2 Start: exactly one index after Window 1 ends ---
    window2_start_idx = window1_end_idx + 1
    if window2_start_idx >= len(T_behavior):
        window2_start_idx = len(T_behavior) - 1
    t2_start = T_behavior[window2_start_idx]
    
    # --- Window 2 End ---
    # Restrict the search to indices from window2_start up to index 500 (or end if 500 exceeds length)

    window2_end_idx = 449
    t2_end = T_behavior[window2_end_idx]
    lowest_grip_ap_value = late_avg_grip_ap[window2_end_idx]
    
    indices = {
        'window1_start_idx': window1_start_idx,
        'window1_end_idx': window1_end_idx,
        'window2_start_idx': window2_start_idx,
        'window2_end_idx': window2_end_idx
    }
    
    return t1_start, t1_end, t2_start, t2_end, indices, lowest_grip_ap_value



def generate_points(X, T, win1_start, win2_start, winsize):
    """
    Generate sets of points (samples) from two specified time windows.
    
    Instead of automatically creating windows before and after a central time,
    this function uses two explicit windows:
      - Window 1: from win1_start to win1_start + winsize (assigned label -1)
      - Window 2: from win2_start to win2_start + winsize (assigned label +1)
      
    For each window, the data is extracted (using the time vector T), a Gaussian kernel is applied,
    and then the resulting points (one per trial per window) are returned along with their labels.
    
    Parameters:
      X         : array, shape (time, neurons, trials)
      T         : 1D time vector corresponding to X
      win1_start: float, start time of the first window
      win2_start: float, start time of the second window
      winsize   : float, duration of each window
      
    Returns:
      data   : (samples, dimensions) array where samples = (n_trials in window1 + n_trials in window2)
      labels : (samples,) array, with label -1 for the first window and +1 for the second window.
    """
    k_gauss = create_gauss_kernel(winsize)
    
    # First window: select time indices in [win1_start, win1_start + winsize)
    t_idx_win1 = (T >= np.round(win1_start,3)) & (T < np.round(win1_start + winsize,3))
    # Rearrange so that the trial dimension comes first.
    A1 = np.transpose(X[t_idx_win1], (2, 0, 1))  # shape: (trials, window_length, neurons)
    B1 = k_gauss @ A1  # Apply the kernel; adjust operator "@" as needed for your kernel
    points_win1 = list(B1)  # Convert to list (each element is one trial's point)
    
    # Second window: select time indices in [win2_start, win2_start + winsize)
    t_idx_win2 = (T >= np.round(win2_start,3)) & (T < np.round(win2_start + winsize,3))
    A2 = np.transpose(X[t_idx_win2], (2, 0, 1))
    B2 = k_gauss @ A2
    points_win2 = list(B2)
    
    # Create labels: here -1 for points from the first window, +1 for the second.
    labels = [-1] * len(points_win1) + [1] * len(points_win2)
    data = np.vstack([points_win1, points_win2])
    labels = np.array(labels)
    
    return data, labels



def get_trigger_projections(neural_data, T_neural, win1_start, win2_start, winsize):
    """
    Fit cross-validated SVMs to neural data using points generated from two specified windows.
    
    Rather than using windows symmetrically distributed about a single center,
    this version extracts data from two separate windows:
      - Window 1: from win1_start to win1_start+winsize (labeled as -1)
      - Window 2: from win2_start to win2_start+winsize (labeled as +1)
    
    Parameters:
      neural_data : array, shape (time, neurons, trials)
      T_neural    : 1D time vector for neural data
      win1_start  : float, start time for the first window
      win2_start  : float, start time for the second window
      winsize     : float, duration (in time units) of each window
      
    Returns:
      trigger_dim_projection : (time, trials) array, the projection of the neural data 
                               onto the SVM-defined trigger dimension (computed over T_neural)
      T_proj                 : the time vector corresponding to the projection (here, simply T_neural)
    """
    # Generate points from the two specified windows.
    points, labels = generate_points(neural_data, T_neural, win1_start, win2_start, winsize)
    
    # Scale the points and remove neurons with zero variance.
    points, good_neurons, sample_std = scale_neural_data(points)
    
    # Set up cross-validated SVM training.
    from sklearn.svm import SVC
    from sklearn.model_selection import KFold
    model = SVC(kernel="linear")
    repetitions = 20
    kf = KFold(n_splits=3, shuffle=False)
    coeffs_total = []
    scores_total = []
    intercepts_total = []
    
    for i in range(repetitions):
        perm = np.random.permutation(labels.shape[0])
        scores, coeffs, intercepts = cross_validate_trigger(model, points, labels, kf, perm)
        scores_total.append(scores)
        coeffs_total.append(coeffs)
        intercepts_total.append(intercepts)
    
    scores_total = np.array(scores_total)
    coeffs_total = np.array(coeffs_total).mean(0).squeeze()
    intercepts_total = np.array(intercepts_total).mean(0)
    
    # Compute the trigger-dimension projection for each time point.
    trigger_dim_projection = []
    T_proj = T_neural
    k_gauss = create_gauss_kernel(winsize)
    
    for t in T_proj:
        # Here we extract a window of neural data from t-winsize to t.
        t_idx = (T_neural >= t - winsize) & (T_neural < t)
        t_start_idx = np.argmin(np.abs(T_neural - (t - winsize)))
        num_steps = np.round(winsize / 0.01).astype(int)
        idx_range = [t_start_idx + i for i in range(num_steps)]
        
        A = np.transpose(neural_data[idx_range], (2, 0, 1))  # shape: (trials, window_steps, neurons)
        B = k_gauss @ A  # Apply the Gaussian kernel (make sure the dimensions agree)
        C = B[:, good_neurons] / sample_std  # Scale the data
        D = C @ coeffs_total.T + intercepts_total  
        trigger_dim_projection.append(D)
    
    trigger_dim_projection = np.squeeze(np.array(trigger_dim_projection))  # (time, trials)
    return trigger_dim_projection, T_proj



import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon



def one_sample_test(metrics, window_label):
    """
    Cleans NaN values from the metric list and applies the one-sample
    Wilcoxon signed-rank test (comparing to 0) for the given window.
    Reports the two-sided p-value as well as the one-tailed p-value based on the sign of the median.
    """
    metrics = np.array(metrics)
    metrics_clean = metrics[~np.isnan(metrics)]
    
    if len(metrics_clean) == 0:
        print(f"No valid data in {window_label} for statistical testing.")
        return
    
    # Two-sided test (H0: median == 0)
    stat_two, p_two = wilcoxon(metrics_clean, alternative='two-sided')
    median_val = np.median(metrics_clean)
    
    # Determine one-tailed alternative based on the median value
    if median_val > 0:
        alternative = 'greater'
        _, p_one = wilcoxon(metrics_clean, alternative='greater')
    elif median_val < 0:
        alternative = 'less'
        _, p_one = wilcoxon(metrics_clean, alternative='less')
    else:
        alternative = None
        p_one = np.nan

    print(f"\nStatistical test results for {window_label}:")
    print(f"  Median = {median_val:.4f}")
    print(f"  Two-sided test: statistic = {stat_two:.4f}, p-value = {p_two:.4f}")
    if alternative is not None:
        print(f"  One-tailed test (alternative='{alternative}'): p-value = {p_one:.4f}")
    else:
        print("  The median is exactly zero; one-tailed test is not applicable.")


def plot_all_metrics_boxplots_4_mpl(sessions_group1, win_duration=0.05):
    """
    Computes weighted lag metrics for two windows and creates a boxplot.
    """
    window1_metrics = []
    window2_metrics = []
    
    for sess in sessions_group1:
        # Window 1 metric (regular)
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])

        print('\n--- ANALYSIS OF WINDOW CONDITIONS ---')
        print(f'Window duration: {win_duration}')
        print(f'Looking for lags between {-win_duration} and {win_duration}')
        
        # Show specific time limits first to verify the boundary is correct
        lower_limit = -win_duration
        upper_limit = win_duration
        print(f'Lower limit: {lower_limit}, Upper limit: {upper_limit}')
        
        # Check which lag values are within the window
        print('\nChecking each lag value:')
        for i, lag in enumerate(lags_reg):
            is_in_window = (lag >= lower_limit) and (lag <= upper_limit)
            print(f'Index {i}: lag = {lag}, within window? {is_in_window}')
            if i >= 10:  # Only show first 10 values to avoid excessive output
                print("... (remaining values omitted)")
                break
        
        # Create mask with correct window boundaries
        mask_win = (lags_reg >= lower_limit) & (lags_reg <= upper_limit)
        mask_pos = xcorr_reg > 0
        mask_reg = mask_win & mask_pos
        
        # Count and show matching values
        window_matches = np.where(mask_win)[0]
        print(f'\nIndices within window ({lower_limit} to {upper_limit}): {window_matches}')
        
        # For each index, show the actual lag value to confirm
        if len(window_matches) > 0:
            print('\nLag values within window:')
            for idx in window_matches:
                print(f'Index {idx}: lag = {lags_reg[idx]}')
        
        if np.any(mask_reg):
            C_reg = np.sum(xcorr_reg[mask_reg])
            weighted_lag_reg = np.sum(xcorr_reg[mask_reg] * lags_reg[mask_reg]) / C_reg
            window1_metrics.append(weighted_lag_reg)
        else:
            window1_metrics.append(np.nan)
        
        # Window 2 metric (new) - using same careful boundary check
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= lower_limit) & (lags_new <= upper_limit) & (xcorr_new > 0)
        if np.any(mask_new):
            C_new = np.sum(xcorr_new[mask_new])
            weighted_lag_new = np.sum(xcorr_new[mask_new] * lags_new[mask_new]) / C_new
            window2_metrics.append(weighted_lag_new)
        else:
            window2_metrics.append(np.nan)
    
    # Create a boxplot for visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([window2_metrics, window1_metrics], 
                    labels=['Window 1', 'Window 2'],
                    patch_artist=True)
    
    # Set box colors
    colors = ['#C106C0', '#0E2841']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Plot individual jittered points
    for i, data in enumerate([window2_metrics, window1_metrics], 1):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.plot(x, data, 'k.', alpha=0.3)
    
    ax.set_title(f'Weighted Lag Metrics (Total sessions: {len(sessions_group1)})')
    ax.grid(True, alpha=0.3)
    plt.show()

    # Perform one-sample tests for each window independently
    one_sample_test(window1_metrics, "Window 1")
    one_sample_test(window2_metrics, "Window 2")




import matplotlib.pyplot as plt

def plot_sessions(sessions_data,n_sessions):
    """
    Plots session data for multiple sessions.

    Parameters:
    - sessions_data: list or dict-like container where each element is a dictionary containing
      the following keys:
          "T_proj", "T_behavior", "C", "early_avg_trig", "late_avg_trig",
          "early_avg_grip_ap", "late_avg_grip_ap", "new_t_start", "new_t_end",
          "t_start", "t_end", "xcorr_trig", "lags_trig", "xcorr_grip_ap",
          "lags_grip_ap", "new_xcorr_trig", "new_lags_trig", "new_xcorr_grip",
          "new_lags_grip", and "session_id".

    The function creates a 7x6 grid of subplots, with each column representing one session.
    """
    # Create a figure with 7 rows and 6 columns
    fig, ax = plt.subplots(7, n_sessions, figsize=(n_sessions*3, 20))
    
    # Assuming there are 6 sessions; adjust range if needed.
    for sess in range(n_sessions):
        # Retrieve dictionary for this session
        sess_data = sessions_data[sess]
        
        # Extract needed variables
        T_proj           = sess_data["T_proj"]
        T_behavior       = sess_data["T_behavior"]
        C                = sess_data["C"] 
        early_avg_trig   = sess_data["early_avg_trig"]
        late_avg_trig    = sess_data["late_avg_trig"]
        early_avg_grip   = sess_data["early_avg_grip_ap"]
        late_avg_grip    = sess_data["late_avg_grip_ap"]
        new_t_start      = sess_data["new_t_start"]
        new_t_end        = sess_data["new_t_end"]
        t_start          = sess_data["t_start"]
        t_end            = sess_data["t_end"]
        xcorr_trig       = sess_data["xcorr_trig"]
        lags_trig        = sess_data["lags_trig"]
        xcorr_grip_ap    = sess_data["xcorr_grip_ap"]
        lags_grip_ap     = sess_data["lags_grip_ap"]
        new_xcorr_trig   = sess_data["new_xcorr_trig"]
        new_lags_trig    = sess_data["new_lags_trig"]
        new_xcorr_grip   = sess_data["new_xcorr_grip"]
        new_lags_grip    = sess_data["new_lags_grip"]
        
        # -------------------------
        # Plotting
        # -------------------------
        # Row 0: Trigger Projection Time Series
        ax[0, sess].plot(T_proj, early_avg_trig, 'b', label="Early")
        ax[0, sess].plot(T_proj, late_avg_trig, 'r', label="Late")
        ax[0, sess].axvline(new_t_start, color='m', linestyle='--')
        ax[0, sess].axvline(new_t_end, color='m', linestyle='--')
        ax[0, sess].axvline(t_start, color='k', linestyle='--')
        ax[0, sess].axvline(t_end, color='k', linestyle='--')
        ax[0, sess].set_xlim(-0.6, 0.6)
        ax[0, sess].set_title(f"Session {sess_data['session_id']}")
    
        # Row 1: Grip Aperture Time Series
        ax[1, sess].plot(T_behavior, early_avg_grip, 'b', label="Early")
        ax[1, sess].plot(T_behavior, late_avg_grip, 'r', label="Late")
        ax[1, sess].axvline(new_t_start, color='m', linestyle='--')
        ax[1, sess].axvline(new_t_end, color='m', linestyle='--')
        ax[1, sess].axvline(t_start, color='k', linestyle='--')
        ax[1, sess].axvline(t_end, color='k', linestyle='--')
        ax[1, sess].set_xlim(-0.6, 0.6)
    
        # Row 2: Mean Trigger Projection (across ROI / trials)
        ax[2, sess].plot(T_proj, C.mean(axis=1), 'k')
        ax[2, sess].axvline(new_t_start, color='m', linestyle='--')
        ax[2, sess].axvline(new_t_end, color='m', linestyle='--')
        ax[2, sess].axvline(t_start, color='k', linestyle='--')
        ax[2, sess].axvline(t_end, color='k', linestyle='--')
        ax[2, sess].set_xlim(-0.5, 0.5)
    
        # Row 3: Trigger Cross‐Correlation for Window 1 (drop/recovery)
        ax[3, sess].plot(new_lags_trig, new_xcorr_trig, 'm')
        ax[3, sess].axvline(0, color='k', linestyle='--')
        ax[3, sess].set_title("Trigger Xcorr W1")
    
        # Row 4: Trigger Cross‐Correlation for Window 2 (original)
        ax[4, sess].plot(lags_trig, xcorr_trig, 'g')
        ax[4, sess].axvline(0, color='k', linestyle='--')
        ax[4, sess].set_title("Trigger Xcorr W2")
    
        # Row 5: Grip Cross‐Correlation for Window 1 (drop/recovery)
        ax[5, sess].plot(new_lags_grip, new_xcorr_grip, 'm')
        ax[5, sess].axvline(0, color='k', linestyle='--')
        ax[5, sess].set_title("Grip Xcorr W1")
    
        # Row 6: Grip Cross‐Correlation for Window 2 (original)
        ax[6, sess].plot(lags_grip_ap, xcorr_grip_ap, 'g')
        ax[6, sess].axvline(0, color='k', linestyle='--')
        ax[6, sess].set_title("Grip Xcorr W2")
    
    # Add y-axis labels for the first column only
    ax[0, 0].set_ylabel("Trigger Projection")
    ax[1, 0].set_ylabel("Grip Aperture")
    ax[2, 0].set_ylabel("Mean Trigger")
    ax[3, 0].set_ylabel("Trigger Xcorr W1")
    ax[4, 0].set_ylabel("Trigger Xcorr W2")
    ax[5, 0].set_ylabel("Grip Xcorr W1")
    ax[6, 0].set_ylabel("Grip Xcorr W2")
    
    fig.tight_layout()
    plt.show()



import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_all_metrics_boxplots(sessions_group1, sessions_group2, win_duration=0.05):
    """
    Computes metrics for each session in two groups and plots three boxplot cells:
      - Column 1: Metrics for all sessions in Mouse #1 (Group 1)
      - Column 2: Metrics for all sessions in Mouse #2 (Group 2)
      - Column 3: Metrics for the combined data (Group 1 + Group 2)
      
    For each session, metrics are computed using weighted lag approach:
      - Window 1 (regular):
           Weighted lag = Sum(xcorr_trig * lags_trig) / Sum(xcorr_trig)
           where lags are within [-win_duration, win_duration] and xcorr > 0
           
      - Window 2 (new): computed similarly using "new_lags_trig" and "new_xcorr_trig".
      
    The resulting figure uses the same y-axis scale for all cells, with both axes drawn in black.
    
    Parameters:
      sessions_group1 : list
          List of session dictionaries for Mouse #1.
      sessions_group2 : list
          List of session dictionaries for Mouse #2.
      win_duration    : float, duration of the integration window (default 0.05).
    """
    # --- Compute metrics for Group 1 (Mouse #1) ---
    window1_metrics_g1 = []
    window2_metrics_g1 = []
    for sess in sessions_group1:
        # Window 1 metric (regular)
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])
        
        # Create mask with window boundaries
        mask_win = (lags_reg >= -win_duration) & (lags_reg <= win_duration)
        mask_pos = xcorr_reg > 0
        mask_reg = mask_win & mask_pos
        
        if np.any(mask_reg):
            C_reg = np.sum(xcorr_reg[mask_reg])
            weighted_lag_reg = np.sum(xcorr_reg[mask_reg] * lags_reg[mask_reg]) / C_reg
            window1_metrics_g1.append(weighted_lag_reg)
        else:
            window1_metrics_g1.append(np.nan)
        
        # Window 2 metric (new)
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= -win_duration) & (lags_new <= win_duration) & (xcorr_new > 0)
        
        if np.any(mask_new):
            C_new = np.sum(xcorr_new[mask_new])
            weighted_lag_new = np.sum(xcorr_new[mask_new] * lags_new[mask_new]) / C_new
            window2_metrics_g1.append(weighted_lag_new)
        else:
            window2_metrics_g1.append(np.nan)
        
    # --- Compute metrics for Group 2 (Mouse #2) ---
    window1_metrics_g2 = []
    window2_metrics_g2 = []
    for sess in sessions_group2:
        # Window 1 metric (regular)
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])
        
        mask_win = (lags_reg >= -win_duration) & (lags_reg <= win_duration)
        mask_pos = xcorr_reg > 0
        mask_reg = mask_win & mask_pos
        
        if np.any(mask_reg):
            C_reg = np.sum(xcorr_reg[mask_reg])
            weighted_lag_reg = np.sum(xcorr_reg[mask_reg] * lags_reg[mask_reg]) / C_reg
            window1_metrics_g2.append(weighted_lag_reg)
        else:
            window1_metrics_g2.append(np.nan)
        
        # Window 2 metric (new)
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= -win_duration) & (lags_new <= win_duration) & (xcorr_new > 0)
        
        if np.any(mask_new):
            C_new = np.sum(xcorr_new[mask_new])
            weighted_lag_new = np.sum(xcorr_new[mask_new] * lags_new[mask_new]) / C_new
            window2_metrics_g2.append(weighted_lag_new)
        else:
            window2_metrics_g2.append(np.nan)
    
    # --- Merge metrics from both groups ---
    merged_window1_metrics = window1_metrics_g1 + window1_metrics_g2
    merged_window2_metrics = window2_metrics_g1 + window2_metrics_g2
    
    # --- Compute common y-axis range with extra vertical padding ---
    all_metrics = window1_metrics_g1 + window1_metrics_g2 + window2_metrics_g1 + window2_metrics_g2
    all_metrics_no_nan = [m for m in all_metrics if not np.isnan(m)]
    if all_metrics_no_nan:
        global_ymin = min(all_metrics_no_nan)
        global_ymax = max(all_metrics_no_nan)
        padding = (global_ymax - global_ymin) * 0.1  # 10% vertical padding
        y_range = [global_ymin - padding, global_ymax + padding]
    else:
        y_range = [-0.1, 0.1]  # Default range if all values are NaN
    
    # --- Create the figure: 1 row, 3 columns ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Mouse #1", "Mouse #2", "Combined Data"],
        horizontal_spacing=0.1  # Increase horizontal spacing between subplots
    )
    
    # Define colors
    color_w1 = "lightgreen"  # Window 1
    color_w2 = "skyblue"     # Window 2
    
    # --- Cell 1: Mouse #1 ---
    # Switch order: Window 2 first, then Window 1
    fig.add_trace(
        go.Box(
            x=["Window 2"] * len(window2_metrics_g1),
            y=window2_metrics_g1,
            marker_color=color_w2,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(
            x=["Window 1"] * len(window1_metrics_g1),
            y=window1_metrics_g1,
            marker_color=color_w1,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # --- Cell 2: Mouse #2 ---
    # Switch order: Window 2 first, then Window 1
    fig.add_trace(
        go.Box(
            x=["Window 2"] * len(window2_metrics_g2),
            y=window2_metrics_g2,
            marker_color=color_w2,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(
            x=["Window 1"] * len(window1_metrics_g2),
            y=window1_metrics_g2,
            marker_color=color_w1,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # --- Cell 3: Combined Data ---
    # Switch order: Window 2 first, then Window 1
    fig.add_trace(
        go.Box(
            x=["Window 2"] * len(merged_window2_metrics),
            y=merged_window2_metrics,
            marker_color=color_w2,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Box(
            x=["Window 1"] * len(merged_window1_metrics),
            y=merged_window1_metrics,
            marker_color=color_w1,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Set all x-axes to categorical and adjust axes style
    for col in [1, 2, 3]:
        fig.update_xaxes(
            type="category", 
            row=1, col=col,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            categoryorder='array',  # Ensure consistent order
            categoryarray=['Window 2', 'Window 1']  # Set the order
        )
        fig.update_yaxes(
            range=y_range,
            row=1, col=col,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            title="Weighted Lag"
        )
    
    # Fix y-axis overlap by adjusting margins and layout
    fig.update_layout(
        height=400,
        width=1200,
        title_text=f"Weighted Lag Metrics (Window Duration: {win_duration}s)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        showlegend=False,
        margin=dict(l=80, r=80, t=80, b=50),  # Increase left, right, top margins
        yaxis=dict(domain=[0, 0.95]),          # Adjust domain for first y-axis
        yaxis2=dict(domain=[0, 0.95]),         # Adjust domain for second y-axis 
        yaxis3=dict(domain=[0, 0.95])          # Adjust domain for third y-axis
    )
    
    # Add a horizontal line at y=0 for reference
    for col in [1, 2, 3]:
        fig.add_shape(
            type="line", 
            x0=-0.5, 
            x1=1.5, 
            y0=0, 
            y1=0,
            line=dict(color="black", width=1, dash="dot"),
            row=1, 
            col=col
        )
    
    fig.show()

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_all_metrics_boxplots(sessions_group1, sessions_group2, win_duration=0.05):
    """
    Computes metrics for each session in two groups and plots three boxplot cells:
      - Column 1: Metrics for all sessions in Mouse #1 (Group 1)
      - Column 2: Metrics for all sessions in Mouse #2 (Group 2)
      - Column 3: Metrics for the combined data (Group 1 + Group 2)
      
    For each session, metrics are computed using weighted lag approach:
      - Window 1 (regular):
           Weighted lag = Sum(xcorr_trig * lags_trig) / Sum(xcorr_trig)
           where lags are within [-win_duration, win_duration] and xcorr > 0
           
      - Window 2 (new): computed similarly using "new_lags_trig" and "new_xcorr_trig".
      
    The resulting figure uses the same y-axis scale for all cells, with both axes drawn in black.
    
    Parameters:
      sessions_group1 : list
          List of session dictionaries for Mouse #1.
      sessions_group2 : list
          List of session dictionaries for Mouse #2.
      win_duration    : float, duration of the integration window (default 0.05).
    """
    # --- Compute metrics for Group 1 (Mouse #1) ---
    window1_metrics_g1 = []
    window2_metrics_g1 = []
    for sess in sessions_group1:
        # Window 1 metric (regular)
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])
        
        # Create mask with window boundaries
        mask_win = (lags_reg >= -win_duration) & (lags_reg <= win_duration)
        mask_pos = xcorr_reg > 0
        mask_reg = mask_win & mask_pos
        
        if np.any(mask_reg):
            C_reg = np.sum(xcorr_reg[mask_reg])
            weighted_lag_reg = np.sum(xcorr_reg[mask_reg] * lags_reg[mask_reg]) / C_reg
            window1_metrics_g1.append(weighted_lag_reg)
        else:
            window1_metrics_g1.append(np.nan)
        
        # Window 2 metric (new)
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= -win_duration) & (lags_new <= win_duration) & (xcorr_new > 0)
        
        if np.any(mask_new):
            C_new = np.sum(xcorr_new[mask_new])
            weighted_lag_new = np.sum(xcorr_new[mask_new] * lags_new[mask_new]) / C_new
            window2_metrics_g1.append(weighted_lag_new)
        else:
            window2_metrics_g1.append(np.nan)
        
    # --- Compute metrics for Group 2 (Mouse #2) ---
    window1_metrics_g2 = []
    window2_metrics_g2 = []
    for sess in sessions_group2:
        # Window 1 metric (regular)
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])
        
        mask_win = (lags_reg >= -win_duration) & (lags_reg <= win_duration)
        mask_pos = xcorr_reg > 0
        mask_reg = mask_win & mask_pos
        
        if np.any(mask_reg):
            C_reg = np.sum(xcorr_reg[mask_reg])
            weighted_lag_reg = np.sum(xcorr_reg[mask_reg] * lags_reg[mask_reg]) / C_reg
            window1_metrics_g2.append(weighted_lag_reg)
        else:
            window1_metrics_g2.append(np.nan)
        
        # Window 2 metric (new)
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= -win_duration) & (lags_new <= win_duration) & (xcorr_new > 0)
        
        if np.any(mask_new):
            C_new = np.sum(xcorr_new[mask_new])
            weighted_lag_new = np.sum(xcorr_new[mask_new] * lags_new[mask_new]) / C_new
            window2_metrics_g2.append(weighted_lag_new)
        else:
            window2_metrics_g2.append(np.nan)
    
    # --- Merge metrics from both groups ---
    merged_window1_metrics = window1_metrics_g1 + window1_metrics_g2
    merged_window2_metrics = window2_metrics_g1 + window2_metrics_g2
    
    # --- Compute common y-axis range with extra vertical padding ---
    all_metrics = window1_metrics_g1 + window1_metrics_g2 + window2_metrics_g1 + window2_metrics_g2
    all_metrics_no_nan = [m for m in all_metrics if not np.isnan(m)]
    if all_metrics_no_nan:
        global_ymin = min(all_metrics_no_nan)
        global_ymax = max(all_metrics_no_nan)
        padding = (global_ymax - global_ymin) * 0.1  # 10% vertical padding
        y_range = [global_ymin - padding, global_ymax + padding]
    else:
        y_range = [-0.1, 0.1]  # Default range if all values are NaN
    
    # --- Create the figure: 1 row, 3 columns ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Mouse #1", "Mouse #2", "Combined Data"],
        horizontal_spacing=0.1  # Increase horizontal spacing between subplots
    )
    
    # Define colors
    color_w1 = "lightgreen"  # Window 1
    color_w2 = "skyblue"     # Window 2
    
    # --- Cell 1: Mouse #1 ---
    # Switch order: Window 2 first, then Window 1
    fig.add_trace(
        go.Box(
            x=["Window 2"] * len(window2_metrics_g1),
            y=window2_metrics_g1,
            marker_color=color_w2,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(
            x=["Window 1"] * len(window1_metrics_g1),
            y=window1_metrics_g1,
            marker_color=color_w1,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # --- Cell 2: Mouse #2 ---
    # Switch order: Window 2 first, then Window 1
    fig.add_trace(
        go.Box(
            x=["Window 2"] * len(window2_metrics_g2),
            y=window2_metrics_g2,
            marker_color=color_w2,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(
            x=["Window 1"] * len(window1_metrics_g2),
            y=window1_metrics_g2,
            marker_color=color_w1,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=2
    )
    
    # --- Cell 3: Combined Data ---
    # Switch order: Window 2 first, then Window 1
    fig.add_trace(
        go.Box(
            x=["Window 2"] * len(merged_window2_metrics),
            y=merged_window2_metrics,
            marker_color=color_w2,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Box(
            x=["Window 1"] * len(merged_window1_metrics),
            y=merged_window1_metrics,
            marker_color=color_w1,
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            showlegend=False
        ),
        row=1, col=3
    )
    
    # Set all x-axes to categorical and adjust axes style
    for col in [1, 2, 3]:
        fig.update_xaxes(
            type="category", 
            row=1, col=col,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            categoryorder='array',  # Ensure consistent order
            categoryarray=['Window 2', 'Window 1']  # Set the order
        )
        fig.update_yaxes(
            range=y_range,
            row=1, col=col,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            title="Weighted Lag"
        )
    
    # Fix y-axis overlap by adjusting margins and layout
    fig.update_layout(
        height=800,  # Increased height from 400 to 800
        width=1800,  # Increased width from 1200 to 1800
        title_text=f"Weighted Lag Metrics (Window Duration: {win_duration}s)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=16),  # Increased font size from 12 to 16
        showlegend=False,
        margin=dict(l=100, r=100, t=100, b=70),  # Increased margins
        yaxis=dict(domain=[0, 0.95]),          # Adjust domain for first y-axis
        yaxis2=dict(domain=[0, 0.95]),         # Adjust domain for second y-axis 
        yaxis3=dict(domain=[0, 0.95])          # Adjust domain for third y-axis
    )
    
    # Add a horizontal line at y=0 for reference
    for col in [1, 2, 3]:
        fig.add_shape(
            type="line", 
            x0=-0.5, 
            x1=1.5, 
            y0=0, 
            y1=0,
            line=dict(color="black", width=1, dash="dot"),
            row=1, 
            col=col
        )
    
    # Add higher resolution config when showing the figure
    fig.show(config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'weighted_lag_metrics',
            'height': 800,
            'width': 1800,
            'scale': 2  # Higher resolution (2x)
        }
    })

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_sessions_of_interest_two(session_data1, sessions_of_interest1, 
                                  session_data2, sessions_of_interest2):
    """
    Plots aligned time-series panels and cross-correlation panels for two sets of sessions,
    one from each of two different session datasets.
    
    Each row is labeled (e.g., "Mouse 1 – Session X" or "Mouse 2 – Session Y") using the built-in row_titles.
    The four columns share common titles at the top:
      "Trigger Projection", "Grip Aperture", "Xcorr W1", and "Xcorr W2".
    
    Also plots average Trigger Projection and Grip Aperture (averaged across early and late trials)
    for both datasets at the bottom.
    
    Parameters:
        session_data1 (list or dict-like): Container holding session dictionaries for dataset 1.
        sessions_of_interest1 (list): List of indices (or keys) for sessions of interest in dataset 1.
        
        session_data2 (list or dict-like): Container holding session dictionaries for dataset 2.
        sessions_of_interest2 (list): List of indices (or keys) for sessions of interest in dataset 2.
        
    Each session dictionary is expected to include the following keys:
        "T_proj", "T_behavior", "early_avg_trig", "late_avg_trig",
        "early_avg_grip_ap", "late_avg_grip_ap", "new_t_start", "new_t_end",
        "t_start", "t_end", "new_lags_trig", "new_xcorr_trig",
        "lags_trig", "xcorr_trig", and optionally "session_id" for display.
    """
    # Define line width (same for all lines)
    line_width = 2
    
    # Build row titles.
    row_titles = []
    for sess in sessions_of_interest1:
        session_id = session_data1[sess].get("session_id", sess)
        row_titles.append(f"Mouse 1 – Session {session_id}")
    for sess in sessions_of_interest2:
        session_id = session_data2[sess].get("session_id", sess)
        row_titles.append(f"Mouse 2 – Session {session_id}")
    
    # Add two more rows for averages
    row_titles.append("Mouse 1 – Average")
    row_titles.append("Mouse 2 – Average")
    
    total_sessions = len(row_titles)
    cols = 4
    # Create the figure with column titles and row titles.
    fig = make_subplots(
        rows=total_sessions,
        cols=cols,
        column_titles=["Trigger Projection", "Grip Aperture", "Xcorr W1", "Xcorr W2"],
        row_titles=row_titles
    )
    
    # By default, the row titles are placed on the right.
    # Adjust the annotations created by make_subplots so that row titles appear on the left.
    for annotation in fig.layout.annotations:
        if annotation.text in row_titles:
            annotation.x = -0.1  # Shift further left in paper coordinates.
            annotation.xanchor = 'left'
    
    # Define color constants.
    ts_color_early = "red"
    ts_color_late = "blue"    
    ts_color_avg = "black"  # Color for the averaged data
    xcorr_color_w1 = "black"  # For cross-correlation window 1.
    xcorr_color_w2 = "black"  # For cross-correlation window 2.
    
    # Data for computing averages
    mouse1_avg_trigs = []  # Will hold averaged early+late for each session
    mouse1_avg_grips = []
    mouse1_t_proj = None
    mouse1_t_behavior = None
    
    mouse2_avg_trigs = []
    mouse2_avg_grips = []
    mouse2_t_proj = None
    mouse2_t_behavior = None
    
    # Collect window boundaries for both mice
    mouse1_window_boundaries = {"new_t_start": [], "new_t_end": [], "t_start": [], "t_end": []}
    mouse2_window_boundaries = {"new_t_start": [], "new_t_end": [], "t_start": [], "t_end": []}
    
    # Helper function to add traces for a given session.
    def add_session_traces(sess_data, row):
        T_proj         = sess_data["T_proj"]
        T_behavior     = sess_data["T_behavior"]
        early_avg_trig = sess_data["early_avg_trig"]
        late_avg_trig  = sess_data["late_avg_trig"]
        early_avg_grip = sess_data["early_avg_grip_ap"]
        late_avg_grip  = sess_data["late_avg_grip_ap"]
        new_t_start    = sess_data["new_t_start"]
        new_t_end      = sess_data["new_t_end"]
        t_start        = sess_data["t_start"]
        t_end          = sess_data["t_end"]
        new_lags_trig  = sess_data["new_lags_trig"]
        new_xcorr_trig = sess_data["new_xcorr_trig"]
        lags_trig      = sess_data["lags_trig"]
        xcorr_trig     = sess_data["xcorr_trig"]
        
        # Calculate the average of early and late for this session
        avg_trig = (early_avg_trig + late_avg_trig) / 2
        avg_grip = (early_avg_grip + late_avg_grip) / 2
        
        # Column 1: Trigger Projection.
        fig.add_trace(
            go.Scatter(x=T_proj, y=early_avg_trig, mode='lines',
                      line=dict(color=ts_color_early, width=line_width),
                      showlegend=False),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=T_proj, y=late_avg_trig, mode='lines',
                      line=dict(color=ts_color_late, width=line_width),
                      showlegend=False),
            row=row, col=1
        )
        for v in [new_t_start, new_t_end]:
            fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w1, row=row, col=1)
        for v in [t_start, t_end]:
            fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w2, row=row, col=1)
        fig.update_xaxes(range=[-0.2, 0.6], row=row, col=1)
        
        # Column 2: Grip Aperture.
        fig.add_trace(
            go.Scatter(x=T_behavior, y=early_avg_grip, mode='lines',
                      line=dict(color=ts_color_early, width=line_width),
                      showlegend=False),
            row=row, col=2
        )
        fig.add_trace(
            go.Scatter(x=T_behavior, y=late_avg_grip, mode='lines',
                      line=dict(color=ts_color_late, width=line_width),
                      showlegend=False),
            row=row, col=2
        )
        for v in [new_t_start, new_t_end]:
            fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w1, row=row, col=2)
        for v in [t_start, t_end]:
            fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w2, row=row, col=2)
        fig.update_xaxes(range=[-0.2, 0.6], row=row, col=2)
        
        # Column 3: Trigger Cross-Correlation Window 1.
        fig.add_trace(
            go.Scatter(x=new_lags_trig, y=new_xcorr_trig, mode='lines',
                      line=dict(color=xcorr_color_w1, width=line_width),
                      showlegend=False),
            row=row, col=3
        )
        fig.add_vline(x=0, line_dash="dash", line_color=xcorr_color_w1, row=row, col=3)
        
        # Column 4: Trigger Cross-Correlation Window 2.
        fig.add_trace(
            go.Scatter(x=lags_trig, y=xcorr_trig, mode='lines',
                      line=dict(color=xcorr_color_w2, width=line_width),
                      showlegend=False),
            row=row, col=4
        )
        fig.add_vline(x=0, line_dash="dash", line_color=xcorr_color_w2, row=row, col=4)
        
        return {
            "T_proj": T_proj,
            "T_behavior": T_behavior,
            "avg_trig": avg_trig,  # Return the averaged trigger values
            "avg_grip": avg_grip,  # Return the averaged grip values
            "new_t_start": new_t_start,
            "new_t_end": new_t_end,
            "t_start": t_start,
            "t_end": t_end
        }
    
    # Process sessions from mouse 1
    current_row = 1
    for sess in sessions_of_interest1:
        sess_data = session_data1[sess]
        session_info = add_session_traces(sess_data, current_row)
        
        # Collect data for averaging
        mouse1_avg_trigs.append(session_info["avg_trig"])
        mouse1_avg_grips.append(session_info["avg_grip"])
        mouse1_t_proj = session_info["T_proj"]  # Assuming all sessions have same timepoints
        mouse1_t_behavior = session_info["T_behavior"]
        
        # Collect window boundaries
        mouse1_window_boundaries["new_t_start"].append(session_info["new_t_start"])
        mouse1_window_boundaries["new_t_end"].append(session_info["new_t_end"])
        mouse1_window_boundaries["t_start"].append(session_info["t_start"])
        mouse1_window_boundaries["t_end"].append(session_info["t_end"])
        
        current_row += 1
    
    # Process sessions from mouse 2
    for sess in sessions_of_interest2:
        sess_data = session_data2[sess]
        session_info = add_session_traces(sess_data, current_row)
        
        # Collect data for averaging
        mouse2_avg_trigs.append(session_info["avg_trig"])
        mouse2_avg_grips.append(session_info["avg_grip"])
        mouse2_t_proj = session_info["T_proj"]  # Assuming all sessions have same timepoints
        mouse2_t_behavior = session_info["T_behavior"]
        
        # Collect window boundaries
        mouse2_window_boundaries["new_t_start"].append(session_info["new_t_start"])
        mouse2_window_boundaries["new_t_end"].append(session_info["new_t_end"])
        mouse2_window_boundaries["t_start"].append(session_info["t_start"])
        mouse2_window_boundaries["t_end"].append(session_info["t_end"])
        
        current_row += 1
    
    # Calculate average window boundaries for mouse 1
    avg_mouse1_new_t_start = np.mean(mouse1_window_boundaries["new_t_start"])
    avg_mouse1_new_t_end = np.mean(mouse1_window_boundaries["new_t_end"])
    avg_mouse1_t_start = np.mean(mouse1_window_boundaries["t_start"])
    avg_mouse1_t_end = np.mean(mouse1_window_boundaries["t_end"])
    
    # Calculate average window boundaries for mouse 2
    avg_mouse2_new_t_start = np.mean(mouse2_window_boundaries["new_t_start"])
    avg_mouse2_new_t_end = np.mean(mouse2_window_boundaries["new_t_end"])
    avg_mouse2_t_start = np.mean(mouse2_window_boundaries["t_start"])
    avg_mouse2_t_end = np.mean(mouse2_window_boundaries["t_end"])
    
    # Calculate averages across sessions for mouse 1
    avg_of_avg_mouse1_trig = np.mean(mouse1_avg_trigs, axis=0)
    avg_of_avg_mouse1_grip = np.mean(mouse1_avg_grips, axis=0)
    
    # Calculate averages across sessions for mouse 2
    avg_of_avg_mouse2_trig = np.mean(mouse2_avg_trigs, axis=0)
    avg_of_avg_mouse2_grip = np.mean(mouse2_avg_grips, axis=0)
    
    # Add average traces for mouse 1
    # Trigger Projection - Mouse 1 Average
    fig.add_trace(
        go.Scatter(x=mouse1_t_proj, y=avg_of_avg_mouse1_trig, mode='lines',
                  line=dict(color=ts_color_avg, width=line_width),
                  showlegend=False),
        row=total_sessions-1, col=1
    )
    # Add window markers
    for v in [avg_mouse1_new_t_start, avg_mouse1_new_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w1, row=total_sessions-1, col=1)
    for v in [avg_mouse1_t_start, avg_mouse1_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w2, row=total_sessions-1, col=1)
    fig.update_xaxes(range=[-0.2, 0.6], row=total_sessions-1, col=1)
    
    # Grip Aperture - Mouse 1 Average
    fig.add_trace(
        go.Scatter(x=mouse1_t_behavior, y=avg_of_avg_mouse1_grip, mode='lines',
                  line=dict(color=ts_color_avg, width=line_width),
                  showlegend=False),
        row=total_sessions-1, col=2
    )
    # Add window markers
    for v in [avg_mouse1_new_t_start, avg_mouse1_new_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w1, row=total_sessions-1, col=2)
    for v in [avg_mouse1_t_start, avg_mouse1_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w2, row=total_sessions-1, col=2)
    fig.update_xaxes(range=[-0.2, 0.6], row=total_sessions-1, col=2)
    
    # Add average traces for mouse 2
    # Trigger Projection - Mouse 2 Average
    fig.add_trace(
        go.Scatter(x=mouse2_t_proj, y=avg_of_avg_mouse2_trig, mode='lines',
                  line=dict(color=ts_color_avg, width=line_width),
                  showlegend=False),
        row=total_sessions, col=1
    )
    # Add window markers
    for v in [avg_mouse2_new_t_start, avg_mouse2_new_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w1, row=total_sessions, col=1)
    for v in [avg_mouse2_t_start, avg_mouse2_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w2, row=total_sessions, col=1)
    fig.update_xaxes(range=[-0.2, 0.6], row=total_sessions, col=1)
    
    # Grip Aperture - Mouse 2 Average
    fig.add_trace(
        go.Scatter(x=mouse2_t_behavior, y=avg_of_avg_mouse2_grip, mode='lines',
                  line=dict(color=ts_color_avg, width=line_width),
                  showlegend=False),
        row=total_sessions, col=2
    )
    # Add window markers
    for v in [avg_mouse2_new_t_start, avg_mouse2_new_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w1, row=total_sessions, col=2)
    for v in [avg_mouse2_t_start, avg_mouse2_t_end]:
        fig.add_vline(x=v, line_dash="dash", line_color=xcorr_color_w2, row=total_sessions, col=2)
    fig.update_xaxes(range=[-0.2, 0.6], row=total_sessions, col=2)
    
    # Add a legend for early/late/average
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode='lines', name='Early',
                   line=dict(color=ts_color_early, width=line_width)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode='lines', name='Late',
                   line=dict(color=ts_color_late, width=line_width)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode='lines', name='Average',
                   line=dict(color=ts_color_avg, width=line_width)),
        row=1, col=1
    )
    
    # Update layout with higher resolution settings
    fig.update_layout(
        height=300 * total_sessions,  # Keep proportional height
        width=1800,                   # Increased from 1400
        title_text="Aligned Trials & Trigger Cross-Correlation",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),           # Increased font size from 12
        margin=dict(l=150),           # Maintain left margin for row titles
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.02,
            xanchor="center",
            x=0.5
        ),
        # Add image export settings for higher resolution
        images=[dict(
            source="",  # This is just a placeholder that gets replaced on export
            xref="paper", yref="paper",
            x=0, y=0,
            sizex=1, sizey=1,
            sizing="stretch",
            opacity=1,
            layer="below"
        )]
    )
    
    # Improve axis appearance
    fig.update_xaxes(
        showline=True, 
        linewidth=2, 
        linecolor='black', 
        mirror=True,      # Add mirror effect
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray'
    )
    
    fig.update_yaxes(
        showline=True, 
        linewidth=2, 
        linecolor='black', 
        mirror=True,      # Add mirror effect
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray'
    )
    
    # Show the figure with higher resolution settings
    fig.show(config={
        'toImageButtonOptions': {
            'format': 'png',      # Export format
            'filename': 'aligned_trials_comparison',
            'height': 300 * total_sessions,
            'width': 1800,
            'scale': 3            # Increase scaling for higher resolution
        }
    })



def get_trigger_projections_w(neural_data, T_neural, win1_start, win2_start, winsize):
    """
    Fit cross-validated SVMs to neural data using points generated from two specified windows.
    
    Rather than using windows symmetrically distributed about a single center,
    this version extracts data from two separate windows:
      - Window 1: from win1_start to win1_start+winsize (labeled as -1)
      - Window 2: from win2_start to win2_start+winsize (labeled as +1)
    
    Parameters:
      neural_data : array, shape (time, neurons, trials)
      T_neural    : 1D time vector for neural data
      win1_start  : float, start time for the first window
      win2_start  : float, start time for the second window
      winsize     : float, duration (in time units) of each window
      
    Returns:
      trigger_dim_projection : (time, trials) array, the projection of the neural data 
                               onto the SVM-defined trigger dimension (computed over T_neural)
      T_proj                 : the time vector corresponding to the projection (here, simply T_neural)
    """
    # Generate points from the two specified windows.
    points, labels = generate_points(neural_data, T_neural, win1_start, win2_start, winsize)
    
    # Scale the points and remove neurons with zero variance.
    points, good_neurons, sample_std = scale_neural_data(points)
    
    # Set up cross-validated SVM training.
    from sklearn.svm import SVC
    from sklearn.model_selection import KFold
    model = SVC(kernel="linear")
    repetitions = 20
    kf = KFold(n_splits=3, shuffle=False)
    
    coeffs_total = []
    scores_total = []
    intercepts_total = []
    
    for i in range(repetitions):
        perm = np.random.permutation(labels.shape[0])
        scores, coeffs, intercepts = cross_validate_trigger(model, points, labels, kf, perm)
        scores_total.append(scores)
        coeffs_total.append(coeffs)
        intercepts_total.append(intercepts)
    
    scores_total = np.array(scores_total)
    coeffs_total = np.array(coeffs_total).mean(0).squeeze()
    intercepts_total = np.array(intercepts_total).mean(0)
    
    # Compute the trigger-dimension projection for each time point.
    trigger_dim_projection = []
    T_proj = T_neural
    k_gauss = create_gauss_kernel(winsize)
    
    for t in T_proj:
        # Here we extract a window of neural data from t-winsize to t.
        t_idx = (T_neural >= t - winsize) & (T_neural < t)
        # To be sure we have a contiguous block, we use the index closest to (t-winsize)
        t_start_idx = np.argmin(np.abs(T_neural - (t - winsize)))
        # Assume that winsize/0.01 gives the number of steps (adjust as needed)
        num_steps = np.round(winsize / 0.01).astype(int)
        idx_range = [t_start_idx + i for i in range(num_steps)]
        
        A = np.transpose(neural_data[idx_range], (2, 0, 1))  # shape: (trials, window_steps, neurons)
        B = k_gauss @ A  # Apply the Gaussian kernel (make sure the dimensions agree)
        C = B[:, good_neurons] / sample_std  # Scale the data
        D = C @ coeffs_total.T + intercepts_total  
        trigger_dim_projection.append(D)
    
    trigger_dim_projection = np.squeeze(np.array(trigger_dim_projection))  # (time, trials)
    return trigger_dim_projection, T_proj,coeffs_total


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon

def plot_all_metrics_boxplots(sessions_group1, sessions_group2, win_duration=0.04):
    """
    Computes metrics for each session in two groups and plots three boxplot cells:
      - Column 1: Metrics for all sessions in Mouse #1 (Group 1)
      - Column 2: Metrics for all sessions in Mouse #2 (Group 2)
      - Column 3: Metrics for the combined data (Group 1 + Group 2)
      
    For each session, metrics are computed using weighted lag approach:
      - Window 1 (regular):
           weighted_lag = sum(xcorr_trig * lags_trig) / sum(xcorr_trig)
           over lags in [-win_duration, win_duration] with xcorr_trig > 0
      - Window 2 (new):
           computed similarly using new_xcorr_trig and new_lags_trig.
      
    After computing metrics, runs Wilcoxon signed‑rank tests (two‑sided,
    greater, and less) against zero for each window and each subgroup,
    then plots the boxplots identically to before.
    """
    # --- Compute metrics for Group 1 (Mouse #1) ---
    window1_metrics_g1 = []
    window2_metrics_g1 = []
    for sess in sessions_group1:
        # Window 1
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])
        mask_reg = (lags_reg >= -win_duration) & (lags_reg <= win_duration) & (xcorr_reg > 0)
        if np.any(mask_reg):
            C_reg = xcorr_reg[mask_reg].sum()
            window1_metrics_g1.append((xcorr_reg[mask_reg] * lags_reg[mask_reg]).sum() / C_reg)
        else:
            window1_metrics_g1.append(np.nan)
        # Window 2
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= -win_duration) & (lags_new <= win_duration) & (xcorr_new > 0)
        if np.any(mask_new):
            C_new = xcorr_new[mask_new].sum()
            window2_metrics_g1.append((xcorr_new[mask_new] * lags_new[mask_new]).sum() / C_new)
        else:
            window2_metrics_g1.append(np.nan)

    # --- Compute metrics for Group 2 (Mouse #2) ---
    window1_metrics_g2 = []
    window2_metrics_g2 = []
    for sess in sessions_group2:
        # Window 1
        lags_reg = np.array(sess["lags_trig"])
        xcorr_reg = np.array(sess["xcorr_trig"])
        mask_reg = (lags_reg >= -win_duration) & (lags_reg <= win_duration) & (xcorr_reg > 0)
        if np.any(mask_reg):
            C_reg = xcorr_reg[mask_reg].sum()
            window1_metrics_g2.append((xcorr_reg[mask_reg] * lags_reg[mask_reg]).sum() / C_reg)
        else:
            window1_metrics_g2.append(np.nan)
        # Window 2
        lags_new = np.array(sess["new_lags_trig"])
        xcorr_new = np.array(sess["new_xcorr_trig"])
        mask_new = (lags_new >= -win_duration) & (lags_new <= win_duration) & (xcorr_new > 0)
        if np.any(mask_new):
            C_new = xcorr_new[mask_new].sum()
            window2_metrics_g2.append((xcorr_new[mask_new] * lags_new[mask_new]).sum() / C_new)
        else:
            window2_metrics_g2.append(np.nan)

    # --- Merge across groups ---
    merged_window1 = window1_metrics_g1 + window1_metrics_g2
    merged_window2 = window2_metrics_g1 + window2_metrics_g2

    # --- Wilcoxon signed‑rank testing ---
    def signrank_test(data, alternative):
        arr = np.array(data)
        vals = arr[(~np.isnan(arr)) & (arr != 0)]
        if len(vals) >= 1:
            W, p = wilcoxon(vals, zero_method='wilcox', alternative=alternative)
            return len(vals), W, p
        else:
            return len(vals), np.nan, np.nan

    groups = {
        "Mouse #1": {"Window 2": window1_metrics_g1, "Window 1": window2_metrics_g1},
        "Mouse #2": {"Window 2": window1_metrics_g2, "Window 1": window2_metrics_g2},
        "Combined": {"Window 2": merged_window1,   "Window 1": merged_window2}
    }

    print("Wilcoxon signed‑rank tests (median vs. 0):")
    for w in ["Window 1", "Window 2"]:
        print(f"\n--- {w} ---")
        for grp, data_dict in groups.items():
            data = data_dict[w]
            for alt, label in [("two-sided", "≠ 0"), ("greater", "> 0"), ("less", "< 0")]:
                n, W, p = signrank_test(data, alt)
                if np.isnan(p):
                    print(f"{grp}, H₀ median {label}: insufficient non-zero data (n={n})")
                else:
                    sig = "p<0.05" if p < 0.05 else f"p={p:.3f}"
                    print(f"{grp}, H₀ median {label}: n={n}, W={W:.2f}, {sig}")

    # --- Determine common y-axis range ---
    all_vals = [v for v in (merged_window1 + merged_window2) if not np.isnan(v)]
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        pad = (ymax - ymin) * 0.1
        y_range = [ymin - pad, ymax + pad]
    else:
        y_range = [-0.1, 0.1]

    # --- Plotting boxplots ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Mouse #1", "Mouse #2", "Combined"],
        horizontal_spacing=0.1
    )
    colors = {"Window 1": "lightgreen", "Window 2": "skyblue"}

    for col_idx, (grp, data_dict) in enumerate(groups.items(), start=1):
        # Window 2 first, then Window 1
        for w in ["Window 1", "Window 2"]:
            fig.add_trace(
                go.Box(
                    x=[w] * len(data_dict[w]),
                    y=data_dict[w],
                    marker_color=colors[w],
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=0,
                    showlegend=False
                ),
                row=1, col=col_idx
            )
        fig.update_xaxes(
            type="category",
            row=1, col=col_idx,
            showline=True, linewidth=2, linecolor="black",
            mirror=True,
            categoryorder="array",
            categoryarray=["Window 1", "Window 2"]
        )
        fig.update_yaxes(
            range=y_range,
            title="Weighted Lag",
            showline=True, linewidth=2, linecolor="black",
            mirror=True,
            row=1, col=col_idx
        )
        # zero line
        fig.add_shape(
            type="line", x0=-0.5, x1=1.5, y0=0, y1=0,
            line=dict(color="black", width=1, dash="dot"),
            row=1, col=col_idx
        )

    fig.update_layout(
        height=500, width=1400,
        title_text=f"Weighted Lag Metrics (Window Duration: {win_duration}s)",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=16),
        margin=dict(l=100, r=100, t=100, b=70)
    )
    fig.show(config={
        "toImageButtonOptions": {
            "format": "png", "filename": "weighted_lag_metrics",
            "height": 500, "width": 1400, "scale": 2
        }
    })




import numpy as np
from scipy.stats import kurtosis
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def kurtosis_bootstrap_test(weights,
                             n_bootstrap=10000,
                             fisher=False,
                             random_state=None):
    """
    Performs the kurtosis + bootstrap test from Ames et al.
    Returns observed kurtosis, array of bootstrap kurtoses, and one-sided p-value for higher-than-chance kurtosis.
    """
    rng = np.random.RandomState(random_state)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D array")
    w /= np.linalg.norm(w)
    obs_kurt = kurtosis(w, fisher=fisher)
    dim = w.size
    rand_kurts = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        v = rng.randn(dim)
        v /= np.linalg.norm(v)
        rand_kurts[i] = kurtosis(v, fisher=fisher)
    # p-value: fraction of random kurtoses >= observed (one-sided test for obs > null)
    p_value = np.mean(rand_kurts >= obs_kurt)
    return obs_kurt, rand_kurts, p_value

def plot_svm_weights(session_data, selected_sessions=None, save_path=None,
                     n_bootstrap=10000, random_state=None):
    """
    Plots sorted SVM weight vectors alongside random projections and
    shows kurtosis significance across sessions.
    """
    if selected_sessions is None:
        selected_sessions = list(range(len(session_data)))
    filtered = [session_data[i] for i in selected_sessions]
    orig_idx = selected_sessions
    all_weights = [np.asarray(sess['coeffs_total'], dtype=float) for sess in filtered]
    n_sessions = len(all_weights)

    # Compute normalized weights, kurtosis test per session
    kurt_vals, thresholds, p_vals, norm_weights = [], [], [], []
    for w in all_weights:
        norm = np.linalg.norm(w)
        if norm == 0:
            kurt_vals.append(np.nan)
            thresholds.append(np.nan)
            p_vals.append(np.nan)
            norm_weights.append(w)
            continue
        w_norm = w / norm
        obs, rand_kurts, p = kurtosis_bootstrap_test(w_norm, n_bootstrap,
                                                      fisher=False, random_state=random_state)
        thresh = np.percentile(rand_kurts, 95)
        kurt_vals.append(obs)
        thresholds.append(thresh)
        p_vals.append(p)
        norm_weights.append(w_norm)

    # Build subplot titles including kurtosis and significance marker
    titles = []
    for i in range(n_sessions):
        kv = kurt_vals[i]
        pv = p_vals[i]
        if np.isnan(kv):
            titles.append(f"Session {orig_idx[i]+1}<br>Kurt=NaN")
        else:
            sig = ' *' if pv < 0.05 else ''
            titles.append(f"Session {orig_idx[i]+1}<br>Kurt={kv:.2f}{sig}")

    # Create two-row subplot layout
    specs = [[{'type':'xy'}]*n_sessions,
             [{'colspan': n_sessions, 'type':'xy'}] + [None]*(n_sessions-1)]
    fig = make_subplots(rows=2, cols=n_sessions, specs=specs,
                        subplot_titles=titles + ['Kurtosis Summary'],
                        vertical_spacing=0.15, horizontal_spacing=0.08)

    # Row 1: weight distributions vs. random bounds
    for i, w_norm in enumerate(norm_weights):
        dim = w_norm.size
        if np.linalg.norm(w_norm) == 0:
            continue
        sorted_idx = np.argsort(np.abs(w_norm))[::-1]
        sorted_act = w_norm[sorted_idx]
        # random bounds
        rand_sorted = np.zeros((n_bootstrap, dim))
        rng = np.random.RandomState(random_state)
        for j in range(n_bootstrap):
            v = rng.randn(dim)
            v /= np.linalg.norm(v)
            idx = np.argsort(np.abs(v))[::-1]
            rand_sorted[j] = v[idx]
        avg_rand = rand_sorted.mean(0)
        sd_rand = rand_sorted.std(0)
        upper, lower = avg_rand + sd_rand, avg_rand - sd_rand
        x = np.arange(1, dim+1)
        showlegend = (i == 0)
        fig.add_trace(go.Bar(x=x, y=sorted_act, marker_color='black', opacity=0.7,
                             showlegend=showlegend, name='Actual Weights'), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=x, y=upper, mode='lines', line=dict(color='red'),
                                 showlegend=showlegend, name='Rand ±1SD'), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=x, y=lower, mode='lines', line=dict(color='red'),
                                 showlegend=False), row=1, col=i+1)

    # Row 2: observed kurtosis and 95th percentile threshold
    sess_lbls = [f"S{orig_idx[i]+1}" for i in range(n_sessions)]
    fig.add_trace(go.Bar(x=sess_lbls, y=kurt_vals, name='Observed Kurtosis'), row=2, col=1)
    fig.add_trace(go.Scatter(x=sess_lbls, y=thresholds, mode='lines+markers',
                             name='95th Percentile'), row=2, col=1)
    # annotate significance stars
    for i, pv in enumerate(p_vals):
        if not np.isnan(pv) and pv < 0.05:
            fig.add_annotation(x=sess_lbls[i], y=max(kurt_vals[i], thresholds[i]),
                               text='*', showarrow=False, yshift=10,
                               row=2, col=1)

    # Axes and layout
    for i in range(1, n_sessions+1):
        fig.update_xaxes(title_text='Units', row=1, col=i)
    fig.update_yaxes(title_text='Weight Magnitude', row=1, col=1)
    fig.update_xaxes(title_text='Session', row=2, col=1)
    fig.update_yaxes(title_text='Kurtosis', row=2, col=1)
    fig.update_layout(title='SVM Weights & Kurtosis Significance', template='plotly_white',
                      showlegend=True, width=300*n_sessions, height=700)

    if save_path:
        fig.write_image(save_path)
    return fig



def plot_multi_session_correlations_separate_plotly(sessions_data, session_ids_to_plot, save_path=None):
    """
    For a list of sessions:
    1. Analyzes each session separately
    2. Creates separate figures for each session showing:
       - Correlation between trigger peak times and max grip times
       - Time differences by trial and selection criteria
    
    Args:
        sessions_data: List of session dictionaries containing neural and behavioral data
        session_ids_to_plot: List of session IDs to analyze
        save_path: Optional base path to save the figures
        
    Returns:
        dict: Dictionary with correlation results for all sessions
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import pearsonr
    
    # Custom function to find local maxima in a signal
    def find_local_maxima(signal, min_height=None, min_distance=3):
        # Pad the signal to handle edge cases
        padded = np.pad(signal, 1, mode='edge')
        # Find local maxima by comparing with left and right neighbors
        mask = (padded[1:-1] > padded[0:-2]) & (padded[1:-1] > padded[2:])
        
        # Apply minimum height filter if specified
        if min_height is not None:
            mask = mask & (signal > min_height)
            
        # Get indices of all local maxima
        indices = np.where(mask)[0]
        
        # Apply minimum distance filter
        if len(indices) > 0 and min_distance > 1:
            # Sort peaks by height
            peak_heights = signal[indices]
            sorted_idx = np.argsort(peak_heights)[::-1]  # Sort in descending order
            sorted_indices = indices[sorted_idx]
            
            # Keep track of which peaks to keep
            keep = np.ones(len(sorted_indices), dtype=bool)
            
            # Iterate through sorted peaks
            for i in range(len(sorted_indices)):
                if keep[i]:
                    # Remove peaks that are within min_distance
                    idx = sorted_indices[i]
                    j = i + 1
                    while j < len(sorted_indices):
                        if abs(sorted_indices[j] - idx) < min_distance:
                            keep[j] = False
                        j += 1
            
            # Return indices of kept peaks, sorted by position
            indices = sorted(sorted_indices[keep])
        
        return np.array(indices)
    
    # Store all session results
    all_results = {}
    
    # Define improved color palette (more visually appealing)
    color_map = {
        'single-before-index-2+': '#1f77b4',     # Deeper blue
        'single-before-no-peaks-after': '#9467bd',  # Rich purple
        'latest-before': '#17becf',              # Turquoise
        'second-peak': '#e41a1c'                 # Vivid red
    }
    
    # Analyze each session
    for session_id in session_ids_to_plot:
        print(f"Analyzing session {session_id}...")
        
        # Find the requested session
        session_found = False
        for session in sessions_data:
            if session["session_id"] == session_id:
                session_found = True
                break
        
        if not session_found:
            print(f"Session {session_id} not found in data")
            continue
        
        # Extract relevant data
        T_behavior = session["T_behavior"]
        T_proj = session["T_proj"]
        grip_ap = session["grip_ap"]
        C = session["C"]
        
        # Lists to store time points for each trial
        trial_indices = []
        trial_max_grip_times = []
        trial_trigger_peak_times = []
        
        # Lists to store data for signal plots
        all_trial_data = []
        
        # Process each trial
        for trial_idx in range(grip_ap.shape[0]):
            # Get this trial's grip aperture and trigger projection
            trial_grip_ap = grip_ap[trial_idx]
            trial_trig = C[:, trial_idx]
            
            # 1. Find the time point where grip aperture is maximum (constrained to after 80ms)
            start_idx_behavior = np.where(T_behavior >= 0.080)[0][0]  # Index where time >= 80ms
            max_grip_idx = start_idx_behavior + np.argmax(trial_grip_ap[start_idx_behavior:])
            max_grip_time = T_behavior[max_grip_idx]
            max_grip_value = trial_grip_ap[max_grip_idx]
            
            # 2. Apply smoothing to the trigger signal for better peak detection
            smoothed_trig = gaussian_filter1d(trial_trig, sigma=1)
            
            # 3. Find all peaks in the trigger signal after 0ms
            start_idx_neural = np.where(T_proj >= 0.0)[0][0]  # Index where time >= 0ms
            signal_after_0 = smoothed_trig[start_idx_neural:]
            
            # Find local maxima using custom function
            peak_indices_rel = find_local_maxima(signal_after_0, min_distance=3)
            
            # Convert to indices in the full signal
            all_peak_indices = peak_indices_rel + start_idx_neural
            all_peak_times = T_proj[all_peak_indices]
            
            if len(all_peak_indices) < 2:
                # Need at least 2 peaks for this analysis
                continue
            
            # 4. Define search window - we're looking both before and after max_grip_time
            search_start_time = max_grip_time - 0.100  # For visualization purposes
            
            # 5. Find peaks before max_grip_time
            peaks_before_mask = all_peak_times <= max_grip_time
            peaks_before_indices = all_peak_indices[peaks_before_mask]
            peaks_before_times = all_peak_times[peaks_before_mask]
            
            # 6. Find peaks shortly after max_grip_time (within 80ms)
            time_buffer_after_max = 0.080  # 80ms after max grip
            peaks_after_mask = (all_peak_times > max_grip_time) & (all_peak_times <= max_grip_time + time_buffer_after_max)
            peaks_after_indices = all_peak_indices[peaks_after_mask]
            
            # 7. Initialize variables to track peak selection
            selected_peak_idx = None
            selected_peak_time = None
            peak_selection = ""
            
            if len(peaks_before_indices) > 1:
                # Multiple peaks before max_grip_time, use the latest one
                selected_peak_idx = peaks_before_indices[-1]
                selected_peak_time = T_proj[selected_peak_idx]
                # Get the 0-based index of this peak in the full list of peaks
                peak_idx_in_all = np.where(all_peak_indices == selected_peak_idx)[0][0]
                peak_selection = "latest-before"
            elif len(peaks_before_indices) == 1:
                # Single peak before max_grip_time
                # Check if its 0-based index in the full list of peaks is at least 2 (3rd peak)
                peak_idx_in_all = np.where(all_peak_indices == peaks_before_indices[0])[0][0]
                
                # Check if there are no peaks for 80ms after max_grip (new condition)
                if len(peaks_after_indices) == 0:
                    # No peaks in the 80ms after max_grip, use the first peak
                    selected_peak_idx = peaks_before_indices[0]
                    selected_peak_time = T_proj[selected_peak_idx]
                    peak_selection = "single-before-no-peaks-after"
                elif peak_idx_in_all >= 2:
                    # It's at least the 3rd peak (index 2), use it
                    selected_peak_idx = peaks_before_indices[0]
                    selected_peak_time = T_proj[selected_peak_idx]
                    peak_selection = "single-before-index-2+"
            
            # If no peak selected yet (either no peaks before max_grip or single peak with index < 2)
            if selected_peak_idx is None:
                # Select specifically the 2nd peak (index 1) from t=0
                if len(all_peak_indices) >= 2:
                    selected_peak_idx = all_peak_indices[1]  # Second peak (index 1)
                    selected_peak_time = T_proj[selected_peak_idx]
                    peak_selection = "second-peak"
                else:
                    # Not enough peaks, skip this trial
                    continue
            
            # If we got here, we have a valid peak
            # Store the times and trial index
            trial_indices.append(trial_idx)
            trial_max_grip_times.append(max_grip_time)
            trial_trigger_peak_times.append(selected_peak_time)
            
            # Get the final peak index for plotting
            peak_idx_in_all = np.where(all_peak_indices == selected_peak_idx)[0][0]
            
            # Store additional data for plotting
            selected_peak_value = trial_trig[selected_peak_idx]
            
            all_trial_data.append({
                'trial_idx': trial_idx,
                'grip_ap': trial_grip_ap,
                'trig_proj': trial_trig,
                'max_grip_time': max_grip_time,
                'max_grip_idx': max_grip_idx,
                'max_grip_value': max_grip_value,
                'selected_peak_time': selected_peak_time,
                'selected_peak_idx': selected_peak_idx,
                'selected_peak_value': selected_peak_value,
                'peak_idx_in_all': peak_idx_in_all,
                'search_start_time': search_start_time,
                'peak_selection': peak_selection,
                'all_peak_indices': all_peak_indices,
                'all_peak_times': all_peak_times,
                'peaks_before_indices': peaks_before_indices,
                'peaks_before_times': peaks_before_times
            })
        
        # Skip if no valid trials
        if len(trial_indices) == 0:
            print(f"No valid trials found for session {session_id}")
            continue
            
        # Create a list to track peak selection criteria
        peak_selection_list = [d['peak_selection'] for d in all_trial_data]
        
        # Create color list
        colors = [color_map[selection] for selection in peak_selection_list]
        
        # Create the figure using plotly
        fig = go.Figure()
        
        # Add scatter plot with color coding based on peak selection criteria
        for selection_type in color_map.keys():
            # Filter indices for this selection type
            indices = [i for i, selection in enumerate(peak_selection_list) if selection == selection_type]
            
            if indices:  # Only add trace if there are points of this type
                labels = {
                    'single-before-index-2+': 'Single Peak Before (≥3rd Peak)',
                    'single-before-no-peaks-after': 'Single Peak Before (No Peaks After)',
                    'latest-before': 'Latest of Multiple Peaks Before',
                    'second-peak': '2nd Peak From t=0'
                }
                
                fig.add_trace(go.Scatter(
                    x=[trial_trigger_peak_times[i] for i in indices],
                    y=[trial_max_grip_times[i] for i in indices],
                    mode='markers',
                    marker=dict(
                        color=color_map[selection_type],
                        size=12,
                        opacity=0.7,
                        line=dict(width=1, color='black')  # Add black outline to markers
                    ),
                    name=labels[selection_type]
                ))
        
        # Calculate and add regression line
        if len(trial_max_grip_times) > 2:
            # Calculate correlation between trigger peak times and max grip times
            corr, p_value = pearsonr(trial_trigger_peak_times, trial_max_grip_times)
            
            # Calculate regression line
            m, b = np.polyfit(trial_trigger_peak_times, trial_max_grip_times, 1)
            x_range = np.linspace(min(trial_trigger_peak_times), max(trial_trigger_peak_times), 100)
            y_range = m * x_range + b
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Regression Line'
            ))
            
            # Add correlation info as text annotation
            time_diff = np.array(trial_max_grip_times) - np.array(trial_trigger_peak_times)
            avg_time_diff = np.mean(time_diff)
            
            corr_note = f"Correlation between:<br>- X: Trigger Peak Times<br>- Y: Max Grip Times"
            text_str = f'Pearson r: {corr:.3f}<br>p-value: {p_value:.4f}<br>Slope: {m:.3f}<br>Intercept: {b:.3f}<br>n = {len(trial_indices)}<br><br>{corr_note}'
            
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=text_str,
                showarrow=False,
                align="left",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            # Also calculate correlation statistics for printing later
            alt_corr, alt_p_value = pearsonr(trial_max_grip_times, trial_trigger_peak_times)
            print(f"Session {session_id} - Alternative correlation (Max Grip → Trigger Peak): r={alt_corr:.3f}, p={alt_p_value:.4f}")
            
            # Also calculate correlation between peak indices and max grip times
            peak_indices = [d['peak_idx_in_all'] + 1 for d in all_trial_data]  # 1-based for display
            idx_corr, idx_p_value = pearsonr(peak_indices, trial_max_grip_times)
            print(f"Session {session_id} - Correlation between peak indices and max grip times: r={idx_corr:.3f}, p={idx_p_value:.4f}")
        else:
            corr, p_value = None, None
            avg_time_diff = None
        
        # Add diagonal line (y=x) for reference
        min_val = min(min(trial_trigger_peak_times), min(trial_max_grip_times))
        max_val = max(max(trial_trigger_peak_times), max(trial_max_grip_times))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dot', width=1),
            opacity=0.5,
            name='y=x line'
        ))
        
        # Determine the axis ranges to make square
        # Find the overall min and max across both x and y
        overall_min = min(min_val, min_val)
        overall_max = max(max_val, max_val)
        
        # Create some padding (5%)
        padding = (overall_max - overall_min) * 0.05
        axis_min = overall_min - padding
        axis_max = overall_max + padding
        
        # Update layout with black axis at bottom and left and square aspect ratio
        fig.update_layout(
            title=f'Session {session_id}: Correlation Analysis',
            xaxis=dict(
                title='Trigger Peak Time (s)',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=False,
                range=[axis_min, axis_max]
            ),
            yaxis=dict(
                title='Max Grip Time (s)',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=False,
                range=[axis_min, axis_max]
            ),
            # Set aspect ratio to be equal
            yaxis_scaleanchor="x",
            yaxis_scaleratio=1,
            # Put legend below the plot
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            width=800,
            height=800,  # Made height equal to width for square aspect ratio
            plot_bgcolor='white',
            margin=dict(t=100, b=150),  # Increased bottom margin for legend
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # Add super title
        fig.update_layout(
            annotations=[
                dict(
                    text=f"Session {session_id} Analysis",
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        
        # Save the figure if a path is provided
        if save_path:
            session_save_path = f"{save_path}_session_{session_id}.html"
            fig.write_html(session_save_path)
            # Also save as image
            img_save_path = f"{save_path}_session_{session_id}.png"
            fig.write_image(img_save_path, scale=3)
        
        fig.show()
        
        # Count selection types
        selection_counts = {}
        for selection in ['single-before-index-2+', 'single-before-no-peaks-after', 'latest-before', 'second-peak']:
            selection_counts[selection] = sum(1 for d in all_trial_data if d['peak_selection'] == selection)
        
        # Print summary
        print(f"\nSummary for Session {session_id}:")
        print(f"Total trials analyzed: {len(trial_indices)}")
        print(f"Trials with single peak before max grip (≥3rd peak): {selection_counts['single-before-index-2+']} ({100*selection_counts['single-before-index-2+']/len(trial_indices):.1f}%)")
        print(f"Trials with single peak before max grip (no peaks for 80ms after): {selection_counts['single-before-no-peaks-after']} ({100*selection_counts['single-before-no-peaks-after']/len(trial_indices):.1f}%)")
        print(f"Trials with latest of multiple peaks before max grip: {selection_counts['latest-before']} ({100*selection_counts['latest-before']/len(trial_indices):.1f}%)")
        print(f"Trials using 2nd peak from t=0: {selection_counts['second-peak']} ({100*selection_counts['second-peak']/len(trial_indices):.1f}%)")
        
        if avg_time_diff is not None:
            print(f"Average time difference (max grip - trigger peak): {avg_time_diff:.3f}s")
        
        if corr is not None:
            print(f"Correlation: r = {corr:.3f}, p = {p_value:.4f}")
        
        # Store results for this session
        all_results[session_id] = {
            'trial_indices': trial_indices,
            'max_grip_times': trial_max_grip_times,
            'trigger_peak_times': trial_trigger_peak_times,
            'correlation': corr,
            'p_value': p_value,
            'selection_counts': selection_counts,
            'avg_time_diff': avg_time_diff
        }
    
    # Also create an overall summary figure
    if len(all_results) > 1:
        # Compile data from all sessions
        all_sessions = []
        all_correlations = []
        all_pvalues = []
        all_avg_diffs = []
        all_trial_counts = []
        
        for session_id, data in all_results.items():
            if data['correlation'] is not None:
                all_sessions.append(session_id)
                all_correlations.append(data['correlation'])
                all_pvalues.append(data['p_value'])
                all_avg_diffs.append(data['avg_time_diff'])
                all_trial_counts.append(len(data['trial_indices']))
        
        # Create summary figure with subplots
        fig = sp.make_subplots(rows=1, cols=2, 
                              subplot_titles=('Correlation by Session', 'Average Time Difference by Session'),
                              horizontal_spacing=0.1)
        
        # Plot correlations by session
        fig.add_trace(
            go.Bar(
                x=[str(s) for s in all_sessions],
                y=all_correlations,
                text=[f'{r:.3f}' for r in all_correlations],
                textposition='outside',
                marker_color='#4c78a8',  # Blue bar color
                marker_line_color='black',
                marker_line_width=1,
                name='Correlation (r)'
            ),
            row=1, col=1
        )
        
        # Plot average time differences by session
        fig.add_trace(
            go.Bar(
                x=[str(s) for s in all_sessions],
                y=all_avg_diffs,
                text=[f'{d:.3f}s' for d in all_avg_diffs],
                textposition='outside',
                marker_color='#e15759',  # Red-orange bar color
                marker_line_color='black',
                marker_line_width=1,
                name='Avg Time Diff (s)'
            ),
            row=1, col=2
        )
        
        # Update layout with black axes
        fig.update_layout(
            title_text='Summary Across All Sessions',
            showlegend=False,
            height=600,   # Increased height for better proportions
            width=1000,
            plot_bgcolor='white',
        )
        
        # Update x-axes (black axis at bottom)
        fig.update_xaxes(
            title_text='Session ID', 
            row=1, col=1, 
            tickangle=45,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False
        )
        
        fig.update_xaxes(
            title_text='Session ID', 
            row=1, col=2, 
            tickangle=45,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False
        )
        
        # Update y-axes (black axis at left)
        fig.update_yaxes(
            title_text='Correlation (r)', 
            row=1, col=1, 
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False
        )
        
        fig.update_yaxes(
            title_text='Avg Time Difference (s)', 
            row=1, col=2, 
            gridcolor='lightgray',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False
        )
        
        # Save the summary figure if a path is provided
        if save_path:
            summary_save_path = f"{save_path}_summary.html"
            fig.write_html(summary_save_path)
            # Also save as image
            img_summary_save_path = f"{save_path}_summary.png"
            fig.write_image(img_summary_save_path, scale=3)
        
        fig.show()
        
        # Print overall summary
        print("\nOverall Summary Across All Sessions:")
        print("---------------------------------")
        print(f"Total sessions analyzed: {len(all_sessions)}")
        print(f"Average correlation: {np.mean(all_correlations):.3f}")
        print(f"Average time difference: {np.mean(all_avg_diffs):.3f}s")
    
    return all_results


def plot_single_session_analysis_with_signals_plotly(sessions_data, session_id_to_plot, max_trials_to_show=10):
    """
    For a specific session:
    1. Plots the identified peak times for each trial
    2. Creates a correlation plot between trigger peak time and max grip time
    3. Plots the trigger and grip aperture signals for each trial with identified peaks using Plotly
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import pearsonr
    import pandas as pd
    
    # Find the requested session
    session_found = False
    for session in sessions_data:
        if session["session_id"] == session_id_to_plot:
            session_found = True
            break
    
    if not session_found:
        print(f"Session {session_id_to_plot} not found in data")
        return None
    
    # Extract relevant data
    T_behavior = session["T_behavior"]
    T_proj = session["T_proj"]
    grip_ap = session["grip_ap"]
    C = session["C"]
    
    # Custom function to find local maxima in a signal
    def find_local_maxima(signal, min_height=None, min_distance=3):
        """Find local maxima in a signal by comparing each point with its neighbors."""
        # Pad the signal to handle edge cases
        padded = np.pad(signal, 1, mode='edge')
        # Find local maxima by comparing with left and right neighbors
        mask = (padded[1:-1] > padded[0:-2]) & (padded[1:-1] > padded[2:])
        
        # Apply minimum height filter if specified
        if min_height is not None:
            mask = mask & (signal > min_height)
            
        # Get indices of all local maxima
        indices = np.where(mask)[0]
        
        # Apply minimum distance filter
        if len(indices) > 0 and min_distance > 1:
            # Sort peaks by height
            peak_heights = signal[indices]
            sorted_idx = np.argsort(peak_heights)[::-1]  # Sort in descending order
            sorted_indices = indices[sorted_idx]
            
            # Keep track of which peaks to keep
            keep = np.ones(len(sorted_indices), dtype=bool)
            
            # Iterate through sorted peaks
            for i in range(len(sorted_indices)):
                if keep[i]:
                    # Remove peaks that are within min_distance
                    idx = sorted_indices[i]
                    j = i + 1
                    while j < len(sorted_indices):
                        if abs(sorted_indices[j] - idx) < min_distance:
                            keep[j] = False
                        j += 1
            
            # Return indices of kept peaks, sorted by position
            indices = sorted(sorted_indices[keep])
        
        return np.array(indices)
    
    # Lists to store time points for each trial
    trial_indices = []
    trial_max_grip_times = []
    trial_trigger_peak_times = []
    
    # Lists to store data for signal plots
    all_trial_data = []
    
    # Process each trial
    for trial_idx in range(grip_ap.shape[0]):
        # Get this trial's grip aperture and trigger projection
        trial_grip_ap = grip_ap[trial_idx]
        trial_trig = C[:, trial_idx]
        
        # 1. Find the time point where grip aperture is maximum (constrained to after 80ms)
        start_idx_behavior = np.where(T_behavior >= 0.080)[0][0]  # Index where time >= 80ms
        max_grip_idx = start_idx_behavior + np.argmax(trial_grip_ap[start_idx_behavior:])
        max_grip_time = T_behavior[max_grip_idx]
        max_grip_value = trial_grip_ap[max_grip_idx]
        
        # 2. Apply smoothing to the trigger signal for better peak detection
        smoothed_trig = gaussian_filter1d(trial_trig, sigma=1)
        
        # 3. Find all peaks in the trigger signal after 0ms
        start_idx_neural = np.where(T_proj >= 0.0)[0][0]  # Index where time >= 0ms
        signal_after_0 = smoothed_trig[start_idx_neural:]
        
        # Find local maxima using custom function
        peak_indices_rel = find_local_maxima(signal_after_0, min_distance=3)
        
        # Convert to indices in the full signal
        all_peak_indices = peak_indices_rel + start_idx_neural
        all_peak_times = T_proj[all_peak_indices]
        
        if len(all_peak_indices) < 2:
            # Need at least 2 peaks for this analysis
            continue
        
        # 4. Define search window - we're looking both before and after max_grip_time
        search_start_time = max_grip_time - 0.100  # For visualization purposes
        
        # 5. Find peaks before max_grip_time
        peaks_before_mask = all_peak_times <= max_grip_time
        peaks_before_indices = all_peak_indices[peaks_before_mask]
        peaks_before_times = all_peak_times[peaks_before_mask]
        
        # 6. Find peaks shortly after max_grip_time (within 80ms)
        time_buffer_after_max = 0.080  # 80ms after max grip
        peaks_after_mask = (all_peak_times > max_grip_time) & (all_peak_times <= max_grip_time + time_buffer_after_max)
        peaks_after_indices = all_peak_indices[peaks_after_mask]
        
        # 7. Initialize variables to track peak selection
        selected_peak_idx = None
        selected_peak_time = None
        peak_selection = ""
        
        if len(peaks_before_indices) > 1:
            # Multiple peaks before max_grip_time, use the latest one
            selected_peak_idx = peaks_before_indices[-1]
            selected_peak_time = T_proj[selected_peak_idx]
            # Get the 0-based index of this peak in the full list of peaks
            peak_idx_in_all = np.where(all_peak_indices == selected_peak_idx)[0][0]
            peak_selection = "latest-before"
        elif len(peaks_before_indices) == 1:
            # Single peak before max_grip_time
            # Check if its 0-based index in the full list of peaks is at least 2 (3rd peak)
            peak_idx_in_all = np.where(all_peak_indices == peaks_before_indices[0])[0][0]
            
            # Check if there are no peaks for 80ms after max_grip (new condition)
            if len(peaks_after_indices) == 0:
                # No peaks in the 80ms after max_grip, use the first peak
                selected_peak_idx = peaks_before_indices[0]
                selected_peak_time = T_proj[selected_peak_idx]
                peak_selection = "single-before-no-peaks-after"
            elif peak_idx_in_all >= 2:
                # It's at least the 3rd peak (index 2), use it
                selected_peak_idx = peaks_before_indices[0]
                selected_peak_time = T_proj[selected_peak_idx]
                peak_selection = "single-before-index-2+"
        
        # If no peak selected yet (either no peaks before max_grip or single peak with index < 2)
        if selected_peak_idx is None:
            # Select specifically the 2nd peak (index 1) from t=0
            if len(all_peak_indices) >= 2:
                selected_peak_idx = all_peak_indices[1]  # Second peak (index 1)
                selected_peak_time = T_proj[selected_peak_idx]
                peak_selection = "second-peak"
            else:
                # Not enough peaks, skip this trial
                continue
        
        # Store peak information for analysis
        trial_indices.append(trial_idx)
        trial_max_grip_times.append(max_grip_time)
        trial_trigger_peak_times.append(selected_peak_time)
        
        # Find y-axis range for this trial based on signals
        grip_min = trial_grip_ap.min()
        grip_max = trial_grip_ap.max()
        trig_min = smoothed_trig.min()
        trig_max = smoothed_trig.max()
        
        # Calculate overall min and max with 10% margin
        y_min = min(grip_min, trig_min)
        y_max = max(grip_max, trig_max)
        y_margin = (y_max - y_min) * 0.1
        y_min -= y_margin
        y_max += y_margin
        
        # Store trial data for later plotting
        all_trial_data.append({
            'trial_idx': trial_idx,
            'trial_grip_ap': trial_grip_ap,
            'smoothed_trig': smoothed_trig,
            'max_grip_time': max_grip_time,
            'max_grip_value': max_grip_value,
            'max_grip_idx': max_grip_idx,
            'selected_peak_time': selected_peak_time,
            'selected_peak_idx': selected_peak_idx,
            'peak_selection': peak_selection,
            'all_peak_indices': all_peak_indices,
            'all_peak_times': all_peak_times,
            'y_min': y_min,
            'y_max': y_max
        })
    
    # Convert to numpy arrays for analysis
    trial_indices = np.array(trial_indices)
    trial_max_grip_times = np.array(trial_max_grip_times)
    trial_trigger_peak_times = np.array(trial_trigger_peak_times)
    
    # Calculate relative timing (trigger peak time relative to max grip time)
    relative_timing = trial_trigger_peak_times - trial_max_grip_times
    
    # Create plotting dataframe
    plot_df = pd.DataFrame({
        'trial': trial_indices,
        'max_grip_time': trial_max_grip_times,
        'trigger_peak_time': trial_trigger_peak_times,
        'relative_timing': relative_timing
    })
    
    # Calculate correlation
    corr, p_value = pearsonr(trial_max_grip_times, trial_trigger_peak_times)
    
    # Limit the number of trials to plot
    trials_to_plot = min(len(all_trial_data), max_trials_to_show)
    
    # Define colors
    grip_color = '#1D1A31'  # blue for grip aperture
    trigger_color = '#F08CAE'  # pink for neural trigger
    
    # Create a figure with subplots - one for each trial
    # First create a separate figure for summary plots
    fig_summary = sp.make_subplots(rows=2, cols=1, vertical_spacing=0.15,
                                   subplot_titles=["Trial Timing Analysis", 
                                                 "Max Grip Time vs. Trigger Peak Time"])
    
    # Now create a figure for each trial's signals
    figs_trials = []
    for i in range(trials_to_plot):
        trial_data = all_trial_data[i]
        # Create individual figure for each trial
        fig_trial = go.Figure()
        
        # Define the window of interest (grey area)
        window_of_interest_start = trial_data['max_grip_time'] - 0.150
        window_of_interest_end = trial_data['max_grip_time'] + 0.050
        
        # Get y-axis range for this trial
        y_min = trial_data['y_min']
        y_max = trial_data['y_max']
        
        # Add window of interest as grey rectangle
        fig_trial.add_shape(
            type="rect",
            x0=window_of_interest_start,
            y0=y_min,
            x1=window_of_interest_end,
            y1=y_max,
            fillcolor="lightgrey",
            opacity=0.3,
            line=dict(width=0)
        )
        
        # Add grip aperture signal in blue
        fig_trial.add_trace(
            go.Scatter(
                x=T_behavior,
                y=trial_data['trial_grip_ap'],
                mode='lines',
                line=dict(color=grip_color, width=2.5),  # Increased line width
                name='Grip Aperture'
            )
        )
        
        # Add trigger signal in pink
        fig_trial.add_trace(
            go.Scatter(
                x=T_proj,
                y=trial_data['smoothed_trig'],
                mode='lines',
                line=dict(color=trigger_color, width=2.5),  # Increased line width
                name='Neural Trigger'
            )
        )
        
        # Add vertical line for max grip time (same color as grip aperture)
        fig_trial.add_shape(
            type="line",
            x0=trial_data['max_grip_time'],
            y0=y_min,
            x1=trial_data['max_grip_time'],
            y1=y_max,
            line=dict(color=grip_color, width=2, dash="dash")  # Thicker line
        )
        
        # Add vertical line for selected trigger peak (same color as trigger)
        fig_trial.add_shape(
            type="line",
            x0=trial_data['selected_peak_time'],
            y0=y_min,
            x1=trial_data['selected_peak_time'],
            y1=y_max,
            line=dict(color=trigger_color, width=2, dash="dash")  # Thicker line
        )
        
        # Add markers for all detected peaks (same color as trigger)
        fig_trial.add_trace(
            go.Scatter(
                x=trial_data['all_peak_times'],
                y=trial_data['smoothed_trig'][trial_data['all_peak_indices']],
                mode='markers',
                marker=dict(color=trigger_color, size=10, symbol='circle'),  # Larger markers
                name='All Detected Peaks'
            )
        )
        
        # Add highlighted marker for selected peak (same color as trigger)
        fig_trial.add_trace(
            go.Scatter(
                x=[trial_data['selected_peak_time']],
                y=[trial_data['smoothed_trig'][trial_data['selected_peak_idx']]],
                mode='markers',
                marker=dict(color=trigger_color, size=14, symbol='star'),  # Larger marker
                name='Selected Peak'
            )
        )
        
        # Add highlighted marker for max grip point (same color as grip aperture)
        fig_trial.add_trace(
            go.Scatter(
                x=[trial_data['max_grip_time']],
                y=[trial_data['max_grip_value']],
                mode='markers',
                marker=dict(color=grip_color, size=14, symbol='star'),  # Larger marker
                name='Max Grip'
            )
        )
        
        # Calculate a sensible x-axis range to focus on the relevant part of the signals
        # Default to a 1-second window centered on max grip time, but ensure we don't go out of bounds
        x_center = trial_data['max_grip_time']
        x_margin = 0.5  # 0.5 seconds on each side
        x_min = max(T_behavior[0], x_center - x_margin)
        x_max = min(T_behavior[-1], x_center + x_margin)
        
        # Update layout for this trial's figure with higher resolution
        fig_trial.update_layout(
            title=f"Trial {trial_data['trial_idx']}",
            title_font=dict(size=18),  # Larger title font
            showlegend=False,
            height=500,  # Increased height for better visibility
            width=1200,  # Increased width
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=60, r=60, t=70, b=60),  # Increased margins
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=16)  # Larger font for all text
        )
        
        # Add thick black axis lines and set appropriate ranges
        fig_trial.update_xaxes(
            showline=True,
            linewidth=2.5,  # Thicker axis lines
            linecolor='black',
            mirror=True,
            gridcolor='lightgrey',
            gridwidth=1,
            range=[x_min, x_max],
            title=dict(text="Time (s)", font=dict(size=16))  # Larger axis title
        )
        
        fig_trial.update_yaxes(
            showline=True,
            linewidth=2.5,  # Thicker axis lines
            linecolor='black',
            mirror=True,
            gridcolor='lightgrey',
            gridwidth=1,
            range=[y_min, y_max],
            title=dict(text="Amplitude", font=dict(size=16))  # Larger axis title
        )
        
        # Add the figure to our collection
        figs_trials.append(fig_trial)
    
    # 1. Add peak timing plot to summary figure
    fig_summary.add_trace(
        go.Scatter(
            x=plot_df['trial'],
            y=plot_df['relative_timing'] * 1000,  # Convert to ms for readability
            mode='markers',
            marker=dict(color='black', size=12),  # Larger markers
            name='Trigger-Max Grip Time Difference (ms)'
        ),
        row=1, col=1
    )
    
    # Add horizontal line at y=0
    fig_summary.add_shape(
        type="line",
        x0=min(plot_df['trial']) - 0.5,
        y0=0,
        x1=max(plot_df['trial']) + 0.5,
        y1=0,
        line=dict(color="gray", width=2, dash="dash"),  # Thicker line
        row=1, col=1
    )
    
    # 2. Add correlation plot to summary figure
    fig_summary.add_trace(
        go.Scatter(
            x=plot_df['max_grip_time'],
            y=plot_df['trigger_peak_time'],
            mode='markers',
            marker=dict(color='black', size=12),  # Larger markers
            name='Max Grip Time vs Trigger Peak Time'
        ),
        row=2, col=1
    )
    
    # Add best fit line
    min_x = min(plot_df['max_grip_time'])
    max_x = max(plot_df['max_grip_time'])
    x_range = np.linspace(min_x, max_x, 100)
    
    # Calculate linear regression
    m, b = np.polyfit(plot_df['max_grip_time'], plot_df['trigger_peak_time'], 1)
    
    fig_summary.add_trace(
        go.Scatter(
            x=x_range,
            y=m * x_range + b,
            mode='lines',
            line=dict(color='red', width=3),  # Thicker line
            name=f'Best Fit (r={corr:.2f}, p={p_value:.4f})'
        ),
        row=2, col=1
    )
    
    # Update layout for summary figure with higher resolution
    fig_summary.update_layout(
        title=dict(
            text=f"Session {session_id_to_plot} Summary Analysis",
            font=dict(size=22)  # Larger title font
        ),
        height=800,  # Increased height
        width=1400,  # Increased width
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=16)  # Larger font for all text
    )
    
    # Add thick black axis lines to summary plots
    fig_summary.update_xaxes(
        showline=True,
        linewidth=2.5,  # Thicker axis lines
        linecolor='black',
        mirror=True,
        gridcolor='lightgrey',
        gridwidth=1
    )
    
    fig_summary.update_yaxes(
        showline=True,
        linewidth=2.5,  # Thicker axis lines
        linecolor='black',
        mirror=True,
        gridcolor='lightgrey',
        gridwidth=1
    )
    
    # Add x-axis titles to summary plots with larger font
    fig_summary.update_xaxes(
        title=dict(text="Trial Number", font=dict(size=18)),
        row=1, col=1
    )
    fig_summary.update_xaxes(
        title=dict(text="Max Grip Time (s)", font=dict(size=18)),
        row=2, col=1
    )
    
    # Add y-axis titles to summary plots with larger font
    fig_summary.update_yaxes(
        title=dict(text="Time Difference (ms)", font=dict(size=18)),
        row=1, col=1
    )
    fig_summary.update_yaxes(
        title=dict(text="Trigger Peak Time (s)", font=dict(size=18)),
        row=2, col=1
    )
    
    # Update subplot titles with larger font
    for i in fig_summary['layout']['annotations']:
        i['font'] = dict(size=20)
    
    # Add a annotation with session info
    info_text = (
        f"Session: {session_id_to_plot} | "
        f"Total trials: {len(plot_df)} | "
        f"Mean timing difference: {np.mean(relative_timing)*1000:.1f} ms | "
        f"Correlation: r = {corr:.2f}, p = {p_value:.4f}"
    )
    
    fig_summary.add_annotation(
        text=info_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=16),  # Larger font
        bordercolor="black",
        borderwidth=1.5,  # Thicker border
        borderpad=6,  # More padding
        bgcolor="white"
    )
    
    # High-resolution export configuration
    config = {
        'toImageButtonOptions': {
            'format': 'png',  # Export format
            'filename': f'session_{session_id_to_plot}',
            'height': 1200,
            'width': 1800,
            'scale': 5  # Increase DPI by 3x for very high resolution
        }
    }
    
    return {
        'summary': fig_summary,
        'trials': figs_trials,
        'info': {
            'session_id': session_id_to_plot,
            'n_trials': len(plot_df),
            'avg_time_diff_ms': np.mean(relative_timing)*1000,
            'correlation': corr,
            'p_value': p_value
        },
        'config': config  # Include config in the returned object
    }