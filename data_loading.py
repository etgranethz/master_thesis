import numpy as np
import h5py
import os
from scipy.signal import resample
import mat73
import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import lfilter
import matplotlib.pyplot as plt


def load_mat_data(data_path):
    """Load MAT file data from the specified path."""
    data = mat73.loadmat(data_path)
    data = data['trial_class']
    return data

import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import lfilter

def extract_data_around_movement_onset(data, n_pre_time_points=300, n_after_time_points=300, behavioral_keys=['digit1', 'digit3', 'hand'], positions=['digit1', 'digit3', 'hand']):
    """
    Extract neural and behavioral data around movement onset with a specified window.
    Neural data uses 2500 + mo_idx as reference, while behavioral data uses mo_idx directly.

    Parameters:
    -----------
    data : dict
        Dictionary containing neural and behavioral data.
    n_pre_time_points : int
        Number of time points to include before movement onset (default: 300).
    n_after_time_points : int
        Number of time points to include after movement onset (default: 300).
    behavioral_keys : list
        List of keys for behavioral data to extract around movement onset.
    positions : list
        List of position variables for which to compute derivatives.

    Returns:
    --------
    data_filtered_mo : dict
        Dictionary containing filtered data with neural and behavioral activity extracted around movement onset.
    """
    # First, filter data to keep only successful trials
    data_filtered = {key: [val[i] for i in range(len(data['success'])) if data['success'][i] == 1]
                    for key, val in data.items()}

    print(f"Total number of trials: {len(data['success'])}")
    print(f"Number of successful trials: {len(data_filtered['success'])}")

    # Extract data around movement onset
    m1_around_mo = []
    # Use a dictionary to store lists for each behavioral key and derivative
    behavioral_around_mo = {key: [] for key in behavioral_keys}
    for pos in positions:
        behavioral_around_mo[f"{pos}_deriv"] = []

    valid_trials = [] # Store indices of trials from the original successful trials list that are valid

    for i, (m1_data, mo_idx) in enumerate(zip(data_filtered['m1'], data_filtered['mo'])):
        # Convert mo_idx to integer
        mo_idx_int = int(mo_idx)

        # For neural data: use 2500 + mo_idx as reference
        neural_mo_idx = 2500 + mo_idx_int

        # --- Calculate start and end indices using n_pre/n_after ---
        neural_start_idx = neural_mo_idx - n_pre_time_points
        neural_end_idx = neural_mo_idx + n_after_time_points  # Add 1 for slicing exclusivity

        # Check if we have enough neural data points before and after mo
        n_neurons, n_time = m1_data.shape
        neural_valid = (neural_start_idx >= 0 and neural_end_idx <= n_time)

        # --- Behavioral validation ---
        behavioral_valid = True # Assume valid initially for this trial
        temp_behavioral_windows = {} # Store windows for *this trial* temporarily

        if neural_valid:
            # Neural part is okay, now check all required behavioral parts
            for key in behavioral_keys:
                if key in data_filtered and i < len(data_filtered[key]):
                    behav_data = data_filtered[key][i]
                    behav_mo_idx = mo_idx_int

                    # Calculate start and end indices for behavioral data
                    behav_start_idx = behav_mo_idx - n_pre_time_points
                    behav_end_idx = behav_mo_idx + n_after_time_points  # Add 1 for slicing exclusivity

                    # Check dimensions and calculate time length
                    if behav_data.ndim > 1:
                        behav_time_dim = behav_data.shape[-1]
                    else:
                        behav_time_dim = len(behav_data)

                    # Check if behavioral window is valid
                    if behav_start_idx >= 0 and behav_end_idx <= behav_time_dim:
                        # Extract window
                        if behav_data.ndim > 1:
                            behav_window = behav_data[..., behav_start_idx:behav_end_idx]
                        else:
                            behav_window = behav_data[behav_start_idx:behav_end_idx]

                        temp_behavioral_windows[key] = behav_window # Store valid window

                        # Compute derivative if needed
                        if key in positions:
                            if behav_data.ndim > 1:
                                deriv = np.diff(behav_window, axis=-1)
                                pad_shape = list(deriv.shape); pad_shape[-1] = 1
                                deriv = np.concatenate([deriv, deriv[..., -1:]], axis=-1)
                            else:
                                deriv = np.diff(behav_window)
                                deriv = np.append(deriv, deriv[-1])
                            temp_behavioral_windows[f"{key}_deriv"] = deriv # Store derivative
                    else:
                        # If *any* behavioral window is invalid, the trial is invalid
                        print(f"Warning: Trial {i}, {key} movement onset at {behav_mo_idx}, " +
                              f"but data has only {behav_time_dim} time points (needed indices {behav_start_idx} to {behav_end_idx-1}).")
                        behavioral_valid = False
                        break # Stop checking other behavioral keys for this trial
                else:
                    # Optional: Handle missing keys if necessary
                    print(f"Warning: Trial {i} missing expected behavioral key '{key}'.")
                    # Depending on requirements, this could invalidate the trial:
                    # behavioral_valid = False
                    # break

            # --- If trial passed all checks (neural and all behavioral) ---
            if behavioral_valid:
                # Extract the neural window now that we know the trial is fully valid
                m1_window = m1_data[:, neural_start_idx:neural_end_idx]
                m1_around_mo.append(m1_window)

                # Append all the temporarily stored behavioral data for this valid trial
                for key, window_data in temp_behavioral_windows.items():
                    # The list for this key should already exist in behavioral_around_mo
                    behavioral_around_mo[key].append(window_data)

                valid_trials.append(i) # Track index of the original successful trial
            else:
                 print(f"Trial {i}: Skipping due to invalid behavioral data window.")

        else:
            # Neural window was invalid
            print(f"Trial {i}: Neural movement onset at {neural_mo_idx}, " +
                  f"but trial has only {n_time} time points (needed indices {neural_start_idx} to {neural_end_idx-1}). Skipping.")

    # --- Construct the final output dictionary using the original logic ---
    # Start by filtering the original metadata keys based on valid_trials
    data_filtered_mo = {}
    original_success_count = len(data_filtered.get('success', [])) # Get length safely

    for key in data_filtered:
        # Keep keys that are lists/arrays aligned with the original successful trial count
        is_trial_aligned = isinstance(data_filtered[key], (list, np.ndarray)) and len(data_filtered[key]) == original_success_count

        # Exclude keys that will be replaced by windowed data
        is_windowed_key = key in ['m1'] or key in behavioral_around_mo # Check against keys we collected

        if is_trial_aligned and not is_windowed_key:
             # Filter this metadata based on the indices of the valid trials
             data_filtered_mo[key] = [data_filtered[key][i] for i in valid_trials]
        elif not is_trial_aligned and not is_windowed_key:
             # Keep non-trial-aligned metadata as is
             data_filtered_mo[key] = data_filtered[key]

    # Add the collected M1 window data
    data_filtered_mo['m1'] = m1_around_mo

    # Add the collected behavioral and derivative window data
    for key, data_list in behavioral_around_mo.items():
        if data_list: # Only add keys if we collected data for them
            data_filtered_mo[key] = data_list
            print(f"Extracted {len(data_list)} valid trials for {key}")

    # --- Smoothing ---
    window_length = 5; std = 1.0
    window = gaussian(window_length, std); window = window / window.sum()

    for pos in positions:
        deriv_key = f"{pos}_deriv"
        smoothed_key = f"{deriv_key}_smoothed"

        if deriv_key in data_filtered_mo: # Check if derivative data exists in the final dict
            valid_deriv_data = data_filtered_mo[deriv_key]
            smoothed_deriv_data = []
            for trial_data in valid_deriv_data:
                smoothed_trial = np.copy(trial_data)
                if trial_data.ndim == 2:
                    for dim in range(trial_data.shape[0]):
                        smoothed_trial[dim] = lfilter(window, 1, trial_data[dim])
                        edge_points = window_length - 1
                        if edge_points > 0: smoothed_trial[dim, :edge_points] = trial_data[dim, :edge_points]
                elif trial_data.ndim == 1:
                    smoothed_trial = lfilter(window, 1, trial_data)
                    edge_points = window_length - 1
                    if edge_points > 0: smoothed_trial[:edge_points] = trial_data[:edge_points]
                smoothed_deriv_data.append(smoothed_trial)

            data_filtered_mo[smoothed_key] = smoothed_deriv_data
            print(f"Created smoothed derivatives for {pos} ({len(smoothed_deriv_data)} trials)")
        else:
             print(f"No valid derivative data found for {pos} to smooth.")

    final_valid_trial_count = len(valid_trials)
    print(f"Number of trials with valid movement onset windows: {final_valid_trial_count}")

    # --- Verification Print Statements ---
    if final_valid_trial_count > 0:
        total_time_points = n_pre_time_points + n_after_time_points + 1
        if 'm1' in data_filtered_mo:
             print(f"Shape of extracted M1 data: {np.array(data_filtered_mo['m1'][0]).shape} (neurons × {total_time_points} time points)")
             print(f"Neural time window: {total_time_points} points ({n_pre_time_points} before, {n_after_time_points} after) centered on movement onset (2500 + mo_idx)")

        # Iterate through the keys we actually added to the final dict
        for key in data_filtered_mo:
            if key != 'm1' and isinstance(data_filtered_mo[key], list) and data_filtered_mo[key]:
                 # Check if it's a derivative or smoothed key
                 is_derivative = key.endswith('_deriv')
                 is_smoothed = key.endswith('_smoothed')
                 is_original_behav = not is_derivative and not is_smoothed and key in behavioral_keys # Check if it's one of the original requested behavioral keys

                 print(f"Shape of extracted {key} data: {np.array(data_filtered_mo[key][0]).shape}")
                 if is_original_behav:
                     print(f"Behavioral time window: {total_time_points} points ({n_pre_time_points} before, {n_after_time_points} after) centered on movement onset (mo_idx)")

    return data_filtered_mo

def process_data(data_filtered, base_behavior_keys=None, behavior_keys=None):
    """
    Process behavioral and neural data from filtered data.
    
    Parameters:
    -----------
    data_filtered : dict
        Filtered data dictionary containing neural and behavioral data
    base_behavior_keys : list, optional
        List of base behavioral keys to process (default: ['digit1', 'digit3', 'hand', 'pd'])
    behavior_keys : list, optional
        List of all behavior keys including derivatives (default: base keys + derivatives)
        
    Returns:
    --------
    tuple
        (session_keys, raw_spikes, behavior, conds)
    """
    import numpy as np
    
    # Set default keys if not provided
    if base_behavior_keys is None:
        base_behavior_keys = ['digit1', 'digit3', 'hand', 'pd']
    
    if behavior_keys is None:
        behavior_keys = base_behavior_keys + ['digit1_deriv', 'digit3_deriv', 'hand_deriv']
    
    # Initialize dictionaries to store data
    raw_spikes = {}      # Neural spike data
    behavior = {}        # Combined behavioral data
    behavior_by_type = {} # Separate dictionaries for each behavior type
    conds = {}           # Condition labels will be processed separately

    # Get unique sessions
    sessions = np.unique(data_filtered['session'])
    session_keys = []    # Will contain only successfully processed sessions
   
    # Process data session by session
    for session_id in sessions:
        # Convert session_id to string when using as a dictionary key
        session_key = str(session_id)
        
        try:
            # Get trial indices for this session
            session_trials = [i for i, s in enumerate(data_filtered['session']) if s == session_id]
            
            # Initialize behavior_by_type dictionary for this session
            behavior_by_type[session_key] = {}
            
            #---------- NEURAL DATA PROCESSING ----------#
            # Get the spike data for these trials
            session_spikes_original = [data_filtered['m1'][i] for i in session_trials]
            print(len(session_spikes_original))

            # Reshape each trial to swap dimensions (neurons and timepoints)
            session_spikes_reshaped = []
            for trial in session_spikes_original:
                # Transpose to swap neurons and timepoints
                reshaped_trial = trial.T  # This swaps the two dimensions
                session_spikes_reshaped.append(reshaped_trial)
            
            #---------- BEHAVIORAL DATA PROCESSING ----------#
            # First determine which trials have valid behavioral data for all keys
            valid_trial_indices = []  # Keep track of which trials have valid behavioral data
            
            # First pass: Check which trials have valid behavioral data for all base keys
            for trial_idx_in_session, global_trial_idx in enumerate(session_trials):
                has_all_behaviors = True
                for bkey in base_behavior_keys:  # Only check base keys, not derivatives
                    # Check if this trial has valid behavioral data
                    if global_trial_idx >= len(data_filtered[bkey]):
                        has_all_behaviors = False
                        break
                
                if has_all_behaviors:
                    valid_trial_indices.append(trial_idx_in_session)
            
            # Now only keep spikes for trials with valid behavioral data
            valid_session_spikes = [session_spikes_reshaped[idx] for idx in valid_trial_indices]
            valid_global_trial_indices = [session_trials[idx] for idx in valid_trial_indices]
            
            # Update session_trials to only include valid trials
            session_trials = valid_global_trial_indices
            
            # Try to convert neural data to numpy array
            # This will only work if all trials have the same shape
            spikes_array = np.array(valid_session_spikes)
            # Use string key instead of numeric key
            raw_spikes[session_key] = spikes_array
            print(f"Neural data for session {session_key}: {spikes_array.shape}")
            
            # Create condition labels (all ones for now)
            n_trials = raw_spikes[session_key].shape[0]
            conds[session_key] = np.ones(n_trials, dtype=np.uint8)
            
            # Process behavioral data for valid trials only
            behavior_arrays = {}
            
            # First process the base behavioral keys
            for bkey in base_behavior_keys:
                # Process each trial individually
                session_behavior_reshaped = []
                
                for trial_idx in session_trials:
                    # Get behavioral data for this specific trial
                    behavior_trial = data_filtered[bkey][trial_idx]
                    
                    # Check if this is the 'pd' key and it's 2D
                    if bkey == 'pd' and len(behavior_trial.shape) == 2:
                        # If pd has shape (timesteps, 1), it's already correct
                        if behavior_trial.shape[1] == 1:
                            pass
                        # If pd has shape (1, timesteps), transpose it
                        elif behavior_trial.shape[0] == 1:
                            behavior_trial = behavior_trial.T
                        # If pd is just a 1D array, expand dimension
                        elif len(behavior_trial.shape) == 1:
                            behavior_trial = behavior_trial.reshape(-1, 1)
                    # For other keys with 3D data
                    elif behavior_trial.shape[0] == 3:  # If first dimension is 3
                        behavior_trial = behavior_trial.T  # Transpose to make it (timesteps, 3)
                    
                    session_behavior_reshaped.append(behavior_trial)
                
                # Try to convert to numpy array
                # This will only work if all trials have the same shape
                behavior_array = np.array(session_behavior_reshaped)
                
                # Special handling for pd - expand dimension if needed
                if bkey == 'pd' and len(behavior_array.shape) == 2:
                    behavior_array = behavior_array.reshape(behavior_array.shape[0], behavior_array.shape[1], 1)
                elif bkey == 'pd' and len(behavior_array.shape) == 3 and behavior_array.shape[2] == 1:
                    # Already correct shape
                    pass
                elif bkey == 'pd':
                    # Expand dimensions to match the other arrays
                    behavior_array = np.expand_dims(behavior_array, axis=2)
                
                # Store in behavior_by_type dictionary
                behavior_by_type[session_key][bkey] = behavior_array
                # Also store in behavior_arrays for later concatenation
                behavior_arrays[bkey] = behavior_array
                print(f"{bkey} for session {session_key}: {behavior_array.shape}")
            
            # Now compute derivatives for digit1, digit3, and hand
            derivative_keys = ['digit1_deriv', 'digit3_deriv', 'hand_deriv']
            base_keys = ['digit1', 'digit3', 'hand']
            
            for deriv_key, base_key in zip(derivative_keys, base_keys):
                if base_key in behavior_arrays:
                    base_data = behavior_arrays[base_key]
                    
                    # Compute derivative for each trial
                    deriv_data = np.zeros_like(base_data)
                    
                    # For each trial
                    for trial in range(base_data.shape[0]):
                        # For each dimension (X, Y, Z)
                        for dim in range(base_data.shape[2]):
                            # Calculate derivative - simple difference
                            deriv_data[trial, 1:, dim] = np.diff(base_data[trial, :, dim], axis=0)
                            # First point has derivative 0
                            deriv_data[trial, 0, dim] = 0
                    
                    # Store in dictionaries
                    behavior_by_type[session_key][deriv_key] = deriv_data
                    behavior_arrays[deriv_key] = deriv_data
                    print(f"{deriv_key} for session {session_key}: {deriv_data.shape}")
            
            #---------- COMBINE BEHAVIORAL DATA ----------#
            # Now combine all behavioral data for this session
            if all(key in behavior_arrays for key in behavior_keys):
                # Check if all arrays have at least 3 dimensions
                all_have_3d = all(len(arr.shape) >= 3 for arr in behavior_arrays.values())
                
                if not all_have_3d:
                    # Fix any 2D arrays by adding a dimension
                    for key, arr in behavior_arrays.items():
                        if len(arr.shape) == 2:
                            behavior_arrays[key] = np.expand_dims(arr, axis=2)
                            print(f"Expanded {key} dimensions to: {behavior_arrays[key].shape}")
                
                # Check if all arrays have the same shape in first 2 dimensions
                shapes_match = all(arr.shape[0:2] == behavior_arrays[behavior_keys[0]].shape[0:2] 
                                for arr in behavior_arrays.values())
                
                if shapes_match:
                    # Concatenate along the last dimension
                    combined_behavior = np.concatenate([
                        behavior_arrays['digit1'],
                        behavior_arrays['digit3'],
                        behavior_arrays['hand'],
                        behavior_arrays['digit1_deriv'],
                        behavior_arrays['digit3_deriv'],
                        behavior_arrays['hand_deriv'],
                        behavior_arrays['pd']
                    ], axis=2)
                    
                    # Store the combined behavior
                    behavior[session_key] = combined_behavior
                    print(f"Combined behavior for session {session_key}: {combined_behavior.shape}")
                else:
                    raise Exception(f"Cannot combine behavioral data for session {session_key}: shapes don't match")
            else:
                raise Exception(f"Cannot combine behavioral data for session {session_key}: missing some behavioral keys")

            # If we got here without exceptions, add to session_keys
            session_keys.append(session_key)
            
        except Exception as e:
            print(f"Error processing session {session_key}: {str(e)}")
            # Remove all data for this session if it exists in any dictionary
            if session_key in raw_spikes:
                del raw_spikes[session_key]
            if session_key in behavior:
                del behavior[session_key]
            if session_key in behavior_by_type:
                del behavior_by_type[session_key]
            if session_key in conds:
                del conds[session_key]
    
    # Print summary
    print("\nData Processing Summary:")
    print(f"Number of sessions processed: {len(session_keys)}")
    print(f"Neural data sessions: {list(raw_spikes.keys())}")
    print(f"Combined behavioral data sessions: {list(behavior.keys())}")

    # Example of how to access the data:
    for session_key in list(raw_spikes.keys())[:3]:  # Just show first 3 sessions
        if session_key in behavior:
            print(f"\nSession {session_key}:")
            print(f"  Neural data: {raw_spikes[session_key].shape}")
            print(f"  Combined behavior: {behavior[session_key].shape}")
    
    return session_keys, raw_spikes, behavior, conds

def movmean_one_sided(arr, window_size):
    """
    Compute the moving average over the current bin and the next (window_size - 1) bins.
    Handles edge effects by reducing the window size when beyond array bounds.
    
    Args:
        arr (numpy.ndarray): Input array of shape (neurons, time) or (time, neurons).
        window_size (int): Size of the moving window (e.g., 5).
        
    Returns:
        numpy.ndarray: Array of the same shape as arr with the moving average applied.
    """
    # Check if array needs to be transposed (if time is first dimension)
    transpose_needed = arr.shape[0] > arr.shape[1]  # Assume more time points than neurons
    
    if transpose_needed:
        # Transpose to (neurons, time) format
        arr = arr.T
    
    neurons, time = arr.shape
    result = np.empty_like(arr)
    
    for i in range(time):
        end_idx = min(i + window_size, time)
        effective_window_size = end_idx - i
        result[:, i] = np.sum(arr[:, i:end_idx], axis=1) #/ effective_window_size
    
    # Transpose back if needed
    if transpose_needed:
        result = result.T
        
    return result


def process_spikes_with_movmean(spikes, window_size=5, subsample_factor=5):
    """
    Process spike data using moving mean and subsampling.
    
    Args:
        spikes (dict): Dictionary with session IDs as keys and spike data as values.
        window_size (int): Window size for moving mean.
        subsample_factor (int): Factor for subsampling.
        
    Returns:
        dict: Processed spikes with same structure as input.
    """
    processed_spikes = {}
    
    # Process each session
    for session_id, session_spikes in spikes.items():
        session_processed = []
        
        # Handle both array and list cases
        if isinstance(session_spikes, np.ndarray):
            # If it's an array: [trials, timepoints, neurons]
            n_trials = session_spikes.shape[0]
            
            for trial_idx in range(n_trials):
                # Get the trial data
                trial_data = session_spikes[trial_idx]  # Shape: [timepoints, neurons]
                
                # Extract specific neural timepoints if needed
                neural_t_start_idx = 0
                neural_t_end_idx = trial_data.shape[0]
                m1_new = trial_data[neural_t_start_idx:neural_t_end_idx]
                
                # 1. Apply moving mean
                m1_movmean = movmean_one_sided(m1_new, window_size=window_size)
                
                # 2. Subsample
                m1_subsampled = m1_movmean[0::subsample_factor]
                
                session_processed.append(m1_subsampled)
        else:
            # If it's a list of trials
            for trial_idx, trial_data in enumerate(session_spikes):
                # Extract specific neural timepoints if needed
                neural_t_start_idx = 0
                neural_t_end_idx = trial_data.shape[0]
                m1_new = trial_data[neural_t_start_idx:neural_t_end_idx]
                
                # Apply moving mean
                m1_movmean = movmean_one_sided(m1_new, window_size=window_size)
                
                # Subsample
                m1_subsampled = m1_movmean[0::subsample_factor]
                
                session_processed.append(m1_subsampled)
        
        # Try to convert to array if possible
        try:
            processed_spikes[session_id] = np.array(session_processed)
        except:
            processed_spikes[session_id] = session_processed

    return processed_spikes

def process_neural_data(raw_spikes, conds, bin_width_sec=0.01, std_sec=0.005):
    """
    Process neural spike data by smoothing with a Gaussian filter and calculating PSTHs.
    
    Parameters:
    -----------
    raw_spikes : dict
        Dictionary containing spike data for each session.
    conds : dict
        Dictionary containing condition labels for each session.
    bin_width_sec : float
        Width of the time bins in seconds. Default is 0.01 (10ms).
    std_sec : float
        Standard deviation of the Gaussian filter in seconds. Default is 0.005 (5ms).
        
    Returns:
    --------
    smth_spikes : dict
        Dictionary containing smoothed spike data for each session.
    psths : dict
        Dictionary containing PSTHs for each condition in each session.
    window : ndarray
        The Gaussian window used for smoothing.
    """
    # Create the Gaussian window
    std = std_sec / bin_width_sec
    M = int(std * 3 * 2)  # 3 standard deviations on each side
    window = gaussian(M, std)
    window = window / window.sum()  # Normalize so the window sums to 1
    
    # Remove convolution artifacts
    invalid_len = len(window) // 2
    
    # Smooth the spike data
    smth_spikes = {}
    for session in raw_spikes:
        # Convolve each session with the gaussian window
        smth_spikes[session] = lfilter(window, 1, raw_spikes[session], axis=1)[:, invalid_len:, :]
    
    # Calculate PSTHs
    psths = {}
    for session in raw_spikes:
        print(session)
        sess_smth_spikes = smth_spikes[session]
        sess_conds = conds[session]
        sess_psths = []
        for cond in np.unique(sess_conds):
            psth = np.mean(sess_smth_spikes[sess_conds == cond], axis=0)
            sess_psths.append(psth)
        psths[session] = np.array(sess_psths)
    
    return smth_spikes, psths, window


def format_session_key(session_key):
    return str(float(session_key))


def update_dict_keys_inplace(dictionary):
    """
    Update dictionary keys in-place from float string format ('2.0') to integer string format ('2')
    """
    if not dictionary:
        return
    
    # Get list of keys that need to be changed
    keys_to_update = []
    for key in list(dictionary.keys()):
        try:
            if '.' in key:  # Only process keys that have a decimal point
                new_key = str(int(float(key)))
                if new_key != key:
                    keys_to_update.append((key, new_key))
        except (ValueError, TypeError):
            # Skip keys that can't be converted
            pass
    
    # Update the keys
    for old_key, new_key in keys_to_update:
        dictionary[new_key] = dictionary.pop(old_key)


import matplotlib.pyplot as plt


def visualize_neural_activity(raw_spikes, smth_spikes, psths, conds, session, condition, T_behavior, T_neural, trial_index=0, figsize=(15, 5)):
    """
    Visualize neural activity data with three subplots showing raw spikes, smoothed spikes, and PSTHs.
    
    Parameters:
    -----------
    raw_spikes : dict
        Dictionary containing raw spike data for each session
    smth_spikes : dict
        Dictionary containing smoothed spike data for each session
    psths : dict
        Dictionary containing PSTH data for each session
    conds : dict
        Dictionary containing condition labels for each session
    session : str
        Session identifier (e.g., '19.0')
    condition : int
        Condition number to visualize
    T_behavior : numpy.ndarray
        Time points for behavioral data
    T_neural : numpy.ndarray
        Time points for neural data
    trial_index : int, optional
        Index of the trial to visualize (default: 0)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (15, 5))
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Create figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=figsize)
    
    # Print data shapes for debugging/verification
    print(f"Spikes shape: {raw_spikes[session].shape}")
    print(f"Smoothed spikes shape: {smth_spikes[session][conds[session] == condition].shape}")
    print(f"PSTHs shape: {psths[session].shape}")
    
    # Plot 1: Raw Spikes
    im1 = ax1.imshow(raw_spikes[session][trial_index].T, 
                     aspect='auto',
                     extent=[T_behavior.min(), T_behavior.max(), 
                            0, raw_spikes[session][trial_index].shape[1]])
    ax1.set_title('Raw Spikes')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Neurons')
    
    # Plot 2: Smoothed Spikes
    im2 = ax2.imshow(smth_spikes[session][conds[session] == condition][trial_index].T,
                     aspect='auto',
                     extent=[T_neural.min(), T_neural.max(),
                            0, smth_spikes[session][conds[session] == condition][trial_index].shape[1]])
    ax2.set_title('Smoothed and binned Spikes')
    ax2.set_xlabel('Time (s)')
    
    # Plot 3: PSTHs
    im3 = ax3.imshow(psths[session][0].T,
                     aspect='auto', 
                     extent=[T_neural.min(), T_neural.max(),
                            0, psths[session][0].shape[1]])
    ax3.set_title('PSTHs')
    ax3.set_xlabel('Time (s)')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, label='Spike Count')
    plt.colorbar(im2, ax=ax2, label='Smoothed Activity')
    plt.colorbar(im3, ax=ax3, label='PSTH Value')
    
    plt.tight_layout()
    
    return fig




def extract_data_around_movement_onset_simple(data, vicinity=300, behavioral_keys=['digit1', 'digit3', 'hand'], positions=['digit1', 'digit3', 'hand']):
    """
    Extract neural and behavioral data around movement onset with a specified vicinity.
    Neural data uses 2500 + mo_idx as reference, while behavioral data uses mo_idx directly.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing neural and behavioral data.
    vicinity : int
        Number of time points to include before and after movement onset (default: 300).
        This results in a total window of 2*vicinity+1 time points.
    behavioral_keys : list
        List of keys for behavioral data to extract around movement onset.
    positions : list
        List of position variables for which to compute derivatives.
    
    Returns:
    --------
    data_filtered_mo : dict
        Dictionary containing filtered data with neural and behavioral activity extracted around movement onset.
    """
    # First, filter data to keep only successful trials
    data_filtered = {key: [val[i] for i in range(len(data['success'])) if data['success'][i] == 1] 
                    for key, val in data.items()}
    
    print(f"Total number of trials: {len(data['success'])}")
    print(f"Number of successful trials: {len(data_filtered['success'])}")
    
    # Extract data around movement onset
    data_filtered_mo = data_filtered.copy()
    m1_around_mo = []
    behavioral_around_mo = {key: [] for key in behavioral_keys}
    valid_trials = []
    
    # Initialize for position derivatives
    for pos in positions:
        behavioral_around_mo[f"{pos}_deriv"] = []
    
    for i, (m1_data, mo_idx) in enumerate(zip(data_filtered['m1'], data_filtered['mo'])):
        # Convert mo_idx to integer
        mo_idx_int = int(mo_idx)
        
        # For neural data: use 2500 + mo_idx as reference
        neural_mo_idx = 2500 + mo_idx_int
        
        # Calculate start and end indices for neural data
        neural_start_idx = neural_mo_idx - vicinity
        neural_end_idx = neural_mo_idx + vicinity
        
        # Check if we have enough neural data points before and after mo
        n_neurons, n_time = m1_data.shape
        
        neural_valid = (neural_start_idx >= 0 and neural_end_idx < n_time)
        behavioral_valid = True  # Will be updated for each behavioral key
        
        if neural_valid:
            # Extract the window around movement onset for M1 data
            m1_window = m1_data[:, neural_start_idx:neural_end_idx]
            
            # Extract the window around movement onset for behavioral data
            # For behavioral data: use mo_idx directly as reference
            for key in behavioral_keys:
                if key in data_filtered and i < len(data_filtered[key]):
                    behav_data = data_filtered[key][i]
                    
                    # For behavioral data, use mo_idx directly (without 2500 offset)
                    behav_mo_idx = mo_idx_int
                    
                    # Calculate start and end indices for behavioral data
                    behav_start_idx = behav_mo_idx - vicinity
                    behav_end_idx = behav_mo_idx + vicinity
                    
                    # Check if behavioral data has enough points
                    if behav_data.ndim > 1:
                        behav_time_dim = behav_data.shape[-1]
                    else:
                        behav_time_dim = len(behav_data)
                    
                    if behav_start_idx >= 0 and behav_end_idx < behav_time_dim:
                        # Extract the window for behavioral data
                        if behav_data.ndim > 1:
                            behav_window = behav_data[..., behav_start_idx:behav_end_idx]
                        else:
                            behav_window = behav_data[behav_start_idx:behav_end_idx]
                        
                        behavioral_around_mo[key].append(behav_window)
                        
                        # Compute derivative for position variables
                        if key in positions:
                            if behav_data.ndim > 1:
                                # For multi-dimensional data, compute derivative across last dimension (time)
                                # Shape: [dimensions, time] -> [dimensions, time-1]
                                deriv = np.diff(behav_window, axis=-1)
                                # Pad the derivative to match original size (repeat last value)
                                pad_shape = list(deriv.shape)
                                pad_shape[-1] = 1
                                deriv = np.concatenate([deriv, deriv[..., -1:]], axis=-1)
                            else:
                                # For 1D data
                                deriv = np.diff(behav_window)
                                # Pad to match original size
                                deriv = np.append(deriv, deriv[-1])
                            
                            behavioral_around_mo[f"{key}_deriv"].append(deriv)
                    else:
                        print(f"Warning: Trial {i}, {key} movement onset at {behav_mo_idx}, " +
                              f"but data has only {behav_time_dim} time points.")
                        behavioral_valid = False
                        behavioral_around_mo[key].append(None)
                        if key in positions:
                            behavioral_around_mo[f"{key}_deriv"].append(None)
            
            # Only add trial if both neural and behavioral data are valid
            if behavioral_valid:
                m1_around_mo.append(m1_window)
                valid_trials.append(i)
            else:
                print(f"Trial {i}: Skipping due to invalid behavioral data.")
        else:
            print(f"Trial {i}: Neural movement onset at {neural_mo_idx}, " +
                  f"but trial has only {n_time} time points. Skipping.")
    
    # Update the filtered data with only valid trials
    data_filtered_mo = {key: [data_filtered[key][i] for i in valid_trials] for key in data_filtered}
    
    # Replace M1 data with the extracted windows
    data_filtered_mo['m1'] = m1_around_mo
    
    # Replace behavioral data with the extracted windows
    for key in behavioral_keys:
        if key in data_filtered:
            # Get only the valid trials
            valid_behav_data = [behavioral_around_mo[key][i] for i in range(len(behavioral_around_mo[key])) 
                               if i < len(valid_trials)]
            
            # Filter out None values
            valid_behav_data = [data for data in valid_behav_data if data is not None]
            
            if valid_behav_data:
                data_filtered_mo[key] = valid_behav_data
                print(f"Extracted {len(valid_behav_data)} valid trials for {key}")
    


    # Create causal Gaussian window for smoothing
    window_length = 5  # Length of 5 units as specified
    std = 1.0  # Standard deviation
    window = gaussian(window_length, std)
    window = window / window.sum()  # Normalize the window

    # Fix the derivative smoothing implementation
    for pos in positions:
        deriv_key = f"{pos}_deriv"
        
        try:
            # Get valid derivative data
            valid_deriv_data = [behavioral_around_mo[deriv_key][i] for i in range(len(behavioral_around_mo[deriv_key]))
                            if i < len(valid_trials)]
            
            # Filter out None values
            valid_deriv_data = [data for data in valid_deriv_data if data is not None]
            
            if valid_deriv_data:
                # Apply causal Gaussian smoothing in the time dimension
                smoothed_deriv_data = []
                
                for trial_data in valid_deriv_data:
                    # Create a copy of the trial data
                    smoothed_trial = np.copy(trial_data)
                    
                    # Check the shape to determine how to apply smoothing
                    if len(trial_data.shape) == 2:  # Shape is (dimensions, time_points)
                        for dim in range(trial_data.shape[0]):
                            # Apply filter along the time dimension
                            smoothed_trial[dim] = lfilter(window, 1, trial_data[dim])
                            
                            # Keep the first few points unchanged (no smoothing at edges)
                            edge_points = window_length - 1
                            if edge_points > 0:
                                smoothed_trial[dim, :edge_points] = trial_data[dim, :edge_points]
                    
                    elif len(trial_data.shape) == 3:  # Shape is (time_points, dimensions, ?)
                        # Handle 3D array (like what appears in your error logs)
                        n_time, n_dims, third_dim = trial_data.shape
                        
                        for dim in range(n_dims):
                            for i in range(third_dim):
                                # Apply filter along time dimension
                                smoothed_trial[:, dim, i] = lfilter(window, 1, trial_data[:, dim, i])
                                
                                # Keep edge points unchanged
                                edge_points = window_length - 1
                                if edge_points > 0:
                                    smoothed_trial[:edge_points, dim, i] = trial_data[:edge_points, dim, i]
                    
                    smoothed_deriv_data.append(smoothed_trial)
                
                # Store both the original and smoothed derivatives
                data_filtered_mo[deriv_key] = valid_deriv_data
                data_filtered_mo[f"{deriv_key}_smoothed"] = smoothed_deriv_data
                print(f"Extracted {len(valid_deriv_data)} valid derivatives for {pos}")
                print(f"Created smoothed derivatives for {pos}")
        
        except Exception as e:
            print(f"Error processing {deriv_key}: {e}")

    print(f"Number of trials with valid movement onset windows: {len(valid_trials)}")

        # Verify the shape of the extracted data
    if m1_around_mo:
        print(f"Shape of extracted M1 data: {m1_around_mo[0].shape} (neurons × time points)")
        print(f"Neural time window: {2*vicinity+1} points centered on movement onset (2500 + mo_idx)")
        
        for key in behavioral_keys:
            if key in data_filtered_mo and data_filtered_mo[key]:
                print(f"Shape of extracted {key} data: {data_filtered_mo[key][0].shape}")
                print(f"Behavioral time window: {2*vicinity+1} points centered on movement onset (mo_idx)")
        
        for pos in positions:
            deriv_key = f"{pos}_deriv"
            if deriv_key in data_filtered_mo and data_filtered_mo[deriv_key]:
                print(f"Shape of {deriv_key} data: {data_filtered_mo[deriv_key][0].shape}")
    
    return data_filtered_mo
