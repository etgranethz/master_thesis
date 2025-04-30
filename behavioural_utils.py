
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Define a function to identify outliers based on PD parameter only
def identify_pd_outliers(behavior_data, T_behavior, std_threshold=4, std_check_start_time=0.15, std_check_end_time=None):
    n_trials, n_timepoints, n_components = behavior_data.shape
    
    # Extract the pd component (should be at index 9 if behavior_data is organized as expected)
    pd_data = behavior_data[:, :, 18]  # Assuming pd is the 10th component (index 9)
    
    # Find the index corresponding to the start time for std check
    start_idx = np.argmin(np.abs(T_behavior - std_check_start_time))
    
    # Find the index corresponding to the end time for std check
    if std_check_end_time is not None:
        end_idx = np.argmin(np.abs(T_behavior - std_check_end_time))
        # Ensure end_idx is after start_idx
        if end_idx <= start_idx:
            end_idx = n_timepoints
    else:
        end_idx = n_timepoints
    
    # Calculate std threshold trials for pd only
    std_threshold_trials = np.zeros(n_trials, dtype=bool)
    
    # For each time point between start and end times
    for t in range(start_idx, end_idx):
        # Get all trials at this time point for pd
        time_slice = pd_data[:, t]
        time_mean = np.mean(time_slice)
        time_std = np.std(time_slice)
        
        # Upper and lower bounds
        upper_bound = time_mean + std_threshold * time_std
        lower_bound = time_mean - std_threshold * time_std
        
        # Check each trial at this time point
        for trial_idx in range(n_trials):
            value = pd_data[trial_idx, t]
            if value > upper_bound or value < lower_bound:
                std_threshold_trials[trial_idx] = True
    
    # We'll still compute grip aperture metrics for visualization
    thumb = behavior_data[:, :, 0:3]
    pinky = behavior_data[:, :, 3:6]
    grip_aperture = np.linalg.norm(thumb - pinky, ord=2, axis=2)
    
    # Find max grip aperture times for each trial (for visualization, not filtering)
    max_grip_times = T_behavior[np.argmax(grip_aperture, axis=1)]
    late_trials = max_grip_times > 0.3  # trials with max grip after 300ms
    
    # All trials are considered valid for pd filtering
    valid_trials = np.ones(n_trials, dtype=bool)
    
    # Return indices categorized
    return {
        'valid': np.where(valid_trials)[0],
        'normal': np.where(~std_threshold_trials)[0],
        'irregular_pd': np.where(std_threshold_trials)[0],
        'late_grip_aperture': np.where(late_trials)[0],  # Just for visualization
        'start_idx': start_idx,
        'end_idx': end_idx
    }

def plot_filtered_pd_trials(grip_ap, behavior_data, T_behavior, session_key, outlier_categories, 
                          filter_criteria, std_threshold, std_check_start_time, std_check_end_time=None):
    """
    Visualize trials filtered based on pd parameter values.
    
    Args:
        grip_ap: Array of shape (n_trials, n_timepoints) with grip aperture for each trial
        behavior_data: Full behavioral data array (n_trials, n_timepoints, n_components)
        T_behavior: Time vector for behavior data
        session_key: String identifying the session
        outlier_categories: Dictionary with categorized trial indices
        filter_criteria: Which criteria was used ('all', 'irregular_pd', 'none')
        std_threshold: Number of standard deviations used for outlier detection
        std_check_start_time: Time point (in seconds) used to start checking for outliers
        std_check_end_time: Optional time point (in seconds) used to end checking for outliers
    """
    # Extract PD data
    pd_data = behavior_data[:, :, 18]  # Assuming pd is the 10th component (index 9)
    
    # Determine which trials to include based on filter criteria
    if filter_criteria == 'all':
        # Exclude all outliers (keep only normal trials)
        included_trials = outlier_categories['normal']
    elif filter_criteria == 'irregular_pd':
        # Exclude trials with irregular pd values
        included_trials = np.setdiff1d(outlier_categories['valid'], outlier_categories['irregular_pd'])
    elif filter_criteria == 'none':
        # For 'none' filter criteria, use all trials
        included_trials = np.arange(grip_ap.shape[0])
    else:
        # Default: include all valid trials
        included_trials = outlier_categories['valid']
    
    # Calculate the set of excluded trials for plotting
    all_valid_trials = outlier_categories['valid']
    excluded_trials = np.setdiff1d(all_valid_trials, included_trials) if filter_criteria != 'none' else np.setdiff1d(np.arange(grip_ap.shape[0]), included_trials)
    
    # Separate excluded trials
    irregular_pd_trials = outlier_categories['irregular_pd']
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 24))
    
    # Define grid for subplots (added one more row for STD plot)
    gs = gridspec.GridSpec(6, 2, height_ratios=[3, 3, 3, 3, 3, 2])
    
    # 1. Plot of grip aperture with color coding by exclusion reason
    ax_grip = plt.subplot(gs[0, :])
    
    # Plot included trials in blue (with lighter transparency)
    for trial_idx in included_trials:
        ax_grip.plot(T_behavior, grip_ap[trial_idx], color='blue', alpha=0.2, linewidth=1)
    
    # Plot excluded trials
    for trial_idx in irregular_pd_trials:
        if trial_idx not in included_trials:
            ax_grip.plot(T_behavior, grip_ap[trial_idx], color='red', alpha=0.4, linewidth=1)
    
    # Plot means for each category
    if len(included_trials) > 0:
        included_mean = np.mean(grip_ap[included_trials], axis=0)
        ax_grip.plot(T_behavior, included_mean, color='blue', linewidth=3, label='Included Trials')
    
    if len(irregular_pd_trials) > 0:
        pd_outlier_mean = np.mean(grip_ap[irregular_pd_trials], axis=0)
        ax_grip.plot(T_behavior, pd_outlier_mean, color='red', linewidth=2, label='PD Outliers')
    
    # Add vertical lines at key time points
    ax_grip.axvline(x=0.0, color='gray', linestyle='--', alpha=0.7)
    ax_grip.axvline(x=std_check_start_time, color='purple', linestyle='--', alpha=0.7, 
                   label=f'STD Check Start ({std_check_start_time*1000:.0f}ms)')
    
    if std_check_end_time is not None:
        ax_grip.axvline(x=std_check_end_time, color='purple', linestyle=':', alpha=0.7, 
                       label=f'STD Check End ({std_check_end_time*1000:.0f}ms)')
    
    # Set labels and title
    ax_grip.set_xlabel('Time (s)')
    ax_grip.set_ylabel('Grip Aperture')
    filter_display = filter_criteria if filter_criteria != 'none' else 'None (Unfiltered)'
    ax_grip.set_title(f'Session {session_key}: Grip Aperture - Filter: {filter_display}')
    
    # Add legend outside the plot
    ax_grip.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    
    # 2. Plot PD component
    ax_pd = plt.subplot(gs[1, :])
    
    # Plot included trials in blue
    for trial_idx in included_trials:
        ax_pd.plot(T_behavior, pd_data[trial_idx], color='blue', alpha=0.2, linewidth=1)
    
    # Plot excluded trials
    for trial_idx in irregular_pd_trials:
        if trial_idx not in included_trials:
            ax_pd.plot(T_behavior, pd_data[trial_idx], color='red', alpha=0.4, linewidth=1)
    
    # Plot means for each category
    if len(included_trials) > 0:
        included_mean = np.mean(pd_data[included_trials], axis=0)
        ax_pd.plot(T_behavior, included_mean, color='blue', linewidth=3, label='Included Trials')
    
    if len(irregular_pd_trials) > 0:
        pd_outlier_mean = np.mean(pd_data[irregular_pd_trials], axis=0)
        ax_pd.plot(T_behavior, pd_outlier_mean, color='red', linewidth=2, label='PD Outliers')
    
    # Add vertical lines
    ax_pd.axvline(x=0.0, color='gray', linestyle='--', alpha=0.7)
    ax_pd.axvline(x=std_check_start_time, color='purple', linestyle='--', alpha=0.7)
    
    if std_check_end_time is not None:
        ax_pd.axvline(x=std_check_end_time, color='purple', linestyle=':', alpha=0.7)
    
    # Get start and end indices for STD check
    start_idx = outlier_categories['start_idx']
    end_idx = outlier_categories['end_idx']
    
    # Calculate and plot std bounds for pd at each timepoint in the check window
    for t in range(start_idx, end_idx, 25):  # Plot every 25th point to avoid clutter
        time_slice = pd_data[:, t]
        time_mean = np.mean(time_slice)
        time_std = np.std(time_slice)
        
        # Plot error bars at this timepoint
        upper_bound = time_mean + std_threshold * time_std
        lower_bound = time_mean - std_threshold * time_std
        ax_pd.plot([T_behavior[t], T_behavior[t]], [lower_bound, upper_bound], 
                  color='purple', alpha=0.3, linewidth=2)
    
    # Set labels and title
    ax_pd.set_xlabel('Time (s)')
    ax_pd.set_ylabel('PD Value')
    ax_pd.set_title(f'Session {session_key}: PD Component')
    
    # 3. Plot the standard deviation of PD over time
    ax_std = plt.subplot(gs[2, :])
    
    # Calculate mean and std of PD at each timepoint
    pd_mean = np.mean(pd_data, axis=0)
    pd_std = np.std(pd_data, axis=0)
    
    # Plot the mean PD value
    ax_std.plot(T_behavior, pd_mean, color='black', linewidth=2, label='Mean PD')
    
    # Plot the standard deviation bounds
    ax_std.fill_between(T_behavior, pd_mean - pd_std, pd_mean + pd_std, color='gray', alpha=0.3, label='±1 STD')
    ax_std.fill_between(T_behavior, pd_mean - 2*pd_std, pd_mean + 2*pd_std, color='gray', alpha=0.2, label='±2 STD')
    ax_std.fill_between(T_behavior, pd_mean - std_threshold*pd_std, pd_mean + std_threshold*pd_std, 
                       color='purple', alpha=0.2, label=f'±{std_threshold} STD (Threshold)')
    
    # Highlight the STD check region
    ax_std.axvspan(T_behavior[start_idx], T_behavior[min(end_idx, len(T_behavior)-1)], 
                  color='yellow', alpha=0.2, label='STD Check Region')
    
    # Add vertical lines
    ax_std.axvline(x=0.0, color='gray', linestyle='--', alpha=0.7)
    ax_std.axvline(x=std_check_start_time, color='purple', linestyle='--', alpha=0.7)
    
    if std_check_end_time is not None:
        ax_std.axvline(x=std_check_end_time, color='purple', linestyle=':', alpha=0.7)
    
    # Set labels and title
    ax_std.set_xlabel('Time (s)')
    ax_std.set_ylabel('PD Value')
    ax_std.set_title(f'Session {session_key}: PD Mean and Standard Deviation')
    ax_std.legend(loc='upper right')
    
    # 4-5. Plot sample hand position components for context
    component_indices = [0, 6]  # Thumb X and Hand X
    component_names = ["Thumb X", "Hand X"]
    
    for i, comp_idx in enumerate(component_indices):
        ax_comp = plt.subplot(gs[i+3, :])
        
        # Plot included trials in blue
        for trial_idx in included_trials:
            ax_comp.plot(T_behavior, behavior_data[trial_idx, :, comp_idx], color='blue', alpha=0.2, linewidth=1)
        
        # Plot excluded trials
        for trial_idx in irregular_pd_trials:
            if trial_idx not in included_trials:
                ax_comp.plot(T_behavior, behavior_data[trial_idx, :, comp_idx], color='red', alpha=0.4, linewidth=1)
        
        # Plot means for each category
        if len(included_trials) > 0:
            included_mean = np.mean(behavior_data[included_trials, :, comp_idx], axis=0)
            ax_comp.plot(T_behavior, included_mean, color='blue', linewidth=3, label='Included Trials')
        
        if len(irregular_pd_trials) > 0:
            pd_outlier_mean = np.mean(behavior_data[irregular_pd_trials, :, comp_idx], axis=0)
            ax_comp.plot(T_behavior, pd_outlier_mean, color='red', linewidth=2, label='PD Outliers')
        
        # Add vertical lines
        ax_comp.axvline(x=0.0, color='gray', linestyle='--', alpha=0.7)
        ax_comp.axvline(x=std_check_start_time, color='purple', linestyle='--', alpha=0.7)
        
        if std_check_end_time is not None:
            ax_comp.axvline(x=std_check_end_time, color='purple', linestyle=':', alpha=0.7)
        
        # Set labels and title
        ax_comp.set_xlabel('Time (s)')
        ax_comp.set_ylabel(component_names[i])
        ax_comp.set_title(f'Session {session_key}: {component_names[i]}')
    
    # 6. Trial distribution - bar chart showing breakdown
    ax_bar = plt.subplot(gs[5, :])
    
    # Create data for the bar chart
    categories = ['PD Outliers', 'Total Excluded', 'Included']
    counts = [
        len(irregular_pd_trials),
        len(excluded_trials),
        len(included_trials)
    ]
    
    # Create bar colors
    colors = ['red', 'crimson', 'blue']
    
    # Create the bar chart
    bars = ax_bar.bar(categories, counts, color=colors, alpha=0.7)
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add total trials information
    total_trials = len(all_valid_trials)
    if filter_criteria == 'none':
        total_trials = grip_ap.shape[0]  # All trials for 'none' filtering
        
    ax_bar.text(0.98, 0.95, f'Total trials: {total_trials}', 
              transform=ax_bar.transAxes, ha='right', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add more detailed info
    check_range_text = f"{std_check_start_time*1000:.0f}ms"
    if std_check_end_time is not None:
        check_range_text += f" to {std_check_end_time*1000:.0f}ms"
    else:
        check_range_text += " to end"
        
    info_text = (
        f"Filter criteria: {filter_display}\n"
        f"STD threshold: {std_threshold}\n"
        f"STD check range: {check_range_text}\n"
        f"% Excluded: {len(excluded_trials)/total_trials*100:.1f}%"
    )
    ax_bar.text(0.02, 0.95, info_text, 
              transform=ax_bar.transAxes, ha='left', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax_bar.set_ylabel('Number of Trials')
    ax_bar.set_title(f'Session {session_key}: Trial Distribution')
    
    # Final layout adjustments
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nSession {session_key} - Filter: {filter_display}")
    print(f"Total trials: {total_trials}")
    print(f"PD outliers: {len(irregular_pd_trials)} ({len(irregular_pd_trials)/total_trials*100:.1f}%)")
    print(f"Excluded trials: {len(excluded_trials)} ({len(excluded_trials)/total_trials*100:.1f}%)")
    print(f"Included trials: {len(included_trials)} ({len(included_trials)/total_trials*100:.1f}%)")
    print(f"STD check range: {T_behavior[start_idx]:.3f}s to {T_behavior[min(end_idx-1, len(T_behavior)-1)]:.3f}s")
    
    # Return the trial categories for further use
    return {
        'included_trials': included_trials,
        'excluded_trials': excluded_trials,
        'irregular_pd': irregular_pd_trials,
        'all_valid_trials': all_valid_trials
    }

def normalize_data(X_data, index_start, index_end,):
    """
    Normalize data over the specified index range.
    
    Parameters:
    -----------
    X_data : np.ndarray
        The input data array with shape (n_time_points, n_features, n_trials).
    index_start : int
        The start index of the subregion used for computing mean and std.
    index_end : int
        The end index of the subregion used for computing mean and std.
    
    Returns:
    --------
    X_normalized : np.ndarray
        The normalized data array with the same shape as X_data.
    """
    # Compute mean and std across time and trials within the specified region
    X_mean = np.mean(X_data[index_start:index_end, :, :], axis=(0, 2))  # Shape: (n_features,)
    X_std = np.std(X_data[index_start:index_end, :, :], axis=(0, 2), ddof=1)  # Shape: (n_features,)
    
    # Avoid division by zero by replacing zeros with ones
    X_std[X_std == 0] = 1

    # Normalize the entire array
    X_normalized = (X_data - X_mean[np.newaxis, :, np.newaxis]) / X_std[np.newaxis, :, np.newaxis]

    return X_normalized


import numpy as np
import plotly.io as pio

def plot_comparison_behavioral_timeseries(
    behavior_data_mouse1,
    behavior_data_mouse2,
    T_behavior,
    session_key_mouse1,
    session_key_mouse2,
    indices_of_interest=None,
    dimension_names=None,
    save_plot=False,
    output_file=None
):
    """
    Plots behavioral time series data from two mice side by side with identical formatting.
    
    Parameters:
        behavior_data_mouse1 (np.ndarray): Behavioral data array for mouse 1
        behavior_data_mouse2 (np.ndarray): Behavioral data array for mouse 2
        T_behavior (np.ndarray): Time array for the x-axis
        session_key_mouse1 (str): Session identifier for mouse 1
        session_key_mouse2 (str): Session identifier for mouse 2
        indices_of_interest (list): Indices of dimensions to plot; default is 0-5 (thumb & pinky)
        dimension_names (list): Custom names for dimensions; if not provided, generic names are used
        save_plot (bool): Whether to save the plot to a file
        output_file (str): Filename to save the plot (if save_plot is True)
    
    Returns:
        fig: Plotly figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import plotly.express as px
    
    # Determine the number of dimensions available per mouse
    _, _, n_dims_mouse1 = behavior_data_mouse1.shape
    _, _, n_dims_mouse2 = behavior_data_mouse2.shape
    
    # Use default indices (and names) if none is provided
    if indices_of_interest is None:
        indices_of_interest = list(range(6))  # Default to first 6 dimensions (thumb & pinky)
        dimension_names = ["Thumb X", "Thumb Y", "Thumb Z", "Pinky X", "Pinky Y", "Pinky Z"]
    if dimension_names is None:
        dimension_names = [f"Dimension {idx}" for idx in indices_of_interest]
    if len(indices_of_interest) != len(dimension_names):
        raise ValueError("indices_of_interest and dimension_names must have the same length")
    
    n_dimensions = len(indices_of_interest)
    
    # Create subplots; 2 columns for mouse1 and mouse2, and one row per dimension
    fig = make_subplots(
        rows=n_dimensions, cols=2,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
        subplot_titles=None
    )
    
    colors = px.colors.qualitative.Plotly
    avg_data_mouse1 = np.mean(behavior_data_mouse1, axis=0)
    avg_data_mouse2 = np.mean(behavior_data_mouse2, axis=0)
    
    for i, dim_idx in enumerate(indices_of_interest):
        row_idx = i + 1
        
        if dim_idx == -1:
            # Special case for grip aperture
            thumb1 = behavior_data_mouse1[:, :, 0:3]
            pinky1 = behavior_data_mouse1[:, :, 3:6]
            all_grip_ap1 = np.linalg.norm(thumb1 - pinky1, axis=2)
            avg_grip_ap1 = np.mean(all_grip_ap1, axis=0)
            
            thumb2 = behavior_data_mouse2[:, :, 0:3]
            pinky2 = behavior_data_mouse2[:, :, 3:6]
            all_grip_ap2 = np.linalg.norm(thumb2 - pinky2, axis=2)
            avg_grip_ap2 = np.mean(all_grip_ap2, axis=0)
            
            y_min1, y_max1 = np.min(all_grip_ap1), np.max(all_grip_ap1)
            padding1 = 0.2 * (y_max1 - y_min1)
            y_range1 = [y_min1 - padding1, y_max1 + padding1]
            
            y_min2, y_max2 = np.min(all_grip_ap2), np.max(all_grip_ap2)
            padding2 = 0.2 * (y_max2 - y_min2)
            y_range2 = [y_min2 - padding2, y_max2 + padding2]
            
            for trial_idx in range(all_grip_ap1.shape[0]):
                color_idx = trial_idx % len(colors)
                fig.add_trace(
                    go.Scatter(
                        x=T_behavior,
                        y=all_grip_ap1[trial_idx, :],
                        mode='lines',
                        line=dict(color=colors[color_idx], width=0.5),
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row_idx, col=1
                )
            fig.add_trace(
                go.Scatter(
                    x=T_behavior,
                    y=avg_grip_ap1,
                    mode='lines',
                    line=dict(color='black', width=3),
                    name="Mouse 1",
                    showlegend=i==0
                ),
                row=row_idx, col=1
            )
            for trial_idx in range(all_grip_ap2.shape[0]):
                color_idx = trial_idx % len(colors)
                fig.add_trace(
                    go.Scatter(
                        x=T_behavior,
                        y=all_grip_ap2[trial_idx, :],
                        mode='lines',
                        line=dict(color=colors[color_idx], width=0.5),
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row_idx, col=2
                )
            fig.add_trace(
                go.Scatter(
                    x=T_behavior,
                    y=avg_grip_ap2,
                    mode='lines',
                    line=dict(color='black', width=3),
                    name="Mouse 2",
                    showlegend=i==0
                ),
                row=row_idx, col=2
            )
            
            fig.update_yaxes(range=y_range1, row=row_idx, col=1)
            fig.update_yaxes(range=y_range2, row=row_idx, col=2)
        
        else:
            if dim_idx >= n_dims_mouse1 or dim_idx >= n_dims_mouse2:
                continue
                
            y_min1, y_max1 = np.min(behavior_data_mouse1[:, :, dim_idx]), np.max(behavior_data_mouse1[:, :, dim_idx])
            padding1 = 0.2 * (y_max1 - y_min1)
            y_range1 = [y_min1 - padding1, y_max1 + padding1]
            
            y_min2, y_max2 = np.min(behavior_data_mouse2[:, :, dim_idx]), np.max(behavior_data_mouse2[:, :, dim_idx])
            padding2 = 0.2 * (y_max2 - y_min2)
            y_range2 = [y_min2 - padding2, y_max2 + padding2]
            
            for trial_idx in range(behavior_data_mouse1.shape[0]):
                color_idx = trial_idx % len(colors)
                fig.add_trace(
                    go.Scatter(
                        x=T_behavior,
                        y=behavior_data_mouse1[trial_idx, :, dim_idx],
                        mode='lines',
                        line=dict(color=colors[color_idx], width=0.5),
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row_idx, col=1
                )
            fig.add_trace(
                go.Scatter(
                    x=T_behavior,
                    y=avg_data_mouse1[:, dim_idx],
                    mode='lines',
                    line=dict(color='black', width=3),
                    name="Mouse 1",
                    showlegend=i==0
                ),
                row=row_idx, col=1
            )
            for trial_idx in range(behavior_data_mouse2.shape[0]):
                color_idx = trial_idx % len(colors)
                fig.add_trace(
                    go.Scatter(
                        x=T_behavior,
                        y=behavior_data_mouse2[trial_idx, :, dim_idx],
                        mode='lines',
                        line=dict(color=colors[color_idx], width=0.5),
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row_idx, col=2
                )
            fig.add_trace(
                go.Scatter(
                    x=T_behavior,
                    y=avg_data_mouse2[:, dim_idx],
                    mode='lines',
                    line=dict(color='black', width=3),
                    name="Mouse 2",
                    showlegend=i==0
                ),
                row=row_idx, col=2
            )
            
            fig.update_yaxes(range=y_range1, row=row_idx, col=1)
            fig.update_yaxes(range=y_range2, row=row_idx, col=2)
        
        # Set left y-axis title to show variable name
        fig.update_yaxes(title_text=dimension_names[i], title_standoff=15, row=row_idx, col=1)
        fig.update_yaxes(title_standoff=15, row=row_idx, col=2)
    
    # Optionally add column titles (you can adjust or uncomment these annotations as needed)
    # For example, add titles for Mouse 1 and Mouse 2 above the plots.
    
    # Update layout with overall settings
    fig.update_layout(
        title=dict(
            text="Behavioral Time Series Comparison",
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        height=200 * n_dimensions,
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1,
            tracegroupgap=0
        ),
        margin=dict(l=120, r=120, t=120, b=120)
    )
    
    # Update every x-axis so that tick labels and gridlines are visible.
    # Only the bottom row will have the title "Time (s)", the rest will have an empty title.
    for row in range(1, n_dimensions+1):
        for col in range(1, 3):
            title_text = "Time (s)" if row == n_dimensions else ""
            fig.update_xaxes(
                title_text=title_text,
                showticklabels=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='gray',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                automargin=True,
                row=row, col=col
            )
    
    # Update all y-axes with compact settings
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='gray',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        automargin=True,
        tickfont=dict(size=10)
    )
    
    # Add vertical lines at t=0 in every subplot
    for row in range(1, n_dimensions + 1):
        for col in range(1, 3):
            fig.add_vline(
                x=0, 
                line_width=1.5, 
                line_dash="dash", 
                line_color="red",
                row=row, col=col
            )
    
    if save_plot and output_file:
# For higher resolution, increase the scale parameter:
        fig.write_image(output_file, scale=2, width=1200, height=200 * n_dimensions)
        
    return fig



def create_time_window_matrix_new(X, T, t0, winsize, return_stats=False, stats=None):
    '''
    ARGS:
        X -> (time, components, trials)
        T -> (time, ) or (trials, time): time array(s)
        t0 -> start time (according to T vector)
        winsize -> (int) number of bins to include going forward
        return_stats -> if True, returns statistics for normalization
        stats -> (stack_mu, stack_std, centering_mu) previously computed stats for applying same normalization

    OUTPUT:
        X_pca_realigned_centered -> (trials, n_behavioral_dims) array
        If return_stats is True, also returns (stack_mu, stack_std, centering_mu).
    '''

    # X is (n_time, n_behavioral_dims, n_trials)
    n_time, n_behavioral_dims, n_trials = X.shape
   # print(X.shape)
    
    # Ensure T is (n_trials, n_time)
    if len(T.shape) == 1:
        T = np.tile(T, (n_trials, 1))

    #print(T)
    # Find the index of t0 for each trial
    idx_t0_across_all_trials = np.argmin(np.abs(T - t0), axis=1)
   # print(t0)
    #print(idx_t0_across_all_trials)
    
    # Check if we successfully found t0 in each trial
    if np.any(np.min(np.abs(T - t0), axis=1) > 1e-2):
        print("failed to find the index of t0={} for at least 1 trial".format(t0))
        raise ValueError

    # Create a boolean mask to select the time window for each trial
    t_idxs = np.full(T.shape, False)
    for n in range(n_trials):
        idx_t0 = idx_t0_across_all_trials[n]
        end_idx = idx_t0 + winsize
        if end_idx > T.shape[1]:
            # If the window goes out of range, truncate at the end
            end_idx = T.shape[1]
        t_idxs[n, idx_t0:end_idx] = True

    # We now want to compute the average over the window for each (trial, dimension)
    # Final shape should be (n_trials, n_behavioral_dims)
    X_avg = np.empty((n_trials, n_behavioral_dims), dtype=np.float32)

    for i in range(n_behavioral_dims):
        for j in range(n_trials):
            # Extract the window data for dimension i in trial j and average
            window_data = X[t_idxs[j], i, j]
            X_avg[j, i] = np.nanmean(window_data) if window_data.size > 0 else np.nan

    # If no stats provided, compute new stats
    if stats is None:
        stack_mu = np.nanmean(X_avg, axis=0, keepdims=True)
        stack_std = np.nanstd(X_avg, axis=0, keepdims=True, ddof=1)
    else:
        stack_mu, stack_std, _ = stats


    # Normalize the data using the computed stats
    X_norm = (X_avg - stack_mu) / stack_std

    """
    # Compute centering_mu (if none provided)
    if stats is None:
        centering_mu = np.nanmean(X_norm, axis=0, keepdims=True)
        #print(centering_mu)
    else:
        centering_mu = stats[2]

    # Center the data
    X_pca_realigned_centered = X_norm - centering_mu
    """
    centering_mu = []
    if not return_stats:
        return X_norm
    else:
        return X_norm, (stack_mu, stack_std, centering_mu)


import pymanopt
from pymanopt.manifolds import Product
from sklearn.decomposition import PCA
from scipy.linalg import polar

def optimize_vector_spaces_former(data, n_components):
    '''
    modified to be more generally useful:
        data -> list of m 2D matrices, each matrix is the output of "create time window matrix", so they are 2 dimensional
        n_components -> list of m integers, each integer corresponds to the number of components dedicated to fitting the mth data matrix in 'data'

    output -> tuple of matrices, the mth matrix having n_components[m] columns of orthonormal vectors
    '''
    #first, eliminate the dimensions in data that don't have any variance (helpful for later)
    non_zero_var_idxs = np.argwhere(np.all([ (np.std(M, axis=0) > 0) for M in data], axis=0)).squeeze()
    data = [M[:, non_zero_var_idxs] for M in data]

    #data = [(M - np.mean(M, axis=0, keepdims=True)) / np.std(M, axis=0, keepdims=True) for M in data]#zscore all the matrices individually across trials
    data_cov = [np.cov(M.T) for M in data]
    data_sv = [np.linalg.svd(C)[1][0:n_components[m]] for m, C in enumerate(data_cov)]#find the first n_components[m] singular values as a maximum explainable variance

    n_behavior_dims = data[0].shape[1]#these should be equivalent for all matrices in data
    ST_m = pymanopt.manifolds.stiefel.Stiefel(n_behavior_dims, np.sum(n_components))

    #find initial starting points based on stiefel projection of PCAs:
    pca_models = [PCA(n) for n in n_components]
    for m, model in enumerate(pca_models):
        model.fit(data[m])

    Q0 = [model.components_.T for model in pca_models]
    Q0 = np.hstack(Q0)#combine these into one rectangulatr matrix

    Q0, _ = polar(Q0)#use polar decomposition to project these onto the steifel manifold.

    def obj_fun(Q_list):
        '''
        general optimization function for vector space Q
        accepts a list of m vector spaces (not the full sum(n_components) orthonormal matrix)
        '''
        explained_variance_ratios = []
        for m in range(len(Q_list)):
            Q_m = Q_list[m]
            C_m = data_cov[m]
            SV_m = data_sv[m]

            explained_variance_ratios.append(np.trace((Q_m.T @ C_m @ Q_m) / np.sum(SV_m)))

        return np.sum(explained_variance_ratios)

    @pymanopt.function.autograd(ST_m)
    def obj_fun_wrapper(Q):
        '''
        assume the input matrix is Q = [Qi | Qj | Qk ...] is a horizontally
        concatenated matrix of orthonormal columns
        wrapper function around the obj_fun defined above.
        '''
        Q_list = []
        idx_0 = 0
        for m in range(len(n_components)):
            Q_list.append(Q[:, idx_0:idx_0 + n_components[m]])
            idx_0 += n_components[m]
        return -1*obj_fun(Q_list)


    problem = pymanopt.Problem(ST_m, obj_fun_wrapper)
    optimizer = pymanopt.optimizers.trust_regions.TrustRegions()
    result = optimizer.run(problem, initial_point=Q0)#this is another horizontally concatenated set of vector spaces.

    #return the parsed result:
    output = tuple()
    idx_0 = 0
    for m in range(len(n_components)):
        output += (result.point[:, idx_0: idx_0 + n_components[m]],)
        idx_0 += n_components[m]

    return output, non_zero_var_idxs


def process_data_for_vector_spaces(X_interp_filtered, T_behavior, tP_0, tL_0,
                                   winsize, dt,  n_behavioral_dims, pre_dims=1):
    """
    Processes data to form pre-lift (P) and post-lift (L) matrices, 
    optimizes vector spaces, and projects the data onto these spaces.

    Parameters
    ----------
    X_interp_filtered : np.ndarray
        Filtered data of shape (n_time_points, n_features, n_trials).
    T_behavior : np.ndarray
        Time points corresponding to behavior.
    tP_0 : float
        Start time of the pre-lift window.
    tL_0 : float
        Start time of the post-lift window.
    winsize : int
        Window size for time window extraction.
    dt : float
        Time step for pre-lift data.
    dtP : float
        Time step for post-lift data (assumed different if provided).
    n_behavioral_dims : int
        Number of behavioral dimensions.
    pre_dims : int, optional
        Number of vectors to use for the pre-lift space (default is 1).

    Returns
    -------
    P : np.ndarray
        Pre-lift data matrix.
    stats_P : dict
        Statistics from the pre-lift data window.
    L : np.ndarray
        Post-lift data matrix.
    stats_L : dict
        Statistics from the post-lift data window.
    Qp : np.ndarray
        Transformation matrix for pre-lift data space.
    Ql : np.ndarray
        Transformation matrix for post-lift data space.
    P_proj : np.ndarray
        Projection of pre-lift data onto its optimized space.
    L_proj : np.ndarray
        Projection of post-lift data onto its optimized space.
    principal_components : np.ndarray
        Concatenation of pre-lift and post-lift projections.
    pre_lift_start : float
        Computed start time for the pre-lift window.
    pre_lift_end : float
        Computed end time for the pre-lift window.
    post_lift_start : float
        Computed start time for the post-lift window.
    post_lift_end : float
        Computed end time for the post-lift window.
    """
    
    # Time window limits
    pre_lift_start = tP_0
    pre_lift_end = tP_0 + winsize * dt
    post_lift_start = tL_0
    post_lift_end = tL_0 + winsize * dt

    # Compile the behavioral data occurring before and after lift
    P, stats_P = create_time_window_matrix_new(X_interp_filtered[:, :n_behavioral_dims, :], 
                                               T_behavior, tP_0, winsize, return_stats=True)
    L, stats_L = create_time_window_matrix_new(X_interp_filtered[:, :n_behavioral_dims, :], 
                                               T_behavior, tL_0, winsize, return_stats=True)

    # Optimize vector spaces to make post-lift space as orthogonal as possible to pre-lift
    (Qp, Ql), nzvi = optimize_vector_spaces_former([P, L], [pre_dims, 1])

    # Print shapes for debugging (as per the original code)
    print(P.shape)
    print(L.shape)
    print(Qp.shape)
    print(Ql.shape)

    # Project the data onto these optimized spaces
    P_proj = P[:, nzvi] @ Qp
    L_proj = L[:, nzvi] @ Ql

    # Combine pre-lift and post-lift projections
    principal_components = np.hstack((P_proj, L_proj))

    # Return all computed variables except nzvi
    return (P, stats_P, L, stats_L,
            Qp, Ql, P_proj, L_proj, principal_components,
            pre_lift_start, pre_lift_end, post_lift_start, post_lift_end)

import os

def plot_and_save_components(
    X_interp_filtered,
    P_proj,
    L_proj,
    T_behavior,
    tP_0,
    tL_0,
    winsize,
    dt,
    movement_onset,
    n_behavioral_dims,
    n_trials,
    mouse,
    session_number,
    list_of_components,
    output_directory=None,
    file_prefix="components",
    dims_to_plot=None,
    dim_names=None,
    dpi=100,
    fig_width=None,
    fig_height=None,
    width_per_component=5,
    height_per_dim=3
):
    """
    Plots pre-lift and post-lift components for behavioral data and saves the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # If no dimensions specified, plot all
    if dims_to_plot is None:
        dims_to_plot = list(range(n_behavioral_dims))
    
    # Generate default dimension names if not provided
    if dim_names is None:
        dim_names = [f"Dimension {i}" for i in dims_to_plot]
    
    # Ensure dim_names matches dims_to_plot in length
    if len(dims_to_plot) != len(dim_names):
        raise ValueError("dims_to_plot and dim_names must have the same length")
    
    n_dims_to_plot = len(dims_to_plot)
    n_components = len(list_of_components)

    # Calculate figure size if not provided
    if fig_width is None:
        fig_width = width_per_component * n_components
    if fig_height is None:
        fig_height = height_per_dim * n_dims_to_plot

    # Time window limits
    pre_lift_start = tP_0
    pre_lift_end = tP_0 + winsize * dt
    post_lift_start = tL_0
    post_lift_end = tL_0 + winsize * dt

    # Prepare data for plotting
    total_projs = np.hstack((P_proj, L_proj))
    T_plotting = [np.tile(T_behavior, (n_trials, 1)) for _ in list_of_components]

    # Create figure with adjusted layout
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create gridspec with proper spacing - less left margin
    gs = fig.add_gridspec(n_dims_to_plot, n_components, 
                          left=0.15,  # Reduced left margin
                          right=0.95,
                          bottom=0.1,
                          top=0.9,
                          wspace=0.3,
                          hspace=0.4)
    
    # Create axes
    ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_components)] 
                  for i in range(n_dims_to_plot)])
    
    # Process labels to handle long ones
    formatted_labels = []
    for name in dim_names:
        if len(name) > 12:
            parts = name.split(' ')
            if len(parts) > 1:
                half = len(parts) // 2
                first_line = ' '.join(parts[:half])
                second_line = ' '.join(parts[half:])
                formatted_labels.append(f"{first_line}\n{second_line}")
            else:
                formatted_labels.append(f"{name[:10]}\n{name[10:]}")
        else:
            formatted_labels.append(name)

    # Loop through each component and selected behavioral dimension
    for n, comp in enumerate(list_of_components):
        pca_colors = np.digitize(total_projs[:, comp], 
                                np.linspace(np.min(total_projs[:, comp]), 
                                           np.max(total_projs[:, comp]), 10)) / 10

        for i, dim_idx in enumerate(dims_to_plot):
            ymin = np.nanmin(X_interp_filtered[:, dim_idx, :])
            ymax = np.nanmax(X_interp_filtered[:, dim_idx, :])
            y_range = ymax - ymin
            y_padding = 0.1 * y_range
            
            # Set y limits with padding
            y_limits = [ymin - y_padding, ymax + y_padding]
            ax[i, n].set_ylim(y_limits)

            for j in range(n_trials):
                ax[i, n].plot(
                    T_plotting[n][j],
                    X_interp_filtered[:, dim_idx, j],
                    color=plt.cm.gist_rainbow(pca_colors[j]),
                    alpha=0.5,
                    linewidth=0.8
                )

            # Add vertical lines for pre-lift and post-lift analysis windows
            ax[i, n].plot([pre_lift_start, pre_lift_start], [ymin, ymax], "r--", alpha=0.8, 
                         label="Pre-lift Start" if i == 0 and n == 0 else None)
            ax[i, n].plot([pre_lift_end, pre_lift_end], [ymin, ymax], "r--", alpha=0.8, 
                         label="Pre-lift End" if i == 0 and n == 0 else None)
            ax[i, n].plot([post_lift_start, post_lift_start], [ymin, ymax], "b--", alpha=0.8, 
                         label="Post-lift Start" if i == 0 and n == 0 else None)
            ax[i, n].plot([post_lift_end, post_lift_end], [ymin, ymax], "b--", alpha=0.8, 
                         label="Post-lift End" if i == 0 and n == 0 else None)

            # Add movement onset vertical line in black
            ax[i, n].plot([movement_onset, movement_onset], [ymin, ymax], "k--", alpha=0.8, 
                         label="Reference point" if i == 0 and n == 0 else None)

            # Adjust x-axis limits
            ax[i, n].set_xlim((-0.3, 0.3))
            
            # Set tick parameters for better readability
            ax[i, n].tick_params(axis='both', which='major', labelsize=8)
            ax[i, n].tick_params(axis='y', pad=5)
            
            # Hide x-axis for all but the last row
            if i != n_dims_to_plot - 1:
                ax[i, n].set_xticklabels([])
            else:
                ax[i, n].set_xlabel("Time (s)", fontsize=12)

    # Position the labels closer to the y-axis
    for i in range(n_dims_to_plot):
        # Use original ylabel method but with precise positioning
        ax[i, 0].set_ylabel(formatted_labels[i], fontsize=10, 
                           rotation='horizontal', 
                           ha='right',  # Right align
                           va='center',  # Center vertically
                           labelpad=10)  # Closer to axis
    
    # Set titles for the first row
    #for n, comp in enumerate(list_of_components):
    #    ax[0, n].set_title(f"Component {comp} Loadings", fontsize=14)

    # Add a legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.98), 
              ncol=5, frameon=False, fontsize=12)

    # Create directory if it doesn't exist
    if output_directory is None:
        output_directory = os.getcwd()
    session_directory = os.path.join(output_directory, f"{mouse}_session_{session_number}")
    os.makedirs(session_directory, exist_ok=True)

    # Save the figure
    output_file = os.path.join(session_directory, f"{file_prefix}_{mouse}_session_{session_number}.png")
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved as {output_file}")

    return output_file


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

def plot_average_rank_correlation(
    X_interp_filtered_list,  # list of session-specific X_interp_filtered
    Qp_list,                 # list of Qp for each session
    Ql_list,                 # list of Ql for each session
    T_behavior_list,         # list of T_behavior arrays (if they differ per session)
    winsize_P_list,          # list of winsize_P for each session
    tP_0_list,               # list of tP_0 for each session (reference pre-lift window start)
    tL_0_list,               # list of tL_0 for each session (reference post-lift window start)
    dt_list,                 # list of dt (time step) for each session
    movement_onset_list,     # list of movement_onset times
    average_max_ga_list,     # list of avg max GA times for each session
    context_window_list,     # list of context_window for each session
    stats_P_list,            # list of stats_P for each session
    stats_L_list,            # list of stats_L for each session
    method="spearman",       # method to compute similarity: "spearman" or "coherence"
    save_plot=False,
    output_file="average_rank_correlation.png",
    dpi=100                  # added DPI parameter with default 100
):
    """
    Computes and plots the average rank correlation across sessions with standard deviation.
    Shows mean and std of trial ordering similarity over time for all sessions combined.
    """
    
    # Create figure with 2 subplots
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=["Average Trial Rank Correlation with Rest Similarity",
                        "Average Trial Rank Correlation with Reach Similarity"]
    )
    
    # Store all session data for averaging
    all_window_centers = []
    all_pre_similarity = []
    all_post_similarity = []
    
    # Process each session
    for i in range(len(X_interp_filtered_list)):
        X_interp_filtered = X_interp_filtered_list[i]
        Qp = Qp_list[i]
        Ql = Ql_list[i]
        T_behavior = T_behavior_list[i]
        winsize_P = winsize_P_list[i]
        tP_0 = tP_0_list[i]
        tL_0 = tL_0_list[i]
        dt = dt_list[i]
        movement_onset = movement_onset_list[i]
        average_max_ga = average_max_ga_list[i]
        context_window = context_window_list[i]
        stats_P = stats_P_list[i]
        stats_L = stats_L_list[i]
        
        # Get the reference window data
        X_ref_pre = create_time_window_matrix_new(
            X_interp_filtered[:, :Qp.shape[0], :],
            T_behavior,
            tP_0,
            winsize_P,
            return_stats=False,
            stats=stats_P
        )
        
        X_ref_post = create_time_window_matrix_new(
            X_interp_filtered[:, :Ql.shape[0], :],
            T_behavior,
            tL_0,
            winsize_P,
            return_stats=False,
            stats=stats_L
        )
        
        # Project reference data onto components
        ref_proj_pre = X_ref_pre @ Qp
        ref_proj_post = X_ref_post @ Ql
        
        # Compute reference trial scores and ranks
        ref_trial_scores_pre = []
        for trial_idx in range(ref_proj_pre.shape[0]):
            if ref_proj_pre.shape[1] > 1:
                score = np.linalg.norm(ref_proj_pre[trial_idx])
            else:
                score = ref_proj_pre[trial_idx, 0]
            ref_trial_scores_pre.append(score)
            
        ref_trial_scores_post = []
        for trial_idx in range(ref_proj_post.shape[0]):
            if ref_proj_post.shape[1] > 1:
                score = np.linalg.norm(ref_proj_post[trial_idx])
            else:
                score = ref_proj_post[trial_idx, 0]
            ref_trial_scores_post.append(score)
            
        ref_trial_ranks_pre = np.argsort(np.argsort(ref_trial_scores_pre))
        ref_trial_ranks_post = np.argsort(np.argsort(ref_trial_scores_post))
        
        # Compute rank correlation for each sliding window
        step_size = 1
        time_points, n_behavioral_dims_full, n_trials = X_interp_filtered.shape
        
        window_centers = []
        pre_similarity = []
        post_similarity = []
        
        for start_idx in range(0, time_points - winsize_P - 1, step_size):
            center_time = T_behavior[start_idx + winsize_P // 2]
            
            # Get current window data
            X_curr_pre = create_time_window_matrix_new(
                X_interp_filtered[:, :Qp.shape[0], :],
                T_behavior,
                T_behavior[start_idx],
                winsize_P,
                return_stats=False,
                stats=stats_P
            )
            
            X_curr_post = create_time_window_matrix_new(
                X_interp_filtered[:, :Ql.shape[0], :],
                T_behavior,
                T_behavior[start_idx],
                winsize_P,
                return_stats=False,
                stats=stats_L
            )
            
            if X_curr_pre.shape[1] != Qp.shape[0] or X_curr_post.shape[1] != Ql.shape[0]:
                continue
                
            curr_proj_pre = X_curr_pre @ Qp
            curr_proj_post = X_curr_post @ Ql
            curr_proj_pre = np.expand_dims(curr_proj_pre[:,0], axis=1)
            
            # Compute current window trial scores and ranks
            curr_trial_scores_pre = []
            for trial_idx in range(curr_proj_pre.shape[0]):
                if curr_proj_pre.shape[1] > 1:
                    score = np.linalg.norm(curr_proj_pre[trial_idx])
                else:
                    score = curr_proj_pre[trial_idx, 0]
                curr_trial_scores_pre.append(score)
                
            curr_trial_scores_post = []
            for trial_idx in range(curr_proj_post.shape[0]):
                if curr_proj_post.shape[1] > 1:
                    score = np.linalg.norm(curr_proj_post[trial_idx])
                else:
                    score = curr_proj_post[trial_idx, 0]
                curr_trial_scores_post.append(score)
                
            curr_trial_ranks_pre = np.argsort(np.argsort(curr_trial_scores_pre))
            curr_trial_ranks_post = np.argsort(np.argsort(curr_trial_scores_post))
            
            try:
                pre_corr, _ = spearmanr(ref_trial_ranks_pre, curr_trial_ranks_pre)
                post_corr, _ = spearmanr(ref_trial_ranks_post, curr_trial_ranks_post)
                
                window_centers.append(center_time)
                pre_similarity.append(pre_corr)
                post_similarity.append(post_corr)
            except Exception as e:
                continue
        
        # Store session data
        all_window_centers.append(window_centers)
        all_pre_similarity.append(pre_similarity)
        all_post_similarity.append(post_similarity)
    
    # Find common time points across all sessions
    min_time = max([min(centers) for centers in all_window_centers])
    max_time = min([max(centers) for centers in all_window_centers])
    
    # Create common time grid
    common_times = np.linspace(min_time, max_time, 100)
    
    # Interpolate each session's data to common time points
    interpolated_pre = []
    interpolated_post = []
    
    for i in range(len(all_window_centers)):
        if len(all_window_centers[i]) > 0:
            pre_interp = np.interp(common_times, all_window_centers[i], all_pre_similarity[i])
            post_interp = np.interp(common_times, all_window_centers[i], all_post_similarity[i])
            interpolated_pre.append(pre_interp)
            interpolated_post.append(post_interp)
    
    # Compute mean and std across sessions
    mean_pre = np.mean(interpolated_pre, axis=0)
    std_pre = np.std(interpolated_pre, axis=0)
    mean_post = np.mean(interpolated_post, axis=0)
    std_post = np.std(interpolated_post, axis=0)
    
    # Plot pre-lift similarity with std
    fig.add_trace(
        go.Scatter(
            x=common_times,
            y=mean_pre,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Mean Rest Similarity'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([common_times, common_times[::-1]]),
            y=np.concatenate([mean_pre + std_pre, (mean_pre - std_pre)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 std',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Plot post-lift similarity with std
    fig.add_trace(
        go.Scatter(
            x=common_times,
            y=mean_post,
            mode='lines',
            line=dict(color='red', width=2),
            name='Mean Reach Similarity'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([common_times, common_times[::-1]]),
            y=np.concatenate([mean_post + std_post, (mean_post - std_post)[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 std',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add vertical lines for average reference windows without annotations
    avg_tP_0 = np.mean(tP_0_list)
    avg_tL_0 = np.mean(tL_0_list)
    avg_winsize_P = np.mean([w * d for w, d in zip(winsize_P_list, dt_list)])
    
    # Pre-lift reference window vertical lines (no annotations)
    fig.add_vline(
        x=avg_tP_0,
        line_dash="dash",
        line_color="black",
        line_width=2,
        row=1,
        col=1
    )
    fig.add_vline(
        x=avg_tP_0 + avg_winsize_P,
        line_dash="dash",
        line_color="black",
        line_width=2,
        row=1,
        col=1
    )
    
    # Post-lift reference window vertical lines (no annotations)
    fig.add_vline(
        x=avg_tL_0,
        line_dash="dash",
        line_color="black",
        line_width=2,
        row=2,
        col=1
    )
    fig.add_vline(
        x=avg_tL_0 + avg_winsize_P,
        line_dash="dash",
        line_color="black",
        line_width=2,
        row=2,
        col=1
    )
    
    # Add max grip aperture marker (as black circle)
    avg_max_ga = np.mean(average_max_ga_list)

    # Find the y-values at the max grip aperture time
    closest_idx_pre = np.argmin(np.abs(common_times - avg_max_ga))
    closest_idx_post = closest_idx_pre  # Same index for post

    # Get the corresponding y-values
    y_pre = mean_pre[closest_idx_pre]
    y_post = mean_post[closest_idx_post]

    # Add circle markers at the correlation values
    fig.add_trace(
        go.Scatter(
            x=[avg_max_ga],
            y=[y_pre],
            mode='markers',
            marker=dict(
                symbol='circle', 
                size=12,
                color='black',
                line=dict(color='black', width=1)
            ),
            name='Max Grip Aperture',
            showlegend=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[avg_max_ga],
            y=[y_post],
            mode='markers',
            marker=dict(
                symbol='circle', 
                size=12,
                color='black',
                line=dict(color='black', width=1)
            ),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add movement onset marker (as rectangle)
    avg_movement_onset = np.mean(movement_onset_list)
    
    # Find the y-values at the movement onset time
    movement_onset_idx_pre = np.argmin(np.abs(common_times - avg_movement_onset))
    movement_onset_idx_post = movement_onset_idx_pre  # Same index for post
    
    # Get the corresponding y-values
    movement_onset_y_pre = mean_pre[movement_onset_idx_pre]
    movement_onset_y_post = mean_post[movement_onset_idx_post]
    
    # Add rectangle markers at the correlation values
    fig.add_trace(
        go.Scatter(
            x=[avg_movement_onset],
            y=[movement_onset_y_pre],
            mode='markers',
            marker=dict(
                symbol='square', 
                size=12,
                color='black',
                line=dict(color='black', width=1)
            ),
            name='Movement Onset',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[avg_movement_onset],
            y=[movement_onset_y_post],
            mode='markers',
            marker=dict(
                symbol='square', 
                size=12,
                color='black',
                line=dict(color='black', width=1)
            ),
            showlegend=False
        ),
        row=2, col=1
    )

    # Update layout with white background and black axes
    fig.update_layout(
        height=800,
        margin=dict(l=100, r=100, t=100, b=100),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="center",
            x=0.5,
            y=-0.2
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(
        title_text="Time (s)",
        showline=True,
        linecolor='black',
        linewidth=2,
        gridcolor='lightgrey',
        mirror=True
    )
    
    fig.update_yaxes(
        title_text="Trial Rank Correlation",
        range=[-1.1, 1.1],
        showline=True,
        linecolor='black',
        linewidth=2,
        gridcolor='lightgrey',
        mirror=True
    )
    
    if save_plot:
        # Use scale parameter to set DPI (scale = dpi/100)
        fig.write_image(output_file, scale=dpi/100)
    
    fig.show()
    
    return fig


def extract_data_around_movement_onset(data, vicinity=300, behavioral_keys=['digit1', 'digit3', 'hand'], positions=['digit1', 'digit3', 'hand']):
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
