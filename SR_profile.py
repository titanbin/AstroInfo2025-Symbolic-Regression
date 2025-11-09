# =============================================================================
# Symbolic Regression Analysis for Globular Cluster Evolution
# =============================================================================
"""
This script analyzes N-body simulation data using Physics-Informed Symbolic Optimization
(PhySO) to discover analytical expressions for globular cluster dynamics. The analysis focuses on temporal evolution of stellar kinematics
and aims to find physically meaningful mathematical relationships.

Key features:
- Processes multiple time snapshots from N-body simulations
- Performs dimensional analysis-constrained symbolic regression
- Handles data normalization and weighting
- Implements parallel processing for HPC environments
- Generates visualizations of fitted expressions and parameter evolution

Author: Soorya Narayan (for AstroInfo2025)
Date: November 2025
"""

# External package imports
import sys
# Add custom package paths for HPC environment - adjust paths as needed
sys.path.append('/pbs/home/a/astroinfo06/.local/lib/python3.11/site-packages')
sys.path.append('/pbs/home/a/astroinfo06/.local/lib/python3.9/site-packages')

# Standard libraries
import datetime
import os
import copy
import multiprocessing

# Scientific computing packages
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Symbolic mathematics and optimization
from sympy import lambdify
import physo  # Physics-Informed Symbolic Optimization
import physo.learn.monitoring as monitoring
from physo.benchmark.utils import symbolic_utils as su

# Force CPU-only mode to avoid CUDA/MPS conflicts
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU access

# Configure parallel processing environment
# Limit threads to avoid conflicts in parallel processing
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS threading
os.environ['MKL_NUM_THREADS'] = '1'       # Intel MKL threading
os.environ['OMP_NUM_THREADS'] = '1'       # OpenMP threading
torch.set_num_threads(1)                  # PyTorch threading

# Configure multiprocessing
multiprocessing.set_start_method('fork', force=True)  # Use 'spawn' on macOS if needed

# Data path configuration
base_path = '/pbs/throng/training/astroinfo2025/data/Nbody/'  # Adjust as needed

models=['1.5M_A_R4_10',
'500k_A_R2_10',
'500k_A_R4_10',
'500k_C_R4_10',
'250k_A_R2_25',
'250k_A_R2_25_vlk',
'250k_A_R2_10',
'250k_A_R2_5',
'250k_A_R4_25',
'250k_A_R4_25_imf50',
'250k_A_R4_25_lk',
'250k_A_R4_25_retr',
'250k_A_R4_25_vlk',
'250k_A_R4_10',
'250k_A_R4_10_retr',
'250k_B_R4_25',
'250k_B_R4_25_lk',
'250k_C_R2_10',
'250k_C_R4_25',
'250k_C_R4_25_lk',
'250k_C_R4_10',
'250k_W6_R4_25',
'250k_W6_R4_25_retr',
'500k_A_R4_LC_part1',
'500k_A_R4_LC_part2']
print('Models defined...')

# Function to populate the smallest and largest radial bins with synthetic data
def populate_boundaries(r_t0, v_phi_t0, num_points=20, split_ratio=0.5, num_bins=100, 
                     sigma_multiplier=1, inner_boundary=5, outer_boundary=40):
    """
    Augment dataset by adding synthetic data points at radial boundaries.
    
    This function improves the symbolic regression's performance at the inner and
    outer radial boundaries by adding statistically consistent synthetic data points.
    It calculates the local statistics (mean, std) in radial bins and generates
    new points following these distributions.
    
    The synthetic points help stabilize the regression at the boundaries where
    data might be sparse or noisy, leading to more robust analytical expressions.
    
    Parameters
    ----------
    r_t0 : np.ndarray
        Radial positions at time t0.
    v_phi_t0 : np.ndarray
        Azimuthal velocities at time t0.
    num_points : int, optional
        Total synthetic points to add (default: 20).
    split_ratio : float, optional
        Fraction of points to add at inner boundary (default: 0.5).
    num_bins : int, optional
        Number of radial bins for statistics (default: 100).
    sigma_multiplier : float, optional
        Factor to scale the sampling standard deviation (default: 1).
    inner_boundary : int, optional
        Number of inner bins to augment (default: 5).
    outer_boundary : int, optional
        Number of outer bins to augment (default: 40).
        
    Returns
    -------
    tuple
        (r_t0_augmented, v_phi_t0_augmented): Arrays with added synthetic points.
    
    Notes
    -----
    - Points are sampled from normal distributions based on local statistics
    - The inner and outer regions are treated separately to maintain appropriate
      physical behavior at the boundaries
    - The number of points is split between inner and outer regions based on
      the split_ratio parameter
    """
    # Bin points in radial bins
    hs, bins = np.histogram(r_t0, bins=num_bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_indices = np.digitize(r_t0, bins) - 1  # Get bin indices for each r_t0 point
    
    # calculate v_phi mean and std in each bin 
    v_phi_means = np.array([v_phi_t0[bin_indices == i].mean() if np.any(bin_indices == i) else 0 for i in range(len(bins)-1)])
    v_phi_stds = np.array([v_phi_t0[bin_indices == i].std() if np.any(bin_indices == i) else 0 for i in range(len(bins)-1)])
    # calculate r mean and std in each bin
    r_means = np.array([r_t0[bin_indices == i].mean() if np.any(bin_indices == i) else 0 for i in range(len(bins)-1)])
    r_stds = np.array([r_t0[bin_indices == i].std() if np.any(bin_indices == i) else 0 for i in range(len(bins)-1)])    
    
    # Sample points from normal distribution at the two extremes
    num_points_inner = int(num_points * split_ratio)
    num_points_outer = num_points - num_points_inner
    print('num_points_inner:', num_points_inner)
    print('num_points_outer:', num_points_outer)

    r_inner_samples = np.array([])
    v_phi_inner_samples = np.array([])
    r_outer_samples = np.array([])
    v_phi_outer_samples = np.array([])
    for i in range(inner_boundary):
        if r_stds[i] == 0:
            r_stds[i] = 0.01 * r_means[i]
        if v_phi_stds[i] == 0:
            v_phi_stds[i] = 0.01 * v_phi_means[i]
        r_inner_samples = np.concatenate([r_inner_samples, np.random.normal(loc=r_means[i], scale=sigma_multiplier*r_stds[i], size=num_points_inner//inner_boundary)])
        v_phi_inner_samples = np.concatenate([v_phi_inner_samples, np.random.normal(loc=v_phi_means[i], scale=sigma_multiplier*v_phi_stds[i], size=num_points_inner//inner_boundary)])

    for i in range(outer_boundary):
        if r_stds[-(i+1)] == 0:
            r_stds[-(i+1)] = 0.01 * r_means[-(i+1)]
        if v_phi_stds[-(i+1)] == 0:
            v_phi_stds[-(i+1)] = 0.01 * v_phi_means[-(i+1)]
        r_outer_samples = np.concatenate([r_outer_samples, np.random.normal(loc=r_means[-(i+1)], scale=sigma_multiplier*r_stds[-(i+1)], size=num_points_outer//outer_boundary)])
        v_phi_outer_samples = np.concatenate([v_phi_outer_samples, np.random.normal(loc=v_phi_means[-(i+1)], scale=sigma_multiplier*v_phi_stds[-(i+1)], size=num_points_outer//outer_boundary)])

    # Combine the new samples with the original data
    r_t0_augmented = np.concatenate([r_t0, r_inner_samples, r_outer_samples])
    v_phi_t0_augmented = np.concatenate([v_phi_t0, v_phi_inner_samples, v_phi_outer_samples])

    return r_t0_augmented, v_phi_t0_augmented

# Function to weight the data points based on radial position
def get_y_weights(X_multi, Y_multi, type='linear'):
    """
    Calculate position-dependent weights for regression data points.
    
    Implements various weighting schemes to control the influence of data points
    based on their radial position. This helps guide the symbolic regression
    to focus on specific regions of interest in the rotation curve.
    
    Parameters
    ----------
    X_multi : np.ndarray
        Array of shape (n_snapshots, n_points) containing radial positions
    Y_multi : np.ndarray
        Array of shape (n_snapshots, n_points) containing velocity values
    type : str, optional
        Weighting scheme to use (default: 'linear')
        Options:
        - 'linear': Linear decay with radius
        - 'sinusoidal': Sine wave pattern
        - 'tanh': Hyperbolic tangent transition
        - 'uniform': Equal weights
    
    Returns
    -------
    np.ndarray
        Weights array of same shape as input, range [1e-3, 1]
    
    Notes
    -----
    Weighting schemes:
    - Linear: Decreases linearly with radius
    - Sinusoidal: Oscillating pattern for periodic emphasis
    - Tanh: Smooth transition between regions
    - Uniform: Equal weighting (for testing)
    """
    def get_sinusoidal_weights(X):
        return (np.sin(2*np.pi*X + np.pi/2) + 1.1) / 2.1
    
    def get_linear_weights(X):
        return np.clip(1 - X, 1e-3, 1)
    
    def get_uniform_weights(X):
        return np.ones_like(X)
    
    def get_tanh_weights(X):
        weights = np.tanh(10*(X-.45)+1) + (np.tanh(10*(-X)+1))
        weights -= np.nanmin(weights)
        weights /= np.nanmax(weights)
        weights += .1
        weights /= 1.1
        return weights

    Y_weights = np.full_like(Y_multi,1)
    if type == 'sinusoidal':
        weights = get_sinusoidal_weights
    elif type == 'linear':
        weights = get_linear_weights
    elif type == 'uniform':
        weights = get_uniform_weights
    elif type == 'tanh':
        weights = get_tanh_weights
    else:
        raise ValueError(f"Unknown weight type: {type}")
    
    for i in range(len(X_multi)):
        # Apply sinusoidal weighting: shifted and scaled to range [1e-3, 1]
        Y_weights[i] = weights(X_multi[i,:])
    Y_weights = np.clip(Y_weights, 1e-3, 1)
    return Y_weights

# Load model data function
def load_model_data_radial(model_name, base_path):
    """
    Load galaxy simulation data from N-body simulation output files.
    
    Reads various physical quantities from the simulation output:
    - Radial positions (r)
    - Azimuthal velocity (v_phi)
    - Velocity dispersion (disp_phi)
    - Stellar ages
    
    Args:
        model_name: String identifier for the simulation model
        base_path: Root directory containing simulation data
        
    Returns:
        dict: Dictionary containing arrays for age, radius, velocity, and dispersion
            - age: 1D array of stellar population ages
            - r: 2D array of radial positions over time
            - disp_phi: 2D array of velocity dispersions
            - v_phi: 2D array of rotation velocities
    """
    # Read radial binning data
    rad = pd.read_csv(base_path + model_name + "/radial/0.0-200.0_radial_bin.dat", delimiter=" ", header=None)
    r = rad.to_numpy()  # shape: (N_timesteps, N_radial_bins)
    
    # Read azimuthal velocity dispersion
    disp_phi = pd.read_csv(base_path + model_name +"/radial/0.0-200.0_disp_phi.dat", delimiter=" ", header=None)
    s_phi = disp_phi.to_numpy()

    # Read rotation velocity
    vel_phi = pd.read_csv(base_path + model_name +"/radial/0.0-200.0_vphi.dat", delimiter=" ", header=None)
    v_phi = vel_phi.to_numpy()
    
    # Read stellar population ages
    age = pd.read_csv(base_path + model_name + "/age.dat",
                      sep=r"\s+", header=None).to_numpy().flatten()
    
    return {
        "age": age,
        "r": r,
        "disp_phi": s_phi,
        "v_phi": v_phi
    }

# Function to run symbolic regression analysis
def run_sr_analysis(X, Y, Y_weights, run_logger=None, run_visualiser=None):
    """
    Execute the symbolic regression analysis using PhySO.
    
    Performs physics-informed symbolic regression to find analytical expressions
    that describe the rotation curve data. Uses dimensional analysis constraints
    and custom operators to ensure physically meaningful results.
    
    Args:
        X: List of arrays containing radial positions (normalized)
        Y: List of arrays containing velocities (normalized)
        Y_weights: Weights for each data point
        run_logger: Optional logger for tracking the optimization
        run_visualiser: Optional visualizer for monitoring progress
        
    Returns:
        tuple: (expression, logs)
            - expression: The best symbolic expression found
            - logs: Training history and optimization metadata
    """
    # Configure parallel processing using SLURM environment variables
    parallel_mode = True
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    if n_cpus == 1:
        n_cpus = int(os.environ.get('SLURM_NTASKS', 1))
    if n_cpus == 1:
        parallel_mode = False

    print(f"Running SR with {n_cpus} CPUs")
    
    # Running SR task
    expression, logs = physo.ClassSR(X, Y, Y_weights, epochs=100, X_units=[[1,0,0]], X_names=["r"], 
                               y_units=[1,-1,0], y_name='vr', 
                               class_free_consts_units=[[0,0,1], [1,0,0], [1,-1,0], [0,1,0]],
                               class_free_consts_names=['M', 'rs', 'vc', 'T'],
                               fixed_consts_units=[[3,-2,-1]], 
                               fixed_consts=[6.67430*1e-11],
                               spe_free_consts_names = [ "k0"      , "k1"        ],
                               spe_free_consts_units = [ [0, 0, 0] , [0, 1, 0]   ],
                               op_names=['mul', 'add', 'sub','div', 'inv', 'n2', 'sqrt', 'neg', 'exp', 'log'],
                               run_config = physo.config.config1b.config1b,
                               parallel_mode=parallel_mode,
                               n_cpus=n_cpus, 
                               get_run_logger=run_logger,
                               get_run_visualiser=run_visualiser)
    return expression, logs

# Function to plot results
def plot_results(best_expr, X, Y, age, num_variables=6, num_subplots=6):
    """
    Visualize the symbolic regression results.
    
    Creates a comprehensive visualization with multiple plots:
    1. A parameter evolution plot showing how parameters vary across classes
    2. A set of 6 subplots, each containing 5 fitted curves
    3. Original data points overlaid with fitted expressions
    4. Color-coded representation for different stellar populations
    
    The visualization includes:
    - Parameter value plots for each symbolic constant
    - Fitted curves with their corresponding data points
    - Class-specific parameter values (k0, t) in legends
    - Vertical offset between curves for better visibility
    
    Args:
        best_expr: The best symbolic expression found by PhySO
        X: Original radial position data (normalized)
        Y: Original velocity data (normalized)
        age: Ages of the stellar populations
    """
    # Prepare data for plotting
    X = X.squeeze()  # Remove singleton dimensions, shape (T, N_points)
    
    # Count total number of classes/expressions found
    num_classes = len(best_expr.get_infix_sympy(evaluate_consts=True))
    plt.figure(figsize=(15, 10))

    # Create colormap for visual distinction between classes
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
    
    # === First Plot: Parameter Evolution Across Classes ===
    # Get dictionaries containing parameter values for each class
    dicts = best_expr.get_sympy_local_dicts(replace_nan_with=False)
    
    # Create a 2x3 grid of subplots for parameter visualization
    if len(list(dicts[0].keys())) > 6 and num_variables <= 6:
        print("Warning: More than 6 parameters to plot, only first 6 will be shown. Change num_variables to see more.")
        num_variables = 6
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    # Extract and plot evolution of each parameter
    keys = dicts[0].keys()
    for i in range(min(len(list(keys)), num_variables)):
        # Collect parameter values across all classes
        data = []
        for item in dicts:
            data.append(item[list(keys)[i]])
            
        # Create parameter evolution plot
        axs[i].set_title(f'Parameter: {list(keys)[i]}')
        axs[i].set_xlabel('Class Index')
        axs[i].set_ylabel('Parameter Value')
        axs[i].plot(data)
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('Parameter_values_all.jpg')
    plt.close()

    # === Second Plot: Fitted Curves and Data Points ===
    # Configure the grid layout for displaying multiple curves
    n_subplots = num_subplots  # Total number of subplots
    curves_per_subplot = num_classes // n_subplots  # Greatest Interger Value per plot
    n_curves = n_subplots * curves_per_subplot  # Total curves to display
    print(f'Plotting {n_curves} curves in total ({curves_per_subplot} per subplot over {n_subplots} subplots).')
    
    # Ensure we don't try to plot more curves than we have classes
    n_available = min(num_classes, n_curves)

    # Create a color palette for all curves
    colors_all = plt.cm.viridis(np.linspace(0, 1, max(1, n_curves)))

    # Extract specialized parameter values for all classes
    spe = best_expr.free_consts.spe_values[0]

    # Get all symbolic expressions for efficiency
    sympy_exprs = best_expr.get_infix_sympy(evaluate_consts=True)

    # Create evaluation points for smooth curve plotting
    x_plot = np.linspace(0, 1, 1000)  # 1000 points for smooth curves

    # Create subplot grid for curve visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # Convert 2D array of axes to 1D for easier indexing

    # Iterate through subplots and curves
    for subplot_idx in range(n_subplots):
        ax = axes[subplot_idx]
        # Plot multiple curves in each subplot with vertical offsets
        for line_idx in range(curves_per_subplot):
            global_idx = subplot_idx * curves_per_subplot + line_idx
            if global_idx >= n_available:
                break

            #print(sympy_exprs)
            sympy_expr = sympy_exprs[global_idx]

            # extract parameters robustly
            k0 = None
            t = None
            try:
                if spe.ndim == 1:
                    # shape (n_classes,) or (n_params,) fallback
                    k0 = float(spe[global_idx])
                elif spe.ndim == 2:
                    # try (n_params, n_classes)
                    k0 = float(spe[0, global_idx])
                    if spe.shape[0] > 1:
                        t = float(spe[1, global_idx])
                else:
                    # fallback: try indexing first axis by class
                    k0 = float(spe[global_idx].flat[0])
            except Exception:
                # silent fallback
                k0 = None
                t = None

            # build numpy function from sympy expression
            variables = list(sympy_expr.free_symbols)
            f = lambdify(variables, sympy_expr, 'numpy')

            # Evaluate function: if single variable, pass x_plot; if multiple, pass zeros for others
            try:
                if len(variables) == 0:
                    y_plot = np.full_like(x_plot, float(sympy_expr.evalf()))
                elif len(variables) == 1:
                    y_plot = f(x_plot)
                else:
                    # pass x_plot for first var, zeros for remaining
                    other_args = [np.zeros_like(x_plot)] * (len(variables) - 1)
                    y_plot = f(x_plot, *other_args)
            except Exception:
                # if evaluation fails, skip this curve
                continue

            # offset each curve slightly for visibility within a subplot
            offset = line_idx * 0.05
            label = f'Age {age[global_idx]:.2f}'
            if k0 is not None and t is not None:
                label += f' k0={k0:.2f}, t={t:.2f}'
            elif k0 is not None:
                label += f' k0={k0:.2f}'

            ax.plot(x_plot, y_plot + offset, label=label, color=colors_all[global_idx], alpha=0.9)
            ax.scatter(X[global_idx], Y[global_idx] + offset, color=colors_all[global_idx], s=3)
        ax.set_title(f'Plot {subplot_idx + 1}')
        ax.grid(True)
        ax.legend(fontsize='small', loc='upper right')

    plt.suptitle('SymReg Fitted Curves with Data Points', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('SR_fitted_curves_all.jpg')
    plt.close()

# Function to prepare data for symbolic regression
def prepare_data(choice_of_model=0, start_snapshot=1750, frequency=100):
    """
    Prepare the simulation data for symbolic regression analysis.
    
    This function performs several key data processing steps:
    1. Loads raw simulation data
    2. Selects specific time snapshots for analysis
    3. Normalizes the data to [0,1] range
    4. Removes any NaN values
    5. Applies weighting scheme
    6. Formats data for PhySO input
    
    Returns:
        tuple: (X, Y, Y_weights, age_selec)
            - X: List of normalized radial positions
            - Y: List of normalized velocities
            - Y_weights: Computed weights for each point
            - age_selec: Ages for selected snapshots
    """
    # Load simulation data
    print(f'Chosen model {models[choice_of_model]}')
    data_1 = load_model_data_radial(models[choice_of_model], base_path)
    r = data_1["r"]
    v_phi = data_1["v_phi"]
    age = data_1["age"].flatten()
    
    # Select every 100th snapshot
    selected_indices = range(start_snapshot, len(r), frequency)
    r_selec = np.stack([r[idx] for idx in selected_indices])
    v_phi_selec = np.stack([v_phi[idx] for idx in selected_indices])
    age_selec = age[selected_indices]
    
    # Normalize data
    r_norm = (r_selec - np.nanmin(r_selec)) / (np.nanmax(r_selec) - np.nanmin(r_selec))
    v_phi_norm = (v_phi_selec - np.nanmin(v_phi_selec)) / (np.nanmax(v_phi_selec) - np.nanmin(v_phi_selec))
    
    X = np.array(r_norm)
    Y = np.array(v_phi_norm)
    
    # Remove NaN values
    indices = np.where(np.isnan(X))[1]
    X = np.delete(X, indices, axis=1)
    Y = np.delete(Y, indices, axis=1)
    Y_weights = get_y_weights(X, Y)
    
    X = X[:, np.newaxis, :]
    X = X.tolist()
    Y = Y.tolist()
    
    return X, Y, Y_weights, age_selec



if __name__ == '__main__':
    # Set random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_path_training_curves = 'MultiT_classSR_DefaultConfig.png'
    save_path_log             = 'MultiT_classSR_DefaultConfig.log'

    run_logger     = lambda : monitoring.RunLogger(save_path = save_path_log,
                                                    do_save = True)

    run_visualiser = lambda : monitoring.RunVisualiser (epoch_refresh_rate = 1,
                                            save_path = save_path_training_curves,
                                            do_show   = False,
                                            do_prints = True,
                                            do_save   = True, )
        
    # Prepare data only once in the main process
    print('Preparing data...')
    X, Y, Y_weights, age_selec = prepare_data(choice_of_model=0, start_snapshot=1750, frequency=100)
    
    # Run SR analysis
    print('Starting SR analysis...')
    expression, logs = run_sr_analysis(X, Y, Y_weights, run_logger, run_visualiser)
    
    # Plot results
    plot_results(expression, X, Y, age_selec, num_variables=6, num_subplots=6)