import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert
import pandas as pd
from functools import partial
import holoviews as hv
hv.extension('matplotlib')

def calculate_order_parameter(phases):
    """Calculate Kuramoto order parameter r(t)
    
    r(t) measures global synchronization, ranging from 0 (no sync) to 1 (perfect sync)
    """
    # r(t) = |1/N * sum_j e^(i*θ_j(t))|
    complex_phases = np.exp(1j * phases)
    return np.abs(np.mean(complex_phases, axis=0))

def calculate_band_order_parameters(time_series):
    """Calculate order parameters for each frequency band"""
    # Group oscillators by band
    band_groups = time_series.groupby('band')
    
    # Calculate order parameter for each band
    order_params = {}
    for band, band_data in band_groups:
        order_params[band] = calculate_order_parameter(band_data.values)
    
    # Create an xarray dataset
    return xr.Dataset({
        band: xr.DataArray(order, dims=['time']) 
        for band, order in order_params.items()
    })

def calculate_phase_coherence(time_series):
    """Calculate pairwise phase coherence matrix between oscillators"""
    n_oscillators = len(time_series.oscillator)
    coherence = np.zeros((n_oscillators, n_oscillators))
    
    phases = time_series.values
    
    # Calculate mean phase coherence for each pair
    for i in range(n_oscillators):
        for j in range(i+1, n_oscillators):
            # Mean phase coherence: |<e^(i*Δθ)>|
            phase_diff = phases[i, :] - phases[j, :]
            coherence_val = np.abs(np.mean(np.exp(1j * phase_diff)))
            coherence[i, j] = coherence_val
            coherence[j, i] = coherence_val  # Symmetric
    
    # Get the actual band values as a numpy array
    oscillator_bands = time_series.band.values
    
    # Convert to xarray with the same coordinates as the coupling matrix
    return xr.DataArray(
        coherence,
        dims=['source', 'target'],
        coords={
            'source': time_series.oscillator.values,
            'source_band': ('source', oscillator_bands),
            'target': time_series.oscillator.values,
            'target_band': ('target', oscillator_bands)
        }
    )

def calculate_band_coherence(coherence_matrix: xr.DataArray):
    """Calculate average coherence between frequency bands"""
    # Get unique band values
    source_bands = np.unique(coherence_matrix.source_band.values)
    target_bands = np.unique(coherence_matrix.target_band.values)
    
    # Create an empty matrix to store results
    band_coherence = np.zeros((len(source_bands), len(target_bands)))
    
    # Calculate mean coherence for each band pair
    for i, source_band in enumerate(source_bands):
        for j, target_band in enumerate(target_bands):
            # Select values for this band pair
            mask = (coherence_matrix.source_band == source_band) & (coherence_matrix.target_band == target_band)
            values = coherence_matrix.where(mask, drop=True)
            band_coherence[i, j] = values.mean().values
    
    # Convert to pandas DataFrame for easier display
    df = pd.DataFrame(band_coherence, index=source_bands, columns=target_bands)
    
    return df

def plot_order_parameters(band_orders):
    """Plot order parameter time series for each band"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for band in band_orders.data_vars:
        ax.plot(band_orders[band].values, label=f"{band}")
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('Order parameter (r)')
    ax.set_title('Synchronization by frequency band')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_phase_distribution(time_series, time_idx=-1):
    """Plot the distribution of phases at a specific time point"""
    # Get phases at the specified time point
    phases = time_series.isel(time=time_idx)
    
    # Create a polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Group by band
    for band in np.unique(phases.band.values):
        band_phases = phases.where(phases.band == band, drop=True)
        # Plot each oscillator as a point on the unit circle
        ax.scatter(
            band_phases.values,  # Phases (angles)
            np.ones(len(band_phases)),  # Radius = 1
            label=band,
            alpha=0.7,
            s=100
        )
    
    ax.set_rticks([])  # Hide radial ticks
    ax.set_title('Phase distribution by band')
    ax.legend(loc='center')
    
    return fig

def plot_coherence_matrix(coherence_matrix):
    """Plot the phase coherence matrix as a heatmap"""
    # Sort by band for better visualization
    sorted_idx = np.lexsort((
        coherence_matrix.source.values, 
        coherence_matrix.source_band.values
    ))
    sorted_coherence = coherence_matrix.isel(
        source=sorted_idx, 
        target=sorted_idx
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sorted_coherence.values, cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Phase coherence')
    
    # Add band separators
    bands = sorted_coherence.source_band.values
    changes = np.where(bands[:-1] != bands[1:])[0]
    
    for change in changes:
        ax.axhline(y=change + 0.5, color='white', linestyle='-', linewidth=1)
        ax.axvline(x=change + 0.5, color='white', linestyle='-', linewidth=1)
    
    # Add band labels
    band_positions = np.concatenate([[0], changes, [len(bands)-1]])
    band_midpoints = [(band_positions[i] + band_positions[i+1]) // 2 
                     for i in range(len(band_positions)-1)]
    
    unique_bands = [bands[pos] for pos in band_positions[:-1]]
    
    ax.set_xticks(band_midpoints)
    ax.set_xticklabels(unique_bands, rotation=45)
    ax.set_yticks(band_midpoints)
    ax.set_yticklabels(unique_bands)
    
    ax.set_title('Phase coherence matrix')
    
    return fig

def plot_band_coherence_heatmap(band_coherence):
    """Plot the inter-band coherence as a heatmap"""
    coherence_values = band_coherence.values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(coherence_values, cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean phase coherence')
    
    # Add labels
    ax.set_xticks(np.arange(len(band_coherence.columns)))
    ax.set_yticks(np.arange(len(band_coherence.index)))
    ax.set_xticklabels(band_coherence.columns, rotation=45)
    ax.set_yticklabels(band_coherence.index)
    
    # Add text annotations
    for i in range(len(band_coherence.index)):
        for j in range(len(band_coherence.columns)):
            ax.text(j, i, f"{coherence_values[i, j]:.2f}",
                    ha="center", va="center", color="white" if coherence_values[i, j] < 0.5 else "black")
    
    ax.set_xlabel('Target band')
    ax.set_ylabel('Source band')
    ax.set_title('Inter-band coherence')
    
    return fig

def analyze_frequency_spectrum(time_series):
    """Analyze the frequency spectrum of oscillators using Hilbert transform"""
    # Apply Hilbert transform to get instantaneous frequencies
    analytic_signal = xr.apply_ufunc(
        lambda x: hilbert(x, axis=-1),
        time_series,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized'
    )
    
    # Unwrap phase to avoid phase jumps
    unwrapped_phase = xr.apply_ufunc(
        lambda x: np.unwrap(np.angle(x)),
        analytic_signal,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized'
    )
    
    # Calculate instantaneous frequency (derivative of phase)
    inst_freq = xr.apply_ufunc(
        lambda x: np.diff(x),
        unwrapped_phase,
        input_core_dims=[['time']],
        output_core_dims=[['time_diff']],
        vectorize=True,
        dask='parallelized'
    )
    
    # Convert to Hz (assuming time step = 1)
    inst_freq = inst_freq / (2 * np.pi)
    
    # Create time coordinate for the differentiated data
    inst_freq = inst_freq.assign_coords(time_diff=np.arange(len(time_series.time)-1))
    
    return inst_freq

def create_interactive_phase_animation(time_series, sample_rate=10):
    """Create an interactive animation of phase evolution over time"""
    # Sample time points for efficiency
    sampled_times = np.arange(0, len(time_series.time), sample_rate)
    sampled_series = time_series.isel(time=sampled_times)
    
    # Convert to a format holoviews can understand
    # Create dataframe with oscillator, time, phase, and band
    df_list = []
    
    for t_idx, t in enumerate(sampled_series.time.values):
        for o_idx, osc in enumerate(sampled_series.oscillator.values):
            df_list.append({
                'time': t,
                'oscillator': osc,
                'phase': sampled_series.isel(oscillator=o_idx, time=t_idx).values.item(),
                'band': sampled_series.band.values[o_idx]
            })
    
    df = pd.DataFrame(df_list)
    
    # Create a HoloViews Dataset from the DataFrame
    ds = hv.Dataset(df, kdims=['time', 'oscillator'], vdims=['phase', 'band'])
    
    # Create a dynamic map of the polar plot
    def phase_plot(time):
        phases = ds.select(time=time)
        # Convert to a format suitable for Points
        data = {
            'angle': phases.data.phase,
            'r': np.ones(len(phases.data.oscillator)),
            'band': phases.data.band
        }
        
        points = hv.Points(
            pd.DataFrame(data),
            kdims=['angle', 'r'],
            vdims=['band']
        ).opts(color='band', cmap='Category10', size=8, alpha=0.7, projection='polar')
        
        return points.relabel(f'Phase distribution at t={time}')
    
    dmap = hv.DynamicMap(phase_plot, kdims=['time']).redim.values(time=sampled_series.time.values)
    
    return dmap

def compute_chimera_index(time_series, window_size=5):
    """Compute the chimera index (local order parameter variability)
    
    A chimera state is characterized by coexisting coherent and incoherent regions
    """
    phases = time_series.values
    n_oscillators = phases.shape[0]
    times = phases.shape[1]
    
    # Calculate local order parameter
    local_order = np.zeros((n_oscillators, times))
    
    for i in range(n_oscillators):
        # Find neighbors (circular boundary conditions)
        neighbors = [(i-j) % n_oscillators for j in range(-window_size, window_size+1)]
        
        # Calculate local order parameter
        complex_phases = np.exp(1j * phases[neighbors, :])
        local_order[i, :] = np.abs(np.mean(complex_phases, axis=0))
    
    # Chimera index is the standard deviation of local order parameters
    chimera_index = np.std(local_order, axis=0)
    
    # Create xarray with time dimension
    return xr.DataArray(
        chimera_index,
        dims=['time'],
        coords={'time': np.arange(times)}
    )

def plot_functional_analysis(time_series):
    """Perform a full functional analysis of Kuramoto simulation results"""
    # Reshape time_series if needed to ensure it has the right dimensions
    # Assuming time_series has shape (oscillators, times)
    
    # Calculate various metrics
    band_orders = calculate_band_order_parameters(time_series)
    coherence_matrix = calculate_phase_coherence(time_series)
    band_coherence = calculate_band_coherence(coherence_matrix)
    chimera_idx = compute_chimera_index(time_series)
    
    # Create plots
    order_fig = plot_order_parameters(band_orders)
    phase_fig = plot_phase_distribution(time_series)
    coherence_fig = plot_coherence_matrix(coherence_matrix)
    band_coherence_fig = plot_band_coherence_heatmap(band_coherence)
    
    # Create a figure for chimera index
    chimera_fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(chimera_idx.values)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Chimera index')
    ax.set_title('Chimera index evolution (higher values indicate more chimera-like states)')
    ax.grid(True, alpha=0.3)
    
    return {
        'band_orders': band_orders,
        'coherence_matrix': coherence_matrix,
        'band_coherence': band_coherence,
        'chimera_index': chimera_idx,
        'figures': {
            'order_parameters': order_fig,
            'phase_distribution': phase_fig,
            'coherence_matrix': coherence_fig,
            'band_coherence': band_coherence_fig,
            'chimera_index': chimera_fig
        }
    }