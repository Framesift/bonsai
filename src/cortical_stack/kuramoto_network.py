from dataclasses import dataclass
import holoviews as hv
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import multiprocessing as mp
from functools import partial
from analyze import calculate_band_order_parameters, create_interactive_phase_animation, plot_functional_analysis
from utils import timer, debug

@dataclass
class KuramotoNetwork:
    """Represents a network of Kuramoto oscillators with coupling matrix"""
    
    # Coupling matrix as an xarray DataArray with named dimensions
    coupling_matrix: xr.DataArray
    
    # Oscillator natural frequencies
    natural_frequencies: xr.DataArray
    
    # Current phase states
    phases: xr.DataArray
    
    @classmethod
    def create(cls, n_oscillators, band_names=None, coupling_strength=0.1, random_seed=None):
        """Create a Kuramoto network with oscillators in different frequency bands"""
        if band_names is None:
            band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create oscillator IDs with their band type
        oscillator_ids = np.arange(n_oscillators)
        oscillator_bands = np.random.choice(band_names, size=n_oscillators)
        
        # Create coupling matrix with named dimensions
        coupling = np.random.uniform(-coupling_strength, coupling_strength, 
                                    size=(n_oscillators, n_oscillators))
        
        # Make it symmetric for physical plausibility
        coupling = (coupling + coupling.T) / 2
        
        coupling_xr = xr.DataArray(
            coupling,
            dims=['source', 'target'],
            coords={
                'source': oscillator_ids,
                'source_band': ('source', oscillator_bands),
                'target': oscillator_ids,
                'target_band': ('target', oscillator_bands)
            }
        )
        
        # Natural frequencies with band information
        frequencies = np.random.uniform(0.5, 50, size=n_oscillators)
        freq_xr = xr.DataArray(
            frequencies,
            dims=['oscillator'],
            coords={
                'oscillator': oscillator_ids,
                'band': ('oscillator', oscillator_bands)
            }
        )
        
        # Initial phases
        phases = np.random.uniform(0, 2*np.pi, size=n_oscillators)
        phases_xr = xr.DataArray(
            phases,
            dims=['oscillator'],
            coords={
                'oscillator': oscillator_ids,
                'band': ('oscillator', oscillator_bands)
            }
        )
        
        return cls(coupling_xr, freq_xr, phases_xr)
    
    #@timer
    def update_step(self, dt=0.01, idx_range=None):
        """Update phases for a subset of oscillators (for parallel processing)"""
        if idx_range is None:
            oscillator_range = slice(None)
        else:
            start, end = idx_range
            oscillator_range = slice(start, end)
        
        phases = self.phases.values
        freqs = self.natural_frequencies.values
        coupling = self.coupling_matrix.values
        
        # Only update the specified range of oscillators
        for i in range(*idx_range) if idx_range else range(len(phases)):
            # Kuramoto model: dθᵢ/dt = ωᵢ + (1/N)∑ⱼKᵢⱼsin(θⱼ-θᵢ)
            phase_diffs = np.sin(phases - phases[i])
            coupling_effect = np.sum(coupling[i, :] * phase_diffs)
            
            # Update phase
            phases[i] += dt * (freqs[i] + coupling_effect)
            
        # Normalize to [0, 2π]
        phases %= (2 * np.pi)
        
        # Only return updated values for the specified range
        if idx_range:
            start, end = idx_range
            return oscillator_range, phases[start:end]
        return oscillator_range, phases
    
    def update_step_vectorized(self, dt=0.01, idx_range=None):
        """Vectorized update of all oscillator phases"""
        phases = self.phases.values
        freqs = self.natural_frequencies.values
        coupling = self.coupling_matrix.values
        
        # Calculate all phase differences at once
        phase_diffs = np.subtract.outer(phases, phases)
        sin_diffs = np.sin(phase_diffs)
        
        # Calculate coupling effect for all oscillators at once
        coupling_effect = np.sum(coupling * sin_diffs, axis=1)
        
        # Update all phases simultaneously
        new_phases = phases + dt * (freqs + coupling_effect)
        
        # Normalize to [0, 2π]
        new_phases %= (2 * np.pi)
        
        # Update phases
        self.phases = xr.DataArray(
            new_phases,
            dims=self.phases.dims,
            coords=self.phases.coords
        )
        
        return new_phases
    
    @timing
    def simulate_vectorized(self, steps=100, dt=0.01):
        """
        Run simulation using vectorized update method and return time series for analysis
        
        Parameters
        ----------
        steps : int
            Number of simulation steps
        dt : float
            Time step size
            
        Returns
        -------
        xr.DataArray
            Time series of phases with dimensions [oscillator, time]
            and coordinates for 'band' on the oscillator dimension
        """
        # Store results
        results = np.zeros((len(self.phases), steps))
        
        # Store initial state
        results[:, 0] = self.phases.values
        
        # Run simulation
        for step in range(1, steps):
            # Update phases using vectorized method
            new_phases = self.update_step_vectorized(dt)
            
            # Store result
            results[:, step] = new_phases
        
        # Create xarray DataArray with proper coordinates
        time_series = xr.DataArray(
            results,
            dims=['oscillator', 'time'],
            coords={
                'oscillator': self.phases.coords['oscillator'],
                'band': ('oscillator', self.phases.coords['band'].values),
                'time': np.arange(steps)
            }
        )
        
        return time_series
    
    @timer
    def simulate_parallel(self, steps=10, dt=0.01, n_processes=None) -> xr.DataArray:
        """Run simulation in parallel across multiple CPU cores"""
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        n_oscillators = len(self.phases)
        chunk_size = n_oscillators // n_processes
        
        # Create chunks for parallel processing
        chunks = []
        for i in range(n_processes):
            start = i * chunk_size
            end = start + chunk_size if i < n_processes - 1 else n_oscillators
            chunks.append((start, end))
        
        results = []
        
        for step in range(steps):
            # Create a pool for each step (to ensure synchronization)
            with mp.Pool(processes=n_processes) as pool:
                # Run update_step in parallel for different chunks
                func = partial(self.update_step, dt)
                chunk_results = pool.map(func, chunks)
            
            # Collect and combine results
            updated_phases = np.zeros_like(self.phases.values)
            for (start, end), (_, phases) in zip(chunks, chunk_results):
                updated_phases[start:end] = phases
            
            # Update the main phases array
            self.phases = xr.DataArray(
                updated_phases,
                dims=self.phases.dims,
                coords=self.phases.coords
            )
            
            # Store result if needed
            results.append(self.phases.copy())
        
        # Return time series of phases
        return xr.concat(results, dim='time')
    
    @timer
    def filter_coupling_by_band(self, source_band=None, target_band=None):
        """Extract coupling submatrix between specific oscillator bands"""
        filtered = self.coupling_matrix
        
        if source_band:
            filtered = filtered.sel(source=filtered.source_band == source_band)
        
        if target_band:
            filtered = filtered.sel(target=filtered.target_band == target_band)
            
        return filtered
    
    @timer
    def apply_transformation(self, transform_func):
        """Apply a transformation function to the coupling matrix"""
        new_coupling = transform_func(self.coupling_matrix.values)
        
        # Create a new instance with the transformed coupling
        return KuramotoNetwork(
            xr.DataArray(
                new_coupling, 
                dims=self.coupling_matrix.dims,
                coords=self.coupling_matrix.coords
            ),
            self.natural_frequencies,
            self.phases
        )
    
    @timer
    def apply_band_specific_transformation(self, source_band, target_band, transform_func):
        """Apply transformation only to connections between specific bands"""
        # Create mask for the specified bands
        source_mask = self.coupling_matrix.source_band == source_band
        target_mask = self.coupling_matrix.target_band == target_band
        
        # Create a copy of the coupling matrix
        new_coupling = self.coupling_matrix.values.copy()
        
        # Apply transformation only to the specified submatrix
        for i in np.where(source_mask)[0]:
            for j in np.where(target_mask)[0]:
                new_coupling[i, j] = transform_func(new_coupling[i, j])
        
        # Create a new instance with the transformed coupling
        return KuramotoNetwork(
            xr.DataArray(
                new_coupling, 
                dims=self.coupling_matrix.dims,
                coords=self.coupling_matrix.coords
            ),
            self.natural_frequencies,
            self.phases
        )
    
if (__name__ == "__main__"):
    # Create a network of 100 oscillators
    network = KuramotoNetwork.create(n_oscillators=100, random_seed=42)

    # Run simulation using multiple CPU cores
    time_series = network.simulate_vectorized(steps=100) #, n_processes=4)

    # Visualize synchronization between different frequency bands
    alpha_beta_coupling = network.filter_coupling_by_band(source_band='alpha', target_band='beta')
    print(f"Average coupling strength from alpha to beta: {alpha_beta_coupling.mean().values}")

    # Apply a transformation to strengthen coupling within the alpha band
    def strengthen_coupling(x):
        return x * 1.5 if x > 0 else x

    alpha_enhanced = network.apply_band_specific_transformation(
        source_band='alpha', 
        target_band='alpha', 
        transform_func=strengthen_coupling
    )

    # Run simulation with the transformed network
    enhanced_time_series = alpha_enhanced.simulate_vectorized(steps=100) #, n_processes=4)
   
   # Assuming you have run your simulation and have time_series data
    # time_series should be an xarray DataArray with dimensions [oscillator, time]
    # and coordinates for 'band' on the oscillator dimension

    # Run the full analysis
    analysis_results = plot_functional_analysis(time_series)

    # Display all figures
    for name, fig in analysis_results['figures'].items():
        plt.figure(fig.number)
        plt.savefig(f"{name}.png", dpi=300, bbox_inches='tight')
        plt.show()

    # Compare two different simulations (e.g., original vs transformed network)
    def compare_simulations(sim1, sim2, labels=['Original', 'Transformed']):
        """Compare two different simulation results"""
        # Calculate order parameters for both simulations
        orders1 = calculate_band_order_parameters(sim1)
        orders2 = calculate_band_order_parameters(sim2)
        
        # Plot comparison
        fig, axes = plt.subplots(len(orders1.data_vars), 1, figsize=(12, 4*len(orders1.data_vars)))
        
        for i, band in enumerate(orders1.data_vars):
            ax = axes[i] if len(orders1.data_vars) > 1 else axes
            
            ax.plot(orders1[band].values, label=f"{labels[0]}")
            ax.plot(orders2[band].values, label=f"{labels[1]}")
            
            ax.set_title(f"{band} band synchronization")
            ax.set_xlabel('Time step')
            ax.set_ylabel('Order parameter (r)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    # Compare original and transformed simulations
    comparison_fig = compare_simulations(time_series, enhanced_time_series)
    plt.savefig("comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create an interactive phase animation
    phase_animation = create_interactive_phase_animation(time_series)
    hv.save(phase_animation, 'phase_animation.html')