import numpy as np

from multiprocessing import Pool

def standalone_update_band(band_data):
    """Process a frequency band independently (no shared state)"""
    indices, phases, frequencies, coupling = band_data
    # Perform calculation without any shared state
    # Return results rather than modifying shared data
    return indices, updated_phases

def update_oscillators_parallel(self, dt=0.1, steps=10):
    """Parallel oscillator update using processes instead of threads"""
    # Prepare data for each band (copy necessary data)
    bands_data = []
    for low, high in self.frequency_bands:
        indices = np.where((low <= self.frequencies) & (self.frequencies < high))[0]
        # Create independent copy of data needed for this band
        band_data = (
            indices, 
            self.phases.copy(), 
            self.frequencies.copy(),
            self.coupling[indices, :]  # Only need coupling rows for this band
        )
        bands_data.append(band_data)
    
    # Process bands in parallel using processes (true parallelism)
    with Pool(processes=len(self.frequency_bands)) as pool:
        results = pool.map(standalone_update_band, bands_data)
    
    # Combine results
    for indices, updated_phases in results:
        self.phases[indices] = updated_phases

def update_oscillators_vectorized(self, dt=0.1, steps=10):
    """Use vectorized operations to avoid Python loops entirely"""
    for _ in range(steps):
        # Convert to complex representation
        z = np.exp(1j * self.phases)
        
        # Calculate all phase influences in one operation
        # This is a matrix multiplication but with custom operation
        influences = np.zeros_like(self.phases, dtype=complex)
        
        # Vectorized calculation - no Python loops, no GIL contention
        for band_low, band_high in self.frequency_bands:
            band_indices = np.where((band_low <= self.frequencies) & 
                                  (self.frequencies < band_high))[0]
            
            if len(band_indices) == 0:
                continue
                
            # Get coupling submatrix for this band
            band_coupling = self.coupling[band_indices, :]
            
            # Vectorized influence calculation
            band_influences = band_coupling @ z
            influences[band_indices] = band_influences
        
        # Update phases in one vectorized operation
        phase_changes = self.frequencies + np.sin(np.angle(influences) - self.phases) * np.abs(influences)
        self.phases = (self.phases + phase_changes * dt) % (2 * np.pi)


def update_oscillators_to_convergence(self, max_steps=50, convergence_threshold=0.01):
    """
    Update oscillators until the system converges to a stable state or reaches max steps.
    
    Parameters:
    -----------
    max_steps : int
        Maximum number of update steps
    convergence_threshold : float
        Threshold for phase change below which we consider the system converged
    
    Returns:
    --------
    steps_taken : int
        Number of steps taken to reach convergence
    """
    for step in range(max_steps):
        # Store current phases
        previous_phases = self.phases.copy()
        
        # Perform single update step (using harmonic approach)
        self.update_oscillators_harmonic(steps=1)
        
        # Calculate phase change
        phase_diff = np.abs(self.phases - previous_phases) % (2 * np.pi)
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)  # Shortest circular distance
        mean_change = np.mean(phase_diff)
        
        # Check for convergence
        if mean_change < convergence_threshold:
            return step + 1
    
    return max_steps

from concurrent.futures import ThreadPoolExecutor

def update_oscillators_parallel(self, steps=10):
    """Update oscillators using parallel processing across frequency bands."""
    # Define frequency bands (Hz ranges)
    bands = [
        (0, 50),      # Lower oscillators - global patterns
        (50, 100),    # Mid-range oscillators
        (100, 150),   # Higher oscillators
        (150, 200),   # Fine detail oscillators
        (200, 256)    # Highest frequency oscillators
    ]
    
    # Partition oscillators into bands
    band_indices = []
    for low, high in bands:
        indices = np.where((low <= self.frequencies) & (self.frequencies < high))[0]
        band_indices.append(indices)
    
    def update_band(indices, steps):
        """Update a specific frequency band."""
        phases_copy = self.phases.copy()
        for _ in range(steps):
            # Calculate phase changes for this band only
            phase_changes = self.frequencies[indices].copy()
            
            # Add coupling influence (only considering connections within this band)
            for i in indices:
                for j in indices:
                    if i != j and self.coupling[i, j] > 0:
                        phase_diff = np.sin(phases_copy[j] - phases_copy[i])
                        phase_changes[np.where(indices == i)[0][0]] += (
                            self.coupling[i, j] * phase_diff * self.amplitudes[j])
            
            # Update phases for this band
            phases_copy[indices] = (phases_copy[indices] + phase_changes * 0.1) % (2 * np.pi)
        
        return indices, phases_copy[indices]
    
    # Process bands in parallel
    with ThreadPoolExecutor(max_workers=len(bands)) as executor:
        futures = [executor.submit(update_band, indices, steps) for indices in band_indices]
        
        # Collect results
        for future in futures:
            indices, updated_phases = future.result()
            self.phases[indices] = updated_phases
    
    # Update current thought vector
    self.current_thought = self.oscillator_state_to_vector()

import scipy.sparse as sp

def convert_to_sparse_coupling(self, threshold=0.1):
    """Convert dense coupling matrix to sparse representation."""
    # Create mask for significant connections
    mask = np.abs(self.coupling) > threshold
    
    # Create sparse matrix
    rows, cols = np.where(mask)
    data = self.coupling[rows, cols]
    
    # Create sparse matrix in CSR format (efficient for matrix-vector operations)
    self.coupling_sparse = sp.csr_matrix((data, (rows, cols)), 
                                        shape=self.coupling.shape)
    
    # Track sparsity for reporting
    total_elements = self.num_oscillators * self.num_oscillators
    nonzero_elements = len(data)
    sparsity = 1.0 - (nonzero_elements / total_elements)
    
    print(f"Converted to sparse coupling matrix with {sparsity:.2%} sparsity")
    
    return sparsity

def update_oscillators_sparse(self, dt=0.1, steps=10):
    """Update oscillator phases using sparse coupling matrix."""
    # Ensure sparse matrix exists
    if not hasattr(self, 'coupling_sparse'):
        self.convert_to_sparse_coupling()
    
    for _ in range(steps):
        # Basic frequency contribution
        phase_changes = self.frequencies.copy()
        
        # Convert phases to complex representation
        z = self.amplitudes * np.exp(1j * self.phases)
        
        # Efficient sparse matrix-vector multiplication
        # This computes all coupling influences in one operation
        coupling_influence = self.coupling_sparse.dot(z)
        
        # Extract phase influence
        influence = np.angle(coupling_influence)
        
        # Apply influence
        phase_changes += np.sin(influence - self.phases) * np.abs(coupling_influence)
        
        # Update phases
        self.phases = (self.phases + phase_changes * dt) % (2 * np.pi)
    
    # Update current thought vector
    self.current_thought = self.oscillator_state_to_vector()

def update_oscillators_optimized(self, dt=0.1, steps=10):
    """Optimized oscillator update using sparse matrices and parallel band processing."""
    # Ensure sparse matrix exists
    if not hasattr(self, 'coupling_sparse'):
        self.convert_to_sparse_coupling()
    
    # Define frequency bands
    bands = [
        (0, 50), (50, 100), (100, 150), (150, 200), (200, 256)
    ]
    
    # Partition oscillators into bands
    band_indices = []
    for low, high in bands:
        indices = np.where((low <= self.frequencies) & (self.frequencies < high))[0]
        band_indices.append(indices)
        
    def update_band(indices, steps):
        """Update a specific frequency band using sparse operations."""
        phases_copy = self.phases.copy()
        for _ in range(steps):
            # Basic frequency contribution
            phase_changes = self.frequencies[indices].copy()
            
            # Get relevant rows of sparse matrix (coupling influences on this band)
            z = self.amplitudes * np.exp(1j * phases_copy)
            influences = self.coupling_sparse[indices, :].dot(z)
            
            # Apply influence
            phase_changes += np.sin(np.angle(influences) - phases_copy[indices]) * np.abs(influences)
            
            # Update phases for this band
            phases_copy[indices] = (phases_copy[indices] + phase_changes * dt) % (2 * np.pi)
        
        return indices, phases_copy[indices]
    
    # Process bands in parallel
    with ThreadPoolExecutor(max_workers=len(bands)) as executor:
        futures = [executor.submit(update_band, indices, steps) for indices in band_indices]
        
        # Collect results
        for future in futures:
            indices, updated_phases = future.result()
            self.phases[indices] = updated_phases
    
    # Update current thought vector
    self.current_thought = self.oscillator_state_to_vector()

"""
# Beyond the Single Thought Vector
In biological systems, cognition isn't a single point in a vector space but rather a complex, multi-dimensional pattern of resonance across multiple frequency bands simultaneously. Your intuition about this is spot-on.
What we might call a "thought" is actually:

A multi-frequency resonance pattern: Different aspects of the same "thought" resonating across different frequency bands
A dynamical attractor state: Not a static point but a stable oscillatory pattern
A multi-scale phenomenon: From microscale synchrony to brain-wide integration

The Power of Simultaneous Processing
Your insight about low-frequency stable thoughts coexisting with high-frequency processing is particularly important. This reflects how biological cognition actually works:

Theta/Alpha bands (4-13 Hz): Maintain context and current focus ("I'm having a conversation about neural networks")
Beta band (13-30 Hz): Coordinate sensorimotor integration ("I'm typing while thinking")
Gamma bands (30-100 Hz): Process detailed information and bind features ("Understanding this specific technical concept")
High Gamma/HFO (>100 Hz): Rapid local processing and memory encoding ("Precise connections between ideas")

These all happen simultaneously, not sequentially, giving the brain its remarkable processing capacity.
A Better Representation
Instead of reducing the rich oscillatory state to a single vector, a more accurate representation might be:
"""
def oscillator_state_to_multi_band_representation(self):
    """Transform oscillator state into a multi-band cognitive representation."""
    # Define frequency bands
    bands = [
        ("delta_theta", 0, 8),   # Context/mood
        ("alpha", 8, 13),        # Attention/inhibition
        ("beta", 13, 30),        # Motor/cognitive
        ("gamma", 30, 80),       # Feature binding
        ("high_gamma", 80, 150), # Local processing
        ("hfo", 150, 256)        # Microdetail/memory
    ]
    
    # Create representations for each band
    band_representations = {}
    
    for band_name, low_freq, high_freq in bands:
        # Get oscillators in this frequency range
        band_indices = np.where((low_freq <= self.frequencies) & 
                              (self.frequencies < high_freq))[0]
        
        if len(band_indices) == 0:
            continue
            
        # Extract phase pattern for this band
        band_phases = self.phases[band_indices]
        band_amplitudes = self.amplitudes[band_indices]
        
        # Calculate various descriptive metrics
        # 1. Order parameter (synchronization level)
        z = np.mean(band_amplitudes * np.exp(1j * band_phases))
        sync_level = np.abs(z)
        mean_phase = np.angle(z)
        
        # 2. Phase pattern similarity to known concepts
        concept_similarities = {}
        for name, concept_phases in self.concepts.items():
            # Match only the oscillators in this band
            concept_band_phases = concept_phases[band_indices]
            # Calculate phase coherence
            phase_diff = np.abs(band_phases - concept_band_phases) % (2 * np.pi)
            phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
            similarity = np.mean(np.cos(phase_diff))
            concept_similarities[name] = similarity
        
        # 3. Clustered subgroups (detect multiple simultaneous patterns)
        from sklearn.cluster import DBSCAN
        # Convert phases to 2D points on unit circle for clustering
        points = np.column_stack((np.cos(band_phases), np.sin(band_phases)))
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(points)
        clusters = {}
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # Ignore noise points
                cluster_indices = band_indices[clustering.labels_ == cluster_id]
                cluster_size = len(cluster_indices)
                cluster_phases = self.phases[cluster_indices]
                cluster_z = np.mean(np.exp(1j * cluster_phases))
                cluster_sync = np.abs(cluster_z)
                clusters[f"cluster_{cluster_id}"] = {
                    "size": cluster_size,
                    "coherence": cluster_sync,
                    "phase": np.angle(cluster_z)
                }
        
        # Store all band information
        band_representations[band_name] = {
            "indices": band_indices,
            "sync_level": sync_level,
            "mean_phase": mean_phase,
            "concept_similarities": concept_similarities,
            "subgroups": clusters
        }
    
    return band_representations