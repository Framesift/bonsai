import numpy as np


def initialize_oscillators(num_oscillators, strategy="random", **params):
    """
    Create initial oscillator state.
    
    Returns:
    --------
    state : dict
        Dictionary containing oscillator state
    """
    # Create state container
    state = {
        "phases": np.zeros(num_oscillators),
        "frequencies": np.zeros(num_oscillators),
        "amplitudes": np.ones(num_oscillators)
    }
    
    # Initialize based on strategy
    if strategy == "random":
        state["phases"] = np.random.uniform(0, 2*np.pi, num_oscillators)
        state["frequencies"] = np.random.normal(10, 2, num_oscillators)
    elif strategy == "harmonic":
        # Create harmonic frequency distribution
        bands = params.get("bands", [(0, 8), (8, 13), (13, 30), (30, 80), (80, 150)])
        oscillators_per_band = num_oscillators // len(bands)
        
        for i, (low, high) in enumerate(bands):
            start_idx = i * oscillators_per_band
            end_idx = (i + 1) * oscillators_per_band if i < len(bands) - 1 else num_oscillators
            band_size = end_idx - start_idx
            
            # Linear distribution within band
            state["frequencies"][start_idx:end_idx] = np.linspace(low, high, band_size)
            # Random phases
            state["phases"][start_idx:end_idx] = np.random.uniform(0, 2*np.pi, band_size)
    
    return state

def initialize_coupling(num_oscillators, strategy="small_world", **params):
    """
    Create coupling matrix.
    
    Returns:
    --------
    coupling : ndarray
        Coupling matrix
    """
    coupling = np.zeros((num_oscillators, num_oscillators))
    
    if strategy == "small_world":
        # Local connections
        k = params.get("local_k", 4)
        local_strength = params.get("local_strength", 0.1)
        
        for i in range(num_oscillators):
            for j in range(-k, k+1):
                if j != 0:
                    neighbor = (i + j) % num_oscillators
                    coupling[i, neighbor] = local_strength
        
        # Add long-range connections
        p_rewire = params.get("p_rewire", 0.1)
        long_strength = params.get("long_strength", 0.05)
        
        for i in range(num_oscillators):
            if np.random.random() < p_rewire:
                # Rewire to random target
                target = np.random.randint(0, num_oscillators)
                if target != i:
                    coupling[i, target] = long_strength
    
    # elif strategy == "connectome":
    #     # Implement connectome-inspired structure
    #     # ...
    
    return coupling

def update_oscillators(state, coupling, dt=0.1, steps=10, strategy="vectorized"):
    """
    Update oscillator state.
    
    Parameters:
    -----------
    state : dict
        Current oscillator state
    coupling : ndarray
        Coupling matrix
    dt : float
        Time step
    steps : int
        Number of integration steps
    strategy : str
        Update strategy to use
        
    Returns:
    --------
    new_state : dict
        Updated oscillator state
    """
    # Extract state components
    phases = state["phases"].copy()
    frequencies = state["frequencies"]
    amplitudes = state["amplitudes"]
    
    if strategy == "vectorized":
        # Vectorized implementation
        for _ in range(steps):
            # Convert to complex representation
            z = amplitudes * np.exp(1j * phases)
            
            # Calculate all influences in one operation
            influences = coupling @ z
            
            # Update phases
            phase_changes = frequencies + np.sin(np.angle(influences) - phases) * np.abs(influences)
            phases = (phases + phase_changes * dt) % (2 * np.pi)
    
    # elif strategy == "harmonic":
    #     # Band-aware implementation
    #     # ...
    
    # Create new state with updated phases
    new_state = state.copy()
    new_state["phases"] = phases
    
    return new_state

def oscillator_state_to_vector(state, oscillator_dim=128):
    """
    Convert oscillator state to feature vector.
    
    Parameters:
    -----------
    state : dict
        Oscillator state
    oscillator_dim : int
        Dimensionality of output vector
        
    Returns:
    --------
    vector : ndarray
        Feature vector representation
    """
    # Extract phases and amplitudes
    phases = state["phases"]
    amplitudes = state["amplitudes"]
    
    # Convert to complex representation
    z = amplitudes * np.exp(1j * phases)
    
    # Project to higher-dimensional space
    vector = np.zeros(oscillator_dim, dtype=complex)
    num_oscillators = len(phases)
    
    for i in range(num_oscillators):
        for j in range(min(5, num_oscillators)):
            idx = (i * 5 + j) % oscillator_dim
            vector[idx] += z[i] * z[(i + j) % num_oscillators]
    
    # Return real component
    result = np.real(vector)
    return result / (np.linalg.norm(result) + 1e-10)