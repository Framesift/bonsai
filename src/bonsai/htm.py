import numpy as np

from bonsai import update_oscillators
from bonsai.perception import inject_perception
from bonsai.visual_pipeline import extract_feature_vector

def maintain_memory(state, memory_buffer, decay=0.9, max_buffer=10):
    """
    Maintain a rolling memory buffer with temporal integration.
    
    Parameters:
    -----------
    state : dict
        Current oscillator state.
    memory_buffer : list
        Buffer of past oscillator states.
    decay : float
        Memory retention factor (higher = longer memory).
    max_buffer : int
        Maximum buffer length.

    Returns:
    --------
    updated_memory : list
    """
    if len(memory_buffer) >= max_buffer:
        memory_buffer.pop(0)

    if memory_buffer:
        averaged_state = {
            "phases": decay * memory_buffer[-1]["phases"] + (1 - decay) * state["phases"],
            "frequencies": decay * memory_buffer[-1]["frequencies"] + (1 - decay) * state["frequencies"],
            "amplitudes": decay * memory_buffer[-1]["amplitudes"] + (1 - decay) * state["amplitudes"]
        }
    else:
        averaged_state = state

    memory_buffer.append(averaged_state)
    return memory_buffer

def compute_sdr_state(state, threshold=0.75):
    """
    Convert oscillator states into a Sparse Distributed Representation (SDR).
    
    Parameters:
    -----------
    state : dict
        Oscillator state.
    threshold : float
        Minimum phase coherence required for activation.

    Returns:
    --------
    sdr : ndarray
        Sparse binary vector.
    """
    phases = state["phases"]
    coherence = np.abs(np.mean(np.exp(1j * phases)))  # Kuramoto order parameter
    
    sdr = (coherence > threshold).astype(int)  # Binary activation
    return sdr

def reinforce_attractors(state, coupling, stability_threshold=0.8, reinforcement_factor=1.1):
    """
    Reinforce attractors by increasing coupling between coherent oscillators.
    
    Parameters:
    -----------
    state : dict
        Current oscillator state.
    coupling : ndarray
        Coupling matrix.
    stability_threshold : float
        Phase coherence threshold for reinforcement.
    reinforcement_factor : float
        How much to increase coupling.

    Returns:
    --------
    updated_coupling : ndarray
    """
    phases = state["phases"]
    phase_coherence = np.abs(np.mean(np.exp(1j * phases)))  # Global coherence measure

    if phase_coherence > stability_threshold:
        coupling *= reinforcement_factor
        np.clip(coupling, 0, 1, out=coupling)  # Ensure within valid range

    return coupling



def adapt_coupling(coupling, oscillator_state, prediction_error, plasticity_rate=0.05):
    """Adapt coupling strength based on prediction error and phase stability."""
    phases = oscillator_state["phases"]
    
    # Compute global phase stability (Kuramoto order parameter)
    phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
    
    # Adapt coupling based on stability-weighted error
    adjustment = plasticity_rate * prediction_error * phase_coherence  
    new_coupling = coupling * (1 - adjustment)
    
    np.clip(new_coupling, 0, 1, out=new_coupling)  # Keep values in range
    return new_coupling


def maintain_contextual_memory(oscillator_state, memory_buffer, decay_factor=0.9, relevance_threshold=0.75):
    """Maintain memory based on relevance to current state."""
    if not memory_buffer:
        return [oscillator_state]
    
    # Compute similarity between current state and past memories
    similarities = [
        np.dot(oscillator_state["phases"], past["phases"]) / 
        (np.linalg.norm(oscillator_state["phases"]) * np.linalg.norm(past["phases"]) + 1e-10)
        for past in memory_buffer
    ]
    
    # Retain most relevant memories, decay others
    updated_memory = []
    for past, sim in zip(memory_buffer, similarities):
        retention = 1 if sim > relevance_threshold else decay_factor
        new_state = {
            "phases": past["phases"] * retention,
            "frequencies": past["frequencies"] * retention,
            "amplitudes": past["amplitudes"] * retention
        }
        updated_memory.append(new_state)
    
    return updated_memory[-10:]  # Keep last N entries


def selective_attention(oscillator_state, suppression_factor=0.2, activation_boost=1.2):
    """Enhance relevant phase-synchronized oscillators, suppress others."""
    phases = oscillator_state["phases"]
    
    # Compute phase coherence (global attention signal)
    coherence = np.abs(np.mean(np.exp(1j * phases)))
    
    # Adjust amplitudes based on coherence
    oscillator_state["amplitudes"] *= (activation_boost if coherence > 0.7 else suppression_factor)
    
    return oscillator_state


def process_perception_htm_kuramoto(image, oscillator_state, coupling, memory_buffer):
    """Process perception through the combined HTM-Kuramoto pipeline."""
    # Extract features
    perception_vector = extract_feature_vector(image)
    
    # Convert to SDR
    current_sdr = compute_sdr_from_vector(perception_vector)
    
    # Predict next SDR based on sequence memory (HTM)
    predicted_sdr = predict_next_pattern(current_sdr, memory_buffer)
    
    # Inject both current perception and prediction into oscillator dynamics
    oscillator_state = inject_perception(oscillator_state, perception_vector, strength=0.6)
    oscillator_state = inject_sdr(oscillator_state, predicted_sdr, strength=0.4)
    
    # Update oscillator dynamics
    oscillator_state = update_oscillators(oscillator_state, coupling)
    
    # Update temporal memory
    memory_buffer = maintain_memory(oscillator_state, memory_buffer)
    
    # Adapt coupling based on prediction accuracy
    prediction_error = compute_prediction_error(current_sdr, predicted_sdr)
    coupling = adapt_coupling(coupling, oscillator_state, prediction_error)
    
    return oscillator_state, coupling, memory_buffer

def process_multiscale_memory(oscillator_state, memory_buffers):
    """Process memory at multiple timescales."""
    # Define timescales with different decay rates
    timescales = [
        {"name": "immediate", "decay": 0.2, "max_buffer": 5},
        {"name": "working", "decay": 0.7, "max_buffer": 20},
        {"name": "episodic", "decay": 0.95, "max_buffer": 100}
    ]
    
    # Process each timescale independently
    updated_buffers = {}
    for scale in timescales:
        buffer = memory_buffers.get(scale["name"], [])
        updated_buffers[scale["name"]] = maintain_memory(
            oscillator_state, 
            buffer, 
            decay=scale["decay"], 
            max_buffer=scale["max_buffer"]
        )
    
    return updated_buffers

def compute_sdr_from_oscillator_state(state, sparsity=0.02):
    """Convert oscillator state to Sparse Distributed Representation."""
    # Extract phases
    phases = state["phases"]
    
    # Compute phase coherence within frequency bands
    bands = [(0, 8), (8, 13), (13, 30), (30, 80), (80, 150)]
    band_coherence = []
    
    for low, high in bands:
        indices = np.where((low <= state["frequencies"]) & (state["frequencies"] < high))[0]
        if len(indices) > 0:
            band_phases = phases[indices]
            z = np.mean(np.exp(1j * band_phases))
            coherence = np.abs(z)
            band_coherence.append(coherence)
    
    # Normalize coherence
    coherence = np.array(band_coherence) / np.sum(band_coherence)
    
    # Determine activation threshold to achieve target sparsity
    n_active = int(len(phases) * sparsity)
    
    # Select most coherent oscillators to activate
    threshold = np.sort(coherence)[-n_active] if n_active < len(coherence) else 0
    
    # Create SDR
    sdr = (coherence >= threshold).astype(int)
    
    return sdr

def sleep_consolidation(oscillator_states, coupling, memory_buffers):
    """Perform sleep consolidation to refine memory and coupling."""
    # Extract patterns from episodic memory
    episodic_buffer = memory_buffers.get("episodic", [])
    if not episodic_buffer:
        return coupling
    
    # Identify recurring patterns
    patterns = extract_recurring_patterns(episodic_buffer)
    
    # Initialize new coupling matrix
    new_coupling = coupling.copy()
    
    # Reinforce connections for recurring patterns
    for pattern in patterns:
        # Replay pattern
        state = pattern["state"]
        
        # Update coupling based on pattern stability
        stability = pattern["stability"]
        
        # Strengthen connections proportional to stability
        for i in range(len(new_coupling)):
            for j in range(len(new_coupling)):
                if i != j and new_coupling[i, j] > 0:
                    # Calculate phase relationship in this pattern
                    phase_i = state["phases"][i]
                    phase_j = state["phases"][j]
                    phase_diff = np.abs((phase_i - phase_j) % (2 * np.pi))
                    
                    # If phases were synchronized in this pattern, strengthen connection
                    if phase_diff < 0.5 or phase_diff > 5.8:  # Near 0 or 2π
                        new_coupling[i, j] *= (1 + 0.1 * stability)
    
    # Ensure coupling stays in valid range
    np.clip(new_coupling, 0, 1, out=new_coupling)
    
    return new_coupling

def integrate_cross_modal(visual_features, audio_features, text_features):
    """Integrate features from multiple modalities."""
    # Create individual SDRs for each modality
    visual_sdr = compute_sdr_from_vector(visual_features)
    audio_sdr = compute_sdr_from_vector(audio_features)
    text_sdr = compute_sdr_from_vector(text_features)
    
    # Combine SDRs using union operation (with slight overlap)
    combined_sdr = np.logical_or(
        np.logical_or(visual_sdr, audio_sdr),
        text_sdr
    ).astype(int)
    
    return combined_sdr