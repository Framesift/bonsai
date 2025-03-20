import numpy as np


def apply_noise_session(kb, duration=20):
    """Apply random noise to reset oscillator states between training sessions."""
    # Save original phases
    original_amplitudes = kb.amplitudes.copy()
    
    # Gradually introduce noise
    for i in range(duration):
        # Generate random noise vector
        noise = np.random.normal(0, 0.1, kb.perception_dim)
        
        # Inject with increasing strength
        strength = i / duration * 0.5  # Max strength 0.5
        kb.inject_perception(noise, strength=strength)
        
        # Run dynamics
        #kb.update_oscillators(steps=5)
        kb.update_oscillators_to_convergence()
        
        # Reduce amplitudes slightly to further aid desynchronization
        kb.amplitudes *= 0.99
    
    # Restore original amplitudes
    kb.amplitudes = original_amplitudes
    
    print("Noise session completed - oscillator state reset")