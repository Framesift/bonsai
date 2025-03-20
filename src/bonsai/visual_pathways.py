import cv2
import numpy as np

from bonsai.htm import compute_sdr_from_oscillator_state
from bonsai.visual_pipeline import compute_shape_dna


def extract_shape_features(image):
    """Extract shape features (ventral 'what' pathway)."""
    # Convert to grayscale for shape processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Multi-scale edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Laplacian Shape DNA features
    shape_dna = compute_shape_dna(edges)
    
    # Contour-based features
    contours, _ = cv2.findContours(edges.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_features = []
    if contours:
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Extract shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
        
        # Approximate shape
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = len(approx)
        
        contour_features = [area, perimeter, circularity, corners]
    else:
        contour_features = [0, 0, 0, 0]
    
    # Combine features
    return np.concatenate([shape_dna, np.array(contour_features)])

def extract_motion_features(image, previous_frames=None):
    """Extract motion features (dorsal 'where/how' pathway)."""
    if previous_frames is None or len(previous_frames) < 2:
        # Without motion information, focus on spatial location features
        h, w = image.shape[:2]
        
        # Create basic grid activation map (where things are in the image)
        grid_size = 4
        grid = np.zeros((grid_size, grid_size))
        
        # Detect edges
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        edges = cv2.Canny(gray, 50, 150)
        
        # Fill grid based on edge density in each cell
        cell_h, cell_w = h // grid_size, w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                y_start, y_end = i * cell_h, (i + 1) * cell_h
                x_start, x_end = j * cell_w, (j + 1) * cell_w
                
                cell = edges[y_start:y_end, x_start:x_end]
                grid[i, j] = np.sum(cell) / (cell_h * cell_w * 255)
        
        return grid.flatten()
    else:
        # With previous frames, compute actual motion features
        prev_frame = previous_frames[-1]
        
        # Calculate optical flow
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            prev_gray = prev_frame.copy()
            
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate flow magnitude and direction
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create histogram of flow directions (8 bins)
        hist, _ = np.histogram(ang, bins=8, range=(0, 2*np.pi), weights=mag)
        normalized_hist = hist / (np.sum(hist) + 1e-10)
        
        return normalized_hist
    
def extract_color_features(image):
    """Extract color features."""
    # If grayscale, return zeros
    if len(image.shape) == 2 or image.shape[2] == 1:
        return np.zeros(16)  # Default size
    
    # Convert to HSV (better for color analysis)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split into channels
    h, s, v = cv2.split(hsv)
    
    # Color histograms
    h_hist = cv2.calcHist([h], [0], None, [8], [0, 180])
    s_hist = cv2.calcHist([s], [0], None, [4], [0, 256])
    v_hist = cv2.calcHist([v], [0], None, [4], [0, 256])
    
    # Normalize
    h_hist = h_hist / np.sum(h_hist)
    s_hist = s_hist / np.sum(s_hist)
    v_hist = v_hist / np.sum(v_hist)
    
    # Dominant colors (top 3 bins from hue histogram)
    dominant_hues = np.argsort(h_hist.flatten())[-3:] * (180/8)
    
    # Colorfulness metric (standard deviation in ab space of Lab color space)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    colorfulness = np.sqrt(np.var(a) + np.var(b))
    
    # Combine features
    return np.concatenate([
        h_hist.flatten(), 
        s_hist.flatten(), 
        v_hist.flatten(),
        [colorfulness]
    ])

def extract_texture_features(image):
    """Extract texture features."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Gabor filter bank for texture analysis
    ksize = 31
    sigma = 5
    theta = np.pi/4
    lambd = 10.0
    gamma = 0.5
    
    # Apply filters at different orientations
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gabor_responses = []
    
    for theta in orientations:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # Get statistical features of response
        mean = np.mean(filtered)
        std = np.std(filtered)
        energy = np.sum(filtered**2)
        
        gabor_responses.extend([mean, std, energy])
    
    # Gray-Level Co-Occurrence Matrix for texture statistics
    # (Simplified version)
    glcm = np.zeros((8, 8))
    quantized = (gray // 32).astype(np.uint8)  # Quantize to 8 levels
    
    h, w = quantized.shape
    for i in range(h-1):
        for j in range(w-1):
            glcm[quantized[i, j], quantized[i+1, j+1]] += 1
    
    # Normalize GLCM
    if np.sum(glcm) > 0:
        glcm = glcm / np.sum(glcm)
    
    # Extract GLCM features
    contrast = 0
    homogeneity = 0
    energy = 0
    correlation = 0
    
    mu_i = 0
    mu_j = 0
    var_i = 0
    var_j = 0
    
    for i in range(8):
        for j in range(8):
            contrast += glcm[i, j] * (i - j)**2
            homogeneity += glcm[i, j] / (1 + (i - j)**2)
            energy += glcm[i, j]**2
            
            mu_i += i * np.sum(glcm[i, :])
            mu_j += j * np.sum(glcm[:, j])
    
    for i in range(8):
        var_i += (i - mu_i)**2 * np.sum(glcm[i, :])
        var_j += (j - mu_j)**2 * np.sum(glcm[:, j])
    
    if var_i > 0 and var_j > 0:
        for i in range(8):
            for j in range(8):
                correlation += glcm[i, j] * (i - mu_i) * (j - mu_j) / np.sqrt(var_i * var_j)
    
    glcm_features = [contrast, homogeneity, energy, correlation]
    
    # Combine all texture features
    return np.array(gabor_responses + glcm_features)

def extract_layout_features(image):
    """Extract global layout features for scene recognition."""
    # Resize to standard size for consistent analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    resized = cv2.resize(gray, (64, 64))
    
    # Gist descriptor (simplified)
    # Divide image into 4x4 grid
    grid_size = 4
    cell_size = 64 // grid_size
    
    grid_features = []
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size
            
            cell = resized[y_start:y_end, x_start:x_end]
            
            # Extract simple statistics for each cell
            mean = np.mean(cell)
            std = np.std(cell)
            grid_features.extend([mean, std])
    
    # Edge orientation histogram for layout
    edges = cv2.Canny(resized, 50, 150)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1)
    
    # Calculate gradient direction on edge pixels only
    edge_mask = edges > 0
    if np.any(edge_mask):
        gx_edges = gx[edge_mask]
        gy_edges = gy[edge_mask]
        
        angles = np.arctan2(gy_edges, gx_edges) % np.pi
        
        # Create histogram of edge orientations (6 bins)
        hist, _ = np.histogram(angles, bins=6, range=(0, np.pi))
        normalized_hist = hist / (np.sum(hist) + 1e-10)
    else:
        normalized_hist = np.zeros(6)
    
    # Combine all layout features
    return np.concatenate([grid_features, normalized_hist])

def integrate_visual_pathways(visual_representation, oscillator_state):
    """Integrate multiple visual pathways into oscillator dynamics."""
    # Define frequency bands for different visual pathways
    pathway_bands = {
        "shape": (80, 120),     # High gamma for detail
        "motion": (25, 40),     # Beta/low gamma for motion
        "color": (8, 13),       # Alpha for feature binding
        "texture": (40, 80),    # Higher gamma for local features
        "layout": (4, 8)        # Theta for global context
    }
    
    # Integrate each pathway with appropriate oscillator frequency band
    for pathway, band_range in pathway_bands.items():
        # Get features for this pathway
        features = visual_representation[pathway]["features"]
        
        # Find oscillators in this frequency range
        low, high = band_range
        band_indices = np.where((low <= oscillator_state["frequencies"]) & 
                              (oscillator_state["frequencies"] < high))[0]
        
        if len(band_indices) == 0:
            continue
            
        # Map features to phase patterns in this band
        band_size = len(band_indices)
        feature_size = len(features)
        
        # Ensure feature dimensionality matches band size through interpolation
        if feature_size != band_size:
            # Simple linear interpolation
            indices = np.linspace(0, feature_size - 1, band_size).astype(int)
            mapped_features = features[indices]
        else:
            mapped_features = features
        
        # Scale features to phase shifts
        phase_shifts = mapped_features * np.pi  # Scale to [0, π]
        
        # Apply phase shifts to this band
        for i, idx in enumerate(band_indices):
            oscillator_state["phases"][idx] = (
                oscillator_state["phases"][idx] + phase_shifts[i]
            ) % (2 * np.pi)
    
    return oscillator_state

def bind_features_adaptively(oscillator_state, visual_representation, binding_threshold=0.75):
    """Dynamically bind features based on phase coherence across modalities."""

    # Extract feature-specific phase patterns
    shape_phases = visual_representation["shape"]["sdr"]
    motion_phases = visual_representation["motion"]["sdr"]
    color_phases = visual_representation["color"]["sdr"]

    # Compute mean phase coherence
    z_shape = np.mean(np.exp(1j * shape_phases))
    z_motion = np.mean(np.exp(1j * motion_phases))
    z_color = np.mean(np.exp(1j * color_phases))

    # Compute binding strength as alignment of phase vectors
    binding_strength = np.abs(z_shape * z_motion * z_color)

    # Reinforce binding if coherence exceeds threshold
    if binding_strength > binding_threshold:
        # Strengthen connections between synchronized oscillators
        for i in range(len(oscillator_state["phases"])):
            for j in range(len(oscillator_state["phases"])):
                if i != j:
                    phase_diff = np.abs((oscillator_state["phases"][i] - oscillator_state["phases"][j]) % (2 * np.pi))

                    # Strengthen coupling for aligned phases
                    if phase_diff < 0.5 or phase_diff > 5.8:
                        oscillator_state["amplitudes"][i] *= 1.1
                        oscillator_state["amplitudes"][j] *= 1.1

    return oscillator_state

"""
Instead of binding all sensory inputs immediately, allow the system to stabilize percepts over time.
This prevents hallucinated associations and ensures concepts persist stably across frames.
"""
def stabilize_perception(oscillator_state, percept_buffer, stabilization_rate=0.3):
    """Maintain percept stability over time by reinforcing persistent activations."""

    # Retrieve last percept from buffer
    if len(percept_buffer) > 0:
        previous_percept = percept_buffer[-1]
    else:
        return oscillator_state

    # Compute similarity between current and past percept
    similarity = np.dot(
        oscillator_state["phases"], previous_percept["phases"]
    ) / (np.linalg.norm(oscillator_state["phases"]) * np.linalg.norm(previous_percept["phases"]) + 1e-10)

    # Reinforce persistent activations
    if similarity > 0.7:
        oscillator_state["amplitudes"] *= (1 + stabilization_rate)

    # Store current percept in buffer
    percept_buffer.append(oscillator_state)

    return oscillator_state, percept_buffer

"""
Concept Learning Mechanism
    •    When a new combination of shape, motion, and color emerges, a new attractor forms
    •    If that combination repeats over time, the attractor solidifies into memory
    •    If the combination is inconsistent, it remains transient
"""
def learn_new_concepts(oscillator_state, concept_memory, min_reinforcement=5):
    """Extract new concepts from stable recurrent activations."""

    # Convert phase activations into a structured SDR
    sdr = compute_sdr_from_oscillator_state(oscillator_state)

    # Check if a similar concept already exists
    for concept in concept_memory:
        similarity = np.dot(sdr, concept["sdr"]) / (np.linalg.norm(sdr) * np.linalg.norm(concept["sdr"]) + 1e-10)

        if similarity > 0.85:
            # Strengthen existing concept
            concept["reinforcement"] += 1
            return concept_memory

    # If no similar concept exists, store a new one
    new_concept = {
        "sdr": sdr,
        "reinforcement": 1
    }

    concept_memory.append(new_concept)

    # Only keep reinforced concepts
    concept_memory = [c for c in concept_memory if c["reinforcement"] >= min_reinforcement]

    return concept_memory

def validate_percept_coherence(visual_modalities, oscillator_state, history_buffer, coherence_threshold=0.7):
    """
    Validate perceptual coherence across modalities before forming a concept.
    
    Parameters:
    -----------
    visual_modalities : dict
        Dictionary of visual processing streams (shape, motion, etc.)
    oscillator_state : dict
        Current oscillator state
    history_buffer : list
        Buffer of recent perceptual states
    coherence_threshold : float
        Minimum coherence required for binding
        
    Returns:
    --------
    is_coherent : bool
        Whether the percept is sufficiently coherent
    confidence : float
        Confidence score for the percept
    """
    # Compute coherence between each modality
    modality_pairs = [
        ("shape", "motion"),
        ("shape", "color"),
        ("motion", "color"),
        ("texture", "shape"),
        ("layout", "motion")
    ]
    
    pair_coherence = {}
    
    for mod1, mod2 in modality_pairs:
        if mod1 in visual_modalities and mod2 in visual_modalities:
            # Convert SDRs to phase space
            phases1 = visual_modalities[mod1]["sdr"]
            phases2 = visual_modalities[mod2]["sdr"]
            
            # Compute phase coherence between modalities
            coherence = compute_phase_coherence(phases1, phases2)
            pair_coherence[(mod1, mod2)] = coherence
    
    # Check temporal stability using history buffer
    temporal_stability = 0.0
    if history_buffer:
        # Compare current activation with recent history
        current_phases = oscillator_state["phases"]
        previous_phases = history_buffer[-1]["phases"]
        
        # Calculate phase consistency over time
        temporal_coherence = np.mean(np.cos(current_phases - previous_phases))
        temporal_stability = (temporal_coherence + 1) / 2  # Scale to [0,1]
    
    # Calculate overall coherence score
    if pair_coherence:
        cross_modal_coherence = np.mean(list(pair_coherence.values()))
        
        # Weight recent coherence more heavily
        coherence_score = 0.7 * cross_modal_coherence + 0.3 * temporal_stability
        
        is_coherent = coherence_score > coherence_threshold
        
        return is_coherent, coherence_score
    
    return False, 0.0

def process_visual_streams_parallel(image, previous_frames=None):
    """Process visual streams in parallel using multiple cores."""
    # Define visual processing functions for each stream
    stream_functions = {
        "shape": extract_shape_features,
        "motion": extract_motion_features,
        "color": extract_color_features,
        "texture": extract_texture_features,
        "layout": extract_layout_features
    }
    
    # Process streams in parallel
    from concurrent.futures import ProcessPoolExecutor
    
    results = {}
    with ProcessPoolExecutor(max_workers=len(stream_functions)) as executor:
        # Submit all processing tasks
        futures = {}
        for stream_name, stream_func in stream_functions.items():
            if stream_name == "motion" and previous_frames:
                futures[stream_name] = executor.submit(stream_func, image, previous_frames)
            else:
                futures[stream_name] = executor.submit(stream_func, image)
        
        # Collect results
        for stream_name, future in futures.items():
            feature_vector = future.result()
            sdr = compute_sdr_from_vector(feature_vector)
            results[stream_name] = {
                "features": feature_vector,
                "sdr": sdr
            }
    
    return results