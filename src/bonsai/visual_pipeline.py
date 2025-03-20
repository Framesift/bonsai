import cv2
import numpy as np

from bonsai import initialize_coupling, initialize_oscillators, update_oscillators
from bonsai.perception import inject_perception
from bonsai.visual_pathways import extract_color_features, extract_layout_features, extract_motion_features, extract_shape_features, extract_texture_features


def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess raw image input.
    
    Parameters:
    -----------
    image : ndarray
        Raw input image (grayscale or RGB)
    target_size : tuple
        Target dimensions for normalization
        
    Returns:
    --------
    processed : ndarray
        Normalized grayscale image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize to target dimensions
    return cv2.resize(gray, target_size)

def extract_edges(image):
    """
    Extract edge information from image.
    
    Parameters:
    -----------
    image : ndarray
        Preprocessed grayscale image
        
    Returns:
    --------
    edges : ndarray
        Edge map
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Detect edges
    return cv2.Canny(blurred, 50, 150)

def compute_shape_dna(binary_image, n_eigenvalues=20):
    """
    Compute Shape DNA (Laplacian eigenvalues).
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary image of shape
    n_eigenvalues : int
        Number of eigenvalues to compute
        
    Returns:
    --------
    eigenvalues : ndarray
        Shape DNA eigenvalues
    """
    # Create graph representation
    h, w = binary_image.shape
    
    # Create adjacency matrix for non-zero pixels
    indices = np.where(binary_image > 0)
    points = list(zip(indices[0], indices[1]))
    n_points = len(points)
    
    if n_points < 5:  # Not enough points for meaningful analysis
        return np.zeros(n_eigenvalues)
    
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Create sparse adjacency matrix
    rows, cols, data = [], [], []
    
    # Connect each point to its neighbors
    for y, x in points:
        idx = point_to_idx[(y, x)]
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if (ny, nx) in point_to_idx:
                neighbor_idx = point_to_idx[(ny, nx)]
                rows.append(idx)
                cols.append(neighbor_idx)
                data.append(1)
    
    # Create sparse adjacency matrix
    import scipy.sparse as sp
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    # Compute Laplacian
    degree = sp.diags(adjacency.sum(axis=1).A1)
    laplacian = degree - adjacency
    
    # Compute eigenvalues
    from scipy.sparse.linalg import eigsh
    try:
        eigenvalues, _ = eigsh(laplacian, k=min(n_eigenvalues, n_points-1), which='SM')
    except:
        # Fallback to dense computation if sparse fails
        lap_dense = laplacian.toarray()
        eigenvalues = np.sort(np.linalg.eigvalsh(lap_dense))[:n_eigenvalues]
    
    # Pad if needed
    if len(eigenvalues) < n_eigenvalues:
        eigenvalues = np.pad(eigenvalues, (0, n_eigenvalues - len(eigenvalues)))
    
    # Normalize
    return eigenvalues / (np.sum(eigenvalues) + 1e-10)

def compute_shape_moments(binary_image):
    """
    Compute shape moments.
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary image of shape
        
    Returns:
    --------
    moments : ndarray
        Normalized Hu moments
    """
    # Compute moments
    moments = cv2.moments(binary_image)
    
    # Compute Hu moments (rotation/scale/translation invariant)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform to reduce dynamic range
    log_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return log_moments

def compute_fourier_descriptors(binary_image, n_descriptors=10):
    """
    Compute Fourier descriptors of shape boundary.
    
    Parameters:
    -----------
    binary_image : ndarray
        Binary image of shape
    n_descriptors : int
        Number of descriptors to compute
        
    Returns:
    --------
    descriptors : ndarray
        Normalized Fourier descriptors
    """
    # Find contours
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros(n_descriptors)
    
    # Get largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Convert to complex representation
    contour = contour.reshape(-1, 2)
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    
    # Compute Fourier descriptors
    fourier_desc = np.fft.fft(complex_contour)
    
    # Normalize by DC component to achieve translation invariance
    fourier_desc = fourier_desc / (fourier_desc[0] + 1e-10)
    fourier_desc = fourier_desc[1:n_descriptors+1]  # Skip DC component
    
    # Use magnitude for rotation invariance
    descriptors = np.abs(fourier_desc)
    
    # Normalize for scale invariance
    return descriptors / (np.linalg.norm(descriptors) + 1e-10)

def multiscale_analysis(image, scales=(1.0, 0.75, 0.5, 0.25)):
    """
    Perform multi-scale feature analysis.
    
    Parameters:
    -----------
    image : ndarray
        Preprocessed image
    scales : tuple
        Scales to analyze at
        
    Returns:
    --------
    features : ndarray
        Multi-scale features
    """
    features = []
    
    for scale in scales:
        # Resize image to current scale
        if scale != 1.0:
            h, w = image.shape
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
        else:
            scaled = image
        
        # Extract edges
        edges = extract_edges(scaled)
        
        # Compute histogram of oriented gradients
        gx = cv2.Sobel(scaled, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(scaled, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Compute gradient histogram (8 bins)
        hist, _ = np.histogram(ang, bins=8, range=(0, 2*np.pi), weights=mag)
        hist = hist / (np.sum(hist) + 1e-10)  # Normalize
        
        features.extend(hist)
    
    return np.array(features)

def extract_feature_vector(image, perception_dim=64):
    """
    Complete pipeline to extract feature vector from image.
    
    Parameters:
    -----------
    image : ndarray
        Raw input image
    perception_dim : int
        Dimensionality of output feature vector
        
    Returns:
    --------
    feature_vector : ndarray
        Normalized feature vector
    """
    # Preprocess image
    processed = preprocess_image(image)
    
    # Extract edges
    edges = extract_edges(processed)
    
    # Extract features from multiple sources
    shape_dna = compute_shape_dna(edges)
    moments = compute_shape_moments(edges)
    fourier_desc = compute_fourier_descriptors(edges)
    multiscale_features = multiscale_analysis(processed)
    
    # Combine features
    combined = np.concatenate([
        shape_dna,
        moments,
        fourier_desc,
        multiscale_features
    ])
    
    # If dimensionality doesn't match perception_dim, adjust
    if len(combined) > perception_dim:
        # Select most important features
        combined = combined[:perception_dim]
    elif len(combined) < perception_dim:
        # Pad with zeros
        combined = np.pad(combined, (0, perception_dim - len(combined)))
    
    # Normalize
    return combined / (np.linalg.norm(combined) + 1e-10)

def process_shape(image):
    """Process a shape image into oscillator space."""
    # Extract feature vector
    features = extract_feature_vector(image)
    
    # Initialize oscillator system
    oscillator_state = initialize_oscillators(num_oscillators=256, strategy="harmonic")
    coupling = initialize_coupling(num_oscillators=256, strategy="connectome")
    
    # Inject perception
    oscillator_state = inject_perception(oscillator_state, features, strength=0.8)
    
    # Update dynamics
    for _ in range(5):
        oscillator_state = update_oscillators(oscillator_state, coupling, steps=10)
    
    return oscillator_state

def process_visual_multimodal(image):
    """Process visual input through multiple parallel pathways."""
    
    # 1. Shape/Form pathway (ventral "what" stream)
    shape_features = extract_shape_features(image)
    shape_sdr = compute_sdr_from_vector(shape_features)
    
    # 2. Motion/Location pathway (dorsal "where/how" stream)
    motion_features = extract_motion_features(image)
    motion_sdr = compute_sdr_from_vector(motion_features)
    
    # 3. Color processing stream
    color_features = extract_color_features(image)
    color_sdr = compute_sdr_from_vector(color_features)
    
    # 4. Texture analysis pathway 
    texture_features = extract_texture_features(image)
    texture_sdr = compute_sdr_from_vector(texture_features)
    
    # 5. Global layout/scene recognition
    layout_features = extract_layout_features(image)
    layout_sdr = compute_sdr_from_vector(layout_features)
    
    # Integrate these pathways with appropriate weighting
    visual_representation = {
        "shape": {"features": shape_features, "sdr": shape_sdr},
        "motion": {"features": motion_features, "sdr": motion_sdr},
        "color": {"features": color_features, "sdr": color_sdr},
        "texture": {"features": texture_features, "sdr": texture_sdr},
        "layout": {"features": layout_features, "sdr": layout_sdr},
    }
    
    return visual_representation