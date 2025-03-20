# Initialize system
from bonsai import initialize_coupling, initialize_oscillators, oscillator_state_to_vector, update_oscillators
from bonsai.perception import inject_perception
from bonsai.visual_pathways import extract_shape_features

oscillator_state = initialize_oscillators(
    num_oscillators=256,
    strategy="harmonic",
    bands=[(0, 8), (8, 13), (13, 30), (30, 80), (80, 150)]
)

coupling = initialize_coupling(
    num_oscillators=256,
    strategy="connectome",
    modules=5,
    within_strength=0.15,
    between_strength=0.05
)

# Process perception
features = extract_shape_features(image)

# Inject perception
oscillator_state = inject_perception(oscillator_state, features, strength=0.8)

# Update dynamics
for _ in range(5):
    oscillator_state = update_oscillators(
        oscillator_state, 
        coupling,
        strategy="vectorized",
        steps=10
    )

# Extract representation
thought_vector = oscillator_state_to_vector(oscillator_state)

