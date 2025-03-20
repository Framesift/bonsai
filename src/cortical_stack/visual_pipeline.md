
```python
# Create test data
frames, percept_vectors = create_test_data(n_frames=30)

# Define feature map with custom visualizations
feature_map = {
    'ventral': visualize_ventral_features,
    'motion': visualize_motion_features
}

# 1. Visualize a single frame with its feature extractions
fig = FeatureVisualizer.visualize_frame_with_features(
    frames[10], 
    percept_vectors[10],
    pathway_names=['ventral', 'dorsal', 'motion'],
    feature_map=feature_map
)
plt.savefig("feature_extraction.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Compare pathways across frames
comparison_fig = FeatureVisualizer.compare_pathways_across_frames(
    frames,
    percept_vectors,
    pathways_to_compare=['ventral', 'motion'],
    features_to_track=None  # Track all features
)
plt.savefig("pathway_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Create an animated feature map
ani = FeatureVisualizer.create_animated_feature_map(
    frames, 
    percept_vectors,
    pathway='motion',
    feature_indices=[0, 1, 2, 3],
    frame_interval=200
)
# Save animation
ani.save('feature_animation.mp4', writer='ffmpeg', dpi=300)

# 4. Create interactive comparison dashboard
dashboard = FeatureVisualizer.create_interactive_comparison(
    frames, 
    percept_vectors
)
# The dashboard can be displayed in a Jupyter notebook with: display(dashboard)
# Or served as a web app with: dashboard.servable()
```

This framework provides several key benefits for your cortical stack modeling:

Semantic representation - Using xarray's labeled dimensions keeps track of what each feature represents
Multi-pathway visualization - Easily compare outputs from different cortical pathways side by side
Temporal analysis - Track how features evolve across frames, essential for understanding motion processing
Customizable visualizations - Map specific features to meaningful visual representations (like highlighting object locations)
Interactive exploration - The dashboard allows you to drill down into specific pathways and features
Functional approach - Pure functions maintain clean code organization without shared mutable state

For real-world use, you would integrate this with your actual feature extraction pipeline:

```python
# Example integration with a real pipeline
def process_video(video_path, frame_limit=100):
    """Process video through cortical stack model"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    percept_vectors = []
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Run frame through your cortical pathways
        ventral_features = extract_ventral_features(frame)
        dorsal_features = extract_dorsal_features(frame)
        color_features = extract_color_features(frame)
        motion_features = extract_motion_features(frame, 
                                                 prev_frame=frames[-2] if len(frames) > 1 else None)
        
        # Create feature dictionary
        vectors = {
            'ventral': ventral_features,
            'dorsal': dorsal_features,
            'color': color_features,
            'motion': motion_features
        }
        
        # Create PerceptVector
        percept_vector = PerceptVector.create(vectors)
        percept_vectors.append(percept_vector)
        
        frame_count += 1
    
    cap.release()
    
    return frames, percept_vectors
    ```

    