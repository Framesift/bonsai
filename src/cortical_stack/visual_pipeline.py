from dataclasses import dataclass
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
from PIL import Image
import io
from functools import partial
import panel as pn
import holoviews as hv
hv.extension('bokeh')

@dataclass
class PerceptVector:
    """Container for feature vectors extracted from visual perception pathways"""
    data: xr.DataArray
    
    @classmethod
    def create(cls, vectors, pathway_names=None, feature_names=None):
        """Create a PerceptVector from raw vectors
        
        Parameters
        ----------
        vectors : np.ndarray
            Array of shape (n_pathways, n_features) or dict mapping pathway names to feature vectors
        pathway_names : list, optional
            Names of the visual pathways
        feature_names : list, optional
            Names of the features in each vector
        """
        if isinstance(vectors, dict):
            # If vectors is a dictionary, extract pathway names and arrays
            pathway_names = list(vectors.keys())
            vectors_array = np.array([vectors[path] for path in pathway_names])
        else:
            vectors_array = np.asarray(vectors)
            
            # Default pathway names if not provided
            if pathway_names is None:
                pathway_names = [
                    f"pathway_{i}" for i in range(vectors_array.shape[0])
                ]
        
        # Default feature names if not provided
        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(vectors_array.shape[1])
            ]
        
        # Create xarray DataArray with semantic coordinates
        data_array = xr.DataArray(
            vectors_array,
            dims=['pathway', 'feature'],
            coords={
                'pathway': pathway_names,
                'feature': feature_names
            }
        )
        
        return cls(data_array)
    
    def get_pathway(self, pathway_name):
        """Get feature vector for a specific pathway"""
        return self.data.sel(pathway=pathway_name).values
    
    def get_feature(self, feature_name):
        """Get values for a specific feature across all pathways"""
        return self.data.sel(feature=feature_name).values
    
    def concatenate(self, *others):
        """Concatenate multiple PerceptVectors along the feature dimension"""
        arrays = [self.data] + [other.data for other in others]
        
        # Make sure all arrays have the same pathways
        common_pathways = set(self.data.pathway.values)
        for arr in arrays[1:]:
            common_pathways &= set(arr.pathway.values)
        
        # Select only common pathways
        aligned_arrays = [arr.sel(pathway=list(common_pathways)) for arr in arrays]
        
        # Concatenate along feature dimension
        concatenated = xr.concat(aligned_arrays, dim='feature')
        
        return PerceptVector(concatenated)
    
    def normalize(self):
        """Normalize feature vectors to unit length"""
        norm = np.sqrt((self.data**2).sum(dim='feature'))
        normalized = self.data / norm
        
        return PerceptVector(normalized)


@dataclass
class FeatureVisualizer:
    """Visualizer for feature extractions from visual pathways"""
    
    @staticmethod
    def visualize_frame_with_features(frame, percept_vector, pathway_names=None, 
                                     feature_map=None, figsize=(15, 10)):
        """
        Visualize original frame alongside feature extractions
        
        Parameters
        ----------
        frame : np.ndarray
            Original image/video frame
        percept_vector : PerceptVector
            Feature vectors extracted from the frame
        pathway_names : list, optional
            Pathways to visualize (defaults to all)
        feature_map : dict, optional
            Maps specific features to visualization functions
        figsize : tuple, optional
            Figure size
        """
        if pathway_names is None:
            pathway_names = percept_vector.data.pathway.values
        
        n_pathways = len(pathway_names)
        
        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_pathways + 1, 2, width_ratios=[1, 3], height_ratios=[1] + [1] * n_pathways)
        
        # Original frame in the top left
        ax_frame = fig.add_subplot(gs[0, 0])
        ax_frame.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_frame.set_title("Original Frame")
        ax_frame.axis('off')
        
        # Feature distribution overview in the top right
        ax_overview = fig.add_subplot(gs[0, 1])
        pathways_to_plot = []
        labels = []
        
        for i, pathway in enumerate(pathway_names):
            features = percept_vector.get_pathway(pathway)
            pathways_to_plot.append(features)
            labels.append(pathway)
        
        # Create violin plots for feature distributions
        ax_overview.violinplot(pathways_to_plot, showmeans=True, showmedians=True)
        ax_overview.set_xticks(range(1, len(labels) + 1))
        ax_overview.set_xticklabels(labels, rotation=45)
        ax_overview.set_title("Feature Distribution by Pathway")
        ax_overview.set_ylabel("Feature Activation")
        
        # Individual pathway visualizations
        for i, pathway in enumerate(pathway_names):
            pathway_data = percept_vector.data.sel(pathway=pathway)
            
            # Pathway features as heatmap
            ax_pathway = fig.add_subplot(gs[i + 1, 0])
            im = ax_pathway.imshow(
                pathway_data.values.reshape(1, -1), 
                aspect='auto', 
                cmap='viridis'
            )
            ax_pathway.set_title(f"{pathway} Features")
            ax_pathway.set_yticks([])
            
            # Add colorbar
            plt.colorbar(im, ax=ax_pathway)
            
            # Custom feature visualization if provided
            ax_custom = fig.add_subplot(gs[i + 1, 1])
            
            if feature_map and pathway in feature_map:
                # Use custom visualization function
                feature_map[pathway](frame, pathway_data.values, ax_custom)
            else:
                # Default visualization: bar chart of feature values
                ax_custom.bar(
                    range(len(pathway_data)), 
                    pathway_data.values, 
                    alpha=0.7
                )
                ax_custom.set_title(f"{pathway} Feature Values")
                ax_custom.set_xlabel("Feature Index")
                ax_custom.set_ylabel("Activation")
                
                # Highlight top activated features
                top_idx = np.argsort(pathway_data.values)[-5:]  # Top 5 features
                for idx in top_idx:
                    ax_custom.get_children()[idx].set_color('red')
                
                # Add feature names as x-tick labels if there aren't too many
                if len(pathway_data) <= 20:
                    ax_custom.set_xticks(range(len(pathway_data)))
                    ax_custom.set_xticklabels(
                        pathway_data.feature.values, 
                        rotation=90
                    )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_interactive_comparison(frames, percept_vectors, pathway_names=None):
        """
        Create an interactive dashboard to compare feature extractions across frames
        
        Parameters
        ----------
        frames : list
            List of image/video frames
        percept_vectors : list
            List of PerceptVector objects, one per frame
        pathway_names : list, optional
            Pathways to visualize (defaults to all in first percept_vector)
        """
        if pathway_names is None:
            pathway_names = percept_vectors[0].data.pathway.values
        
        # Create selection widgets
        frame_slider = pn.widgets.IntSlider(
            name="Frame", 
            start=0, 
            end=len(frames)-1, 
            value=0
        )
        
        pathway_select = pn.widgets.MultiSelect(
            name="Pathways",
            options=list(pathway_names),
            value=list(pathway_names)[:2]  # Default to first two pathways
        )
        
        feature_slider = pn.widgets.RangeSlider(
            name="Feature Range",
            start=0,
            end=len(percept_vectors[0].data.feature)-1,
            value=(0, min(20, len(percept_vectors[0].data.feature)-1))
        )
        
        # Define plot generation function
        def generate_plots(frame_idx, selected_pathways, feature_range):
            frame = frames[frame_idx]
            vector = percept_vectors[frame_idx]
            
            # Original frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_plot = hv.RGB(frame_rgb, bounds=(0, 0, frame.shape[1], frame.shape[0]))
            
            # Feature matrices
            feature_plots = []
            for pathway in selected_pathways:
                # Get selected feature range
                pathway_data = vector.data.sel(
                    pathway=pathway,
                    feature=slice(feature_range[0], feature_range[1]+1)
                )
                
                # Create heatmap
                heatmap = hv.HeatMap(
                    (pathway_data.feature.values, ['value'], pathway_data.values.reshape(-1, 1)),
                    kdims=['Feature', 'Type'],
                    vdims='Activation'
                ).opts(
                    cmap='viridis',
                    colorbar=True,
                    width=800,
                    height=100,
                    title=f"{pathway} Features"
                )
                
                # Bar chart
                bars = hv.Bars(
                    (pathway_data.feature.values, pathway_data.values),
                    kdims=['Feature'],
                    vdims=['Activation']
                ).opts(
                    width=800,
                    height=300,
                    color='steelblue',
                    title=f"{pathway} Feature Values"
                )
                
                feature_plots.append(heatmap)
                feature_plots.append(bars)
            
            # Combine plots
            combined = pn.Column(
                pn.pane.Markdown(f"# Frame {frame_idx}"),
                frame_plot.opts(width=400, height=300, title="Original Frame"),
                *feature_plots
            )
            
            return combined
        
        # Connect widgets to plot function
        dashboard = pn.bind(
            generate_plots, 
            frame_idx=frame_slider,
            selected_pathways=pathway_select,
            feature_range=feature_slider
        )
        
        # Create layout
        app = pn.Column(
            pn.Row(frame_slider, pathway_select, feature_slider),
            dashboard
        )
        
        return app
    
    @staticmethod
    def compare_pathways_across_frames(frames, percept_vectors, pathways_to_compare, 
                                      features_to_track=None):
        """
        Compare specific pathways across multiple frames
        
        Parameters
        ----------
        frames : list
            List of image/video frames
        percept_vectors : list
            List of PerceptVector objects, one per frame
        pathways_to_compare : list
            Pathways to compare
        features_to_track : list, optional
            Specific features to track (defaults to all)
        """
        n_frames = len(frames)
        n_pathways = len(pathways_to_compare)
        
        # Create figure
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, n_pathways + 1, height_ratios=[1, 3])
        
        # Show first and last frame
        ax_first = fig.add_subplot(gs[0, 0])
        ax_first.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        ax_first.set_title("First Frame")
        ax_first.axis('off')
        
        ax_last = fig.add_subplot(gs[0, -1])
        ax_last.imshow(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
        ax_last.set_title("Last Frame")
        ax_last.axis('off')
        
        # Plot pathway evolution
        for i, pathway in enumerate(pathways_to_compare):
            ax = fig.add_subplot(gs[1, i])
            
            # Collect data across frames
            if features_to_track:
                # Track specific features
                data = np.array([
                    vec.data.sel(pathway=pathway, feature=features_to_track).values 
                    for vec in percept_vectors
                ])
                feature_names = features_to_track
            else:
                # Track all features
                data = np.array([
                    vec.data.sel(pathway=pathway).values 
                    for vec in percept_vectors
                ])
                feature_names = percept_vectors[0].data.feature.values
            
            # Create heatmap of feature evolution
            im = ax.imshow(
                data.T, 
                aspect='auto', 
                cmap='viridis',
                extent=[0, n_frames-1, 0, len(feature_names)]
            )
            
            ax.set_title(f"{pathway} Evolution")
            ax.set_xlabel("Frame")
            
            # Add feature names if there aren't too many
            if len(feature_names) <= 10:
                ax.set_yticks(np.arange(len(feature_names)) + 0.5)
                ax.set_yticklabels(feature_names)
            else:
                ax.set_ylabel("Feature Index")
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        # Plot overall pathway similarity
        ax_sim = fig.add_subplot(gs[1, -1])
        
        # Calculate cosine similarity between consecutive frames for each pathway
        similarities = {}
        for pathway in pathways_to_compare:
            pathway_similarities = []
            
            for i in range(1, n_frames):
                vec1 = percept_vectors[i-1].get_pathway(pathway)
                vec2 = percept_vectors[i].get_pathway(pathway)
                
                # Normalize vectors
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                
                # Calculate cosine similarity
                similarity = np.dot(vec1_norm, vec2_norm)
                pathway_similarities.append(similarity)
            
            similarities[pathway] = pathway_similarities
        
        # Plot similarities
        for pathway, sim_values in similarities.items():
            ax_sim.plot(
                range(1, n_frames), 
                sim_values, 
                label=pathway
            )
        
        ax_sim.set_title("Pathway Stability")
        ax_sim.set_xlabel("Frame")
        ax_sim.set_ylabel("Cosine Similarity")
        ax_sim.set_ylim(0, 1.05)
        ax_sim.legend()
        ax_sim.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_animated_feature_map(frames, percept_vectors, pathway, feature_indices, 
                                   frame_interval=100):
        """
        Create an animation showing how specific features evolve over frames
        
        Parameters
        ----------
        frames : list
            List of image/video frames
        percept_vectors : list
            List of PerceptVector objects, one per frame
        pathway : str
            Pathway to visualize
        feature_indices : list
            Indices of features to track
        frame_interval : int, optional
            Interval between frames in milliseconds
        """
        # Prepare figure
        fig, (ax_frame, ax_features) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Function to update the plot for each frame
        def update(frame_idx):
            ax_frame.clear()
            ax_features.clear()
            
            # Show frame
            ax_frame.imshow(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB))
            ax_frame.set_title(f"Frame {frame_idx}")
            ax_frame.axis('off')
            
            # Get feature values
            feature_values = percept_vectors[frame_idx].data.sel(
                pathway=pathway,
                feature=[percept_vectors[0].data.feature.values[i] for i in feature_indices]
            ).values
            
            feature_names = [percept_vectors[0].data.feature.values[i] for i in feature_indices]
            
            # Plot feature values
            bars = ax_features.bar(
                feature_names, 
                feature_values, 
                color='steelblue'
            )
            
            # Highlight max activated feature
            max_idx = np.argmax(feature_values)
            bars[max_idx].set_color('red')
            
            ax_features.set_title(f"{pathway} Selected Features")
            ax_features.set_ylim(0, max([vec.data.sel(pathway=pathway).max().values 
                                        for vec in percept_vectors]) * 1.1)
            ax_features.tick_params(axis='x', rotation=45)
            
            return (ax_frame, ax_features)
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=frame_interval, blit=False
        )
        
        plt.tight_layout()
        return ani


# Demo function to create test data
def create_test_data(n_frames=20):
    """Create test frames and percept vectors"""
    # Create synthetic frames
    frames = []
    for i in range(n_frames):
        # Create gradient frame with moving circle
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(300):
            for x in range(400):
                frame[y, x, 0] = int(255 * (x / 400))  # R
                frame[y, x, 1] = int(255 * (y / 300))  # G
                frame[y, x, 2] = int(255 * ((x+y) / 700))  # B
        
        # Add moving circle
        center_x = int(200 + 150 * np.sin(i / 10 * np.pi))
        center_y = 150
        cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)
        
        frames.append(frame)
    
    # Create synthetic percept vectors
    percept_vectors = []
    
    pathway_names = ['ventral', 'dorsal', 'color', 'motion']
    feature_counts = {'ventral': 64, 'dorsal': 48, 'color': 32, 'motion': 24}
    
    # Generate feature names
    feature_names = {}
    for pathway, count in feature_counts.items():
        if pathway == 'ventral':
            feature_names[pathway] = [f"shape_{i}" for i in range(count)]
        elif pathway == 'dorsal':
            feature_names[pathway] = [f"spatial_{i}" for i in range(count)]
        elif pathway == 'color':
            feature_names[pathway] = [f"color_{i}" for i in range(count)]
        elif pathway == 'motion':
            feature_names[pathway] = [f"motion_{i}" for i in range(count)]
    
    for i in range(n_frames):
        # Create vectors for each pathway with different patterns
        vectors = {}
        
        # Ventral pathway: shape features (respond to circle presence)
        circle_x = int(200 + 150 * np.sin(i / 10 * np.pi)) / 400  # Normalized position
        ventral = np.zeros(feature_counts['ventral'])
        for j in range(feature_counts['ventral']):
            # Some features activate based on circle position
            if j < 20:
                ventral[j] = np.exp(-(circle_x - j/20)**2 / 0.1)
            else:
                ventral[j] = np.random.normal(0.2, 0.1)
        vectors['ventral'] = ventral
        
        # Dorsal pathway: spatial features
        dorsal = np.zeros(feature_counts['dorsal'])
        for j in range(feature_counts['dorsal']):
            if j < 10:
                # Features tracking position
                dorsal[j] = np.abs(circle_x - j/10)
            else:
                dorsal[j] = np.random.normal(0.3, 0.1)
        vectors['dorsal'] = dorsal
        
        # Color pathway
        color = np.zeros(feature_counts['color'])
        for j in range(feature_counts['color']):
            if j < 10:
                # White detection (for the circle)
                color[j] = 0.8 if 0.3 < circle_x < 0.7 else 0.2
            else:
                # Background colors
                color[j] = np.random.normal(0.4, 0.1)
        vectors['color'] = color
        
        # Motion pathway
        motion = np.zeros(feature_counts['motion'])
        for j in range(feature_counts['motion']):
            if j < 8:
                # Motion detection based on circle velocity
                if i > 0:
                    prev_x = int(200 + 150 * np.sin((i-1) / 10 * np.pi)) / 400
                    motion[j] = np.abs(circle_x - prev_x) * 10 * (j+1)/8
                else:
                    motion[j] = 0
            else:
                motion[j] = np.random.normal(0.1, 0.1)
        vectors['motion'] = motion
        
        # Create PerceptVector with all pathways
        percept_vector = PerceptVector.create(
            vectors,
            pathway_names=list(vectors.keys()),
            feature_names={pathway: feature_names[pathway] for pathway in vectors}
        )
        
        percept_vectors.append(percept_vector)
    
    return frames, percept_vectors

# Example of a custom visualization function for a feature map
def visualize_ventral_features(frame, features, ax):
    """Custom visualization for ventral pathway features"""
    # Find circle center based on highest activations
    max_feature = np.argmax(features[:20])
    estimated_x = int((max_feature / 20) * 400)
    
    # Display frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Add rectangle around estimated object location
    rect = Rectangle((estimated_x-50, 100), 100, 100, 
                    linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_title("Ventral Pathway: Object Detection")
    ax.axis('off')

# Example of a custom visualization function for motion pathway
def visualize_motion_features(frame, features, ax):
    """Custom visualization for motion pathway features"""
    # Calculate motion magnitude and direction
    motion_magnitude = np.sum(features[:8])
    
    # Display frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Add motion vectors
    if motion_magnitude > 0.5:
        # Strong motion detected
        center_x = 200
        center_y = 150
        motion_angle = np.pi * 0.5  # Assume horizontal motion
        
        # Draw arrow indicating motion
        arrow_length = motion_magnitude * 50
        dx = arrow_length * np.cos(motion_angle)
        dy = arrow_length * np.sin(motion_angle)
        
        ax.arrow(center_x, center_y, dx, dy, 
                head_width=20, head_length=20, 
                fc='y', ec='y', linewidth=2)
    
    ax.set_title(f"Motion Pathway: Magnitude {motion_magnitude:.2f}")
    ax.axis('off')