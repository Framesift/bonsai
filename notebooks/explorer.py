import ipywidgets as widgets
from IPython.display import display
import matplotlib as plt
import numpy as np

from src.cortical_stack.analyze import calculate_order_parameter

def create_interactive_explorer(time_series):
    """Create an interactive widget to explore the simulation results"""
    # Time slider
    time_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(time_series.time)-1,
        step=1,
        description='Time:',
        continuous_update=False
    )
    
    # Band selector
    bands = np.unique(time_series.band.values)
    band_selector = widgets.SelectMultiple(
        options=list(bands),
        value=[bands[0]],
        description='Bands:',
        disabled=False
    )
    
    # Plot type selector
    plot_selector = widgets.RadioButtons(
        options=['Phase distribution', 'Time series', 'Order parameter'],
        description='Plot type:',
        disabled=False
    )
    
    # Output widget
    output = widgets.Output()
    
    # Update function
    def update_plot(*args):
        with output:
            output.clear_output(wait=True)
            
            if plot_selector.value == 'Phase distribution':
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='polar')
                
                selected_time = time_series.isel(time=time_slider.value)
                
                for band in band_selector.value:
                    band_phases = selected_time.where(selected_time.band == band, drop=True)
                    ax.scatter(
                        band_phases.values,
                        np.ones(len(band_phases)),
                        label=band,
                        alpha=0.7,
                        s=100
                    )
                
                ax.set_rticks([])
                ax.set_title(f'Phase distribution at t={time_slider.value}')
                ax.legend()
                
            elif plot_selector.value == 'Time series':
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for band in band_selector.value:
                    band_data = time_series.where(time_series.band == band, drop=True)
                    
                    # Plot all oscillators in this band
                    for i in range(len(band_data.oscillator)):
                        ax.plot(band_data.isel(oscillator=i).values, 
                                alpha=0.3, color=f'C{list(bands).index(band)}')
                    
                    # Highlight current time
                    ax.axvline(x=time_slider.value, color='red', linestyle='--')
                
                ax.set_xlabel('Time step')
                ax.set_ylabel('Phase')
                ax.set_title('Phase time series by band')
                
                # Create custom legend entries
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color=f'C{list(bands).index(band)}', lw=2) 
                               for band in band_selector.value]
                ax.legend(custom_lines, band_selector.value)
                
            elif plot_selector.value == 'Order parameter':
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Calculate order parameters for selected bands
                for band in band_selector.value:
                    band_data = time_series.where(time_series.band == band, drop=True)
                    order = calculate_order_parameter(band_data.values)
                    ax.plot(order, label=band)
                
                # Highlight current time
                ax.axvline(x=time_slider.value, color='red', linestyle='--')
                
                ax.set_xlabel('Time step')
                ax.set_ylabel('Order parameter (r)')
                ax.set_title('Synchronization by band')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # Connect widgets to update function
    time_slider.observe(update_plot, names='value')
    band_selector.observe(update_plot, names='value')
    plot_selector.observe(update_plot, names='value')
    
    # Initial plot
    update_plot()
    
    # Create layout
    controls = widgets.VBox([plot_selector, time_slider, band_selector])
    explorer = widgets.HBox([controls, output])
    
    return explorer

# Create and display the interactive explorer
explorer = create_interactive_explorer(time_series)
display(explorer)