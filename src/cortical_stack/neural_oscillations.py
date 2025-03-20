from dataclasses import dataclass
import numpy as np
import xarray as xr

@dataclass
class NeuralOscillations:
    """Container for frequency band data organized by neural oscillation types"""
    data: xr.DataArray
    
    @classmethod
    def from_array(cls, array, band_names=None):
        """Create from a raw array of [[x, y], ...] frequency bands"""
        if band_names is None:
            # Default neural oscillation band names
            band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        # Ensure we have the right number of bands
        assert len(array) == len(band_names), f"Expected {len(band_names)} bands, got {len(array)}"
        
        # Create xarray DataArray with semantic coordinates
        data_array = xr.DataArray(
            array,
            dims=['band', 'range_bound'],
            coords={
                'band': band_names,
                'range_bound': ['min_freq', 'max_freq']
            }
        )
        
        return cls(data_array)
    
    def get_band(self, band_name):
        """Get frequency range for a specific oscillation band"""
        return self.data.sel(band=band_name).values
    
    def within_band(self, frequency, band_name):
        """Check if a frequency is within a specific band"""
        band_range = self.get_band(band_name)
        return band_range[0] <= frequency <= band_range[1]
    
    def classify_frequency(self, frequency):
        """Determine which oscillation band a frequency belongs to"""
        for band in self.data.band.values:
            if self.within_band(frequency, band):
                return band
        return None
    
if (__name__ == "__main__"):
    # Standard frequency bands in Hz
    bands = np.array([
        [0.5, 4],    # Delta
        [4, 8],      # Theta
        [8, 13],     # Alpha
        [13, 30],    # Beta
        [30, 100]    # Gamma
    ])

    oscillations = NeuralOscillations.from_array(bands)

    # Get a specific band range
    print(f"Alpha band: {oscillations.get_band('alpha')} Hz")

    # Slice multiple bands
    print(oscillations.data.sel(band=['alpha', 'beta']))

    # Check what band a frequency belongs to
    print(f"10 Hz belongs to: {oscillations.classify_frequency(10)}")

    # Access as a pandas DataFrame for further analysis
    df = oscillations.data.to_dataframe().unstack()