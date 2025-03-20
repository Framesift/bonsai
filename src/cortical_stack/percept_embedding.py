class PerceptEmbedding:
    """Embedding vector combining visual, text and audio percepts.
    
    Structure:
    - dims[0:256]: Visual embedding from ResNet50
    - dims[256:512]: Text embedding from BERT
    - dims[512:1024]: Audio embedding from Wav2Vec
    """
    def __init__(self, data):
        self.data = data  # The actual numpy array
        assert data.shape == (1024,), f"Expected 1024D vector, got {data.shape}"
    
    # Delegate to the numpy array for most operations
    def __array__(self):
        return self.data
    
    @property
    def visual(self):
        return self.data[0:256]
    
    @property
    def text(self):
        return self.data[256:512]
    
    @property
    def audio(self):
        return self.data[512:1024]