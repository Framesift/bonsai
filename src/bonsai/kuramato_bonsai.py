from bonsai import initialize_coupling, initialize_oscillators, oscillator_state_to_vector, update_oscillators
from bonsai.perception import inject_perception


class KuramotoBonsai:
    def __init__(self, num_oscillators=256, **params):
        self.state = initialize_oscillators(num_oscillators, **params)
        self.coupling = initialize_coupling(num_oscillators, **params)
        self.params = params
    
    def update(self, steps=10):
        self.state = update_oscillators(self.state, self.coupling, steps=steps)
    
    def inject_perception(self, features, strength=0.5):
        self.state = inject_perception(self.state, features, strength)
    
    def get_thought_vector(self):
        return oscillator_state_to_vector(self.state)