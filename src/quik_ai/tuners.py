import keras_tuner as kt

class TunerContainer:
    def __init__(self, tuner, **kwargs):
        self.tuner = tuner
        self.kwargs = kwargs
        
    def get_tuner_params(self):
        return self.kwargs.copy()