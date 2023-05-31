from quik_ai import tuning

import numpy as np
import tensorflow as tf

class Predictor(tuning.Tunable):
    def __init__(self, name, numeric=True, drop=False, **kwargs):
        super().__init__(name, **kwargs)
        self.numeric = numeric
        self.drop = drop
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'drop' : self._get_hp(None, 'drop', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        if self.get_parameters(hp)['drop']:
            return None
        
        return inputs
    
class NumericalPredictor(Predictor):
    def __init__(self, name, normalize=False, **kwargs):
        super().__init__(name, **kwargs)
        self.normalize = normalize
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'normalize' : self._get_hp(None, 'normalize', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        if self.get_parameters(hp)['normalize']:
            normal_layer = tf.keras.layers.Normalization()
            normal_layer.adapt(driver.get_training_data(self.name))
            inputs = normal_layer(inputs)
        
        return inputs

class PeriodicPredictor(NumericalPredictor):
    def __init__(self, name, period, **kwargs):
        super().__init__(name, **kwargs)
        self.period = period
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'period' : self._get_hp(None, 'period', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        theta = 2 * np.pi * inputs / self.get_parameters(hp)['period']
        
        return tf.concat([tf.math.sin(theta), tf.math.cos(theta)], axis=-1)