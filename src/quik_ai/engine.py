from quik_ai import tuning
from quik_ai import optimizers

import tensorflow as tf

class Driver(tuning.Tunable):
    
    def __init__(
        self, 
        training_data, 
        validation_data, 
        testing_data, 
        optimizer,
        shuffle=True,
        batch_size=tuning.HyperChoice([64, 128, 256, 512, 1024, 2048, 4096]),
        **kwargs
    ):
        super().__init__('Driver', **kwargs)
        
        self.training_data = training_data
        self.validation_data = validation_data 
        self.testing_data = testing_data
        self.optimizer = optimizer
        
        self.shuffle = shuffle
        self.batch_size = batch_size
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'shuffle' : self._get_hp(None, 'shuffle', hp),
            'batch_size' : self._get_hp(None, 'batch_size', hp),
        })
        return config
    
    def get_training_data(self, columns):
        if columns is None:
            return self.training_data
        return self.training_data[columns]
    
    def get_validation_data(self, columns):
        if columns is None:
            return self.validation_data
        return self.validation_data[columns]
    
    def get_testing_data(self, columns):
        if columns is None:
            return self.testing_data
        return self.testing_data[columns]
    
    def get_input_dtype(self, column):
        return tf.float32 if self.training_data[column].dtype.kind in 'iufcb' else tf.string
    
    def get_input_shape(self, column):
        return (1,)
    
    def get_optimizer(self, hp):
        if isinstance(self.optimizer, optimizers.Optimizer):
            return self.optimizer.build(hp)
        
        return self.optimizer