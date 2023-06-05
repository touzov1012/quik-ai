from quik_ai import tuning

import tensorflow as tf

class Head(tuning.Tunable):
    def __init__(
        self, 
        name, 
        non_linear_projection=tuning.HyperBoolean(),
        projection_scale=tuning.HyperInt(min_value=1, max_value=4),
        activation=tuning.HyperChoice(['relu','gelu']),
        objective_direction='min',
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.non_linear_projection = non_linear_projection
        self.projection_scale = projection_scale
        self.activation = activation
        self.objective_direction = objective_direction
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        
        config['non_linear_projection'] = self._get_hp(None, 'non_linear_projection', hp)
        with self._condition_on_parent(hp, 'non_linear_projection', [True], scope=None) as scope:
            config['projection_scale'] = self._get_hp(scope, 'projection_scale', hp)
            config['activation'] = self._get_hp(scope, 'activation', hp)
            
        return config
    
    def transform(self, hp, inputs):
        config = self.get_parameters(hp)
        
        if config['non_linear_projection']:
            inputs = tf.keras.layers.Dense(inputs.shape[-1] * config['projection_scale'], activation=config['activation'])(inputs)
        
        return self.body(inputs, **config)

class RegressionHead(Head):
    def __init__(self, event_size=1, loss_name='mean_squared_error', **kwargs):
        super().__init__('RegressionHead', **kwargs)
        
        self.event_size = event_size
        self.loss_name = loss_name
    
    def body(self, inputs, **kwargs):
        return tf.keras.layers.Dense(self.event_size)(inputs)
    
    def monitor(self):
        return self.loss_name
    
    def loss(self):
        return self.loss_name
    
    def metrics(self):
        return self.loss_name