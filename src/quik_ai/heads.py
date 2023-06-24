from quik_ai import tuning
from quik_ai import layers
from quik_ai import losses
from quik_ai import metrics

import tensorflow as tf
import tensorflow_probability as tfp

losses_dictionary = {
    'mean_squared_error' : tf.keras.losses.MeanSquaredError,
    'log_prob' : losses.LogProbLoss,
}

metrics_dictionary = {
    'mean_squared_error' : metrics.MeanSquaredErrorMetric,
    'log_prob' : metrics.LogProbMetric,
}

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

class Regression(Head):
    def __init__(self, event_size=1, loss_name='mean_squared_error', name='Regression', **kwargs):
        super().__init__(name, **kwargs)
        
        self.event_size = event_size
        self.loss_name = loss_name
    
    def body(self, inputs, **kwargs):
        return tf.keras.layers.Dense(self.event_size)(inputs)
    
    def monitor(self):
        return self.loss_name
    
    def loss(self):
        return losses_dictionary[self.loss_name]()
    
    def metrics(self):
        return metrics_dictionary[self.loss_name]()

class Logistic(Head):
    def __init__(self, event_size, sparse_response=True, multi_label=False, name='Logistic', **kwargs):
        super().__init__(name, **kwargs)
        
        self.event_size = event_size
        self.sparse_response = sparse_response
        self.multi_label = multi_label
    
    def body(self, inputs, **kwargs):
        return tf.keras.layers.Dense(self.event_size)(inputs)
    
    def monitor(self):
        if self.sparse_response:
            return 'sparse_categorical_crossentropy'
        elif self.multi_label:
            return 'binary_crossentropy'
        else:
            return 'categorical_crossentropy'
    
    def loss(self):
        if self.sparse_response:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif self.multi_label:
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def metrics(self):
        if self.sparse_response:
            return [tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True), 'sparse_categorical_accuracy']
        elif self.multi_label:
            return [tf.keras.metrics.BinaryCrossentropy(from_logits=True), 'binary_accuracy']
        else:
            return [tf.keras.metrics.CategoricalCrossentropy(from_logits=True), 'categorical_accuracy']
        
class GaussianMixture(Head):
    def __init__(
        self,  
        event_shape=[1], 
        mix_components=tuning.HyperChoice([8, 16, 32, 64, 128]), 
        positive_only=False,
        response_noise=0.0,
        log_response=False, 
        name='GaussianMixture', 
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.event_shape = event_shape
        self.mix_components = mix_components
        self.positive_only = positive_only
        self.response_noise = response_noise
        self.log_response = log_response
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'mix_components' : self._get_hp(None, 'mix_components', hp)
        })
        return config
    
    def body(self, inputs, mix_components, **kwargs):
        # get the parameter size
        params_size = tfp.layers.MixtureNormal.params_size(mix_components, self.event_shape)
        
        # final output to the parameter size
        inputs = tf.keras.layers.Dense(
            params_size, 
            activation='softplus' if self.positive_only else None
        )(inputs)
        
        return layers.GaussianMixtureLayer(mix_components, self.event_shape)(inputs)
    
    def monitor(self):
        return 'log_prob'
    
    def loss(self):
        return losses.LogProbLoss(response_noise=self.response_noise, log_response=self.log_response)
    
    def metrics(self):
        return metrics.LogProbMetric(log_response=self.log_response)