import tensorflow as tf

from keras import backend
from keras.engine import base_layer
from keras.utils import control_flow_util

class CategoricalDropout(base_layer.BaseRandomLayer):
    def __init__(self, dropout, dropout_token='[UNK]', seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        
        if isinstance(dropout, (int, float)) and not 0 <= dropout <= 1:
            raise ValueError('Invalid value %s for dropout, expected a value in [0, 1]' % dropout)
        
        self.dropout = dropout
        self.dropout_token = dropout_token
        self.seed = seed
    
    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        
        def dropped_inputs():
            switchs = tf.less(self._random_generator.random_uniform(tf.shape(inputs)), self.dropout)
            return tf.where(switchs, self.dropout_token, tf.identity(inputs))
        
        output = control_flow_util.smart_cond(
            training, dropped_inputs, lambda: tf.identity(inputs)
        )
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dropout' : self.dropout,
            'dropout_token' : self.dropout_token,
            'seed' : self.seed,
        })
        return config

class ResNetBlock(tf.keras.layers.Layer):
    
    def __init__(self, activation, dropout, projection_scale, **kwargs):
        super().__init__(**kwargs)
        
        self.activation = activation
        self.dropout = dropout
        self.projection_scale = projection_scale
    
    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.projection_scale * input_shape[-1], activation=self.activation),
            tf.keras.layers.Dense(units=input_shape[-1]),
            tf.keras.layers.Dropout(rate=self.dropout),
        ])
    
    def call(self, inputs):
        return inputs + self.ffn(self.norm(inputs))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'activation' : self.activation,
            'dropout' : self.dropout,
            'projection_scale' : self.projection_scale,
        })
        return config