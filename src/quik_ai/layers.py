import tensorflow as tf
import tensorflow_probability as tfp

from keras import backend
from keras.engine import base_layer
from keras.utils import control_flow_util

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class GaussianMixtureLayer(tf.keras.layers.Layer):
    def __init__(self, num_components, event_shape, **kwargs):
        super().__init__(**kwargs)
        self.num_components = num_components
        self.event_shape = event_shape

    def call(self, inputs):
        return tfp.layers.MixtureNormal(self.num_components, self.event_shape)(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_components' : self.num_components,
            'event_shape' : self.event_shape
        })
        return config

@tf.keras.utils.register_keras_serializable(package="quik_ai")
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

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class HistoryDropout(base_layer.BaseRandomLayer):
    def __init__(self, dropout, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        
        if isinstance(dropout, (int, float)) and not 0 <= dropout <= 1:
            raise ValueError('Invalid value %s for dropout, expected a value in [0, 1]' % dropout)
        
        self.dropout = dropout
        self.seed = seed
    
    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()
        
        multi_input = isinstance(inputs, (list, tuple))
        if not multi_input:
            outputs = [inputs]
        else:
            outputs = inputs
        
        def dropped_inputs():
            # inputs is a list of tensors of same batch size and time
            shape = tf.shape(outputs[0])
            
            # create a count of each time steps spot
            t_cnt = tf.repeat(tf.expand_dims(tf.range(shape[1], delta=1, dtype=tf.float32), 0), repeats=shape[0], axis=0)
            
            # switch a random number of time steps off sequentially
            sub_switchs = self._random_generator.random_uniform((shape[0], 1)) * tf.cast(shape[1] - 1, tf.float32)
            sub_switchs = tf.less(sub_switchs, t_cnt)
            sub_switchs = tf.expand_dims(sub_switchs, -1)
            
            # only use the sub sample at our specified rate
            switchs = tf.less(self._random_generator.random_uniform((shape[0], 1, 1)), self.dropout)
            
            # apply the switch
            samples = []
            for out in outputs:
                sub_sample = tf.where(sub_switchs, tf.identity(out), tf.zeros_like(out, dtype=out.dtype))
                samples.append(tf.where(switchs, sub_sample, tf.identity(out)))
            
            return samples
        
        output = control_flow_util.smart_cond(
            training, dropped_inputs, lambda: [tf.identity(out) for out in outputs]
        )
        
        return output if multi_input else output[0]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dropout' : self.dropout,
            'seed' : self.seed,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="quik_ai")
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