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