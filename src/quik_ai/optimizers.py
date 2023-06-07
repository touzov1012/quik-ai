from quik_ai import tuning

import tensorflow as tf

class Optimizer(tuning.Tunable):
    pass

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, model_dim, warmup_steps, **kwargs):
        super().__init__(**kwargs)
        
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(tf.cast(self.model_dim, tf.float32)) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        config = {
            'model_dim' : self.model_dim,
            'warmup_steps' : self.warmup_steps,
        }
        return config
    
class Noam(Optimizer):
    def __init__(
        self, 
        model_dim=tuning.HyperChoice([64, 128, 256, 512, 1024, 2048, 4096]), 
        warmup_steps=tuning.HyperChoice([1000, 2000, 4000, 8000]),
        **kwargs
    ):
        super().__init__('Noam', **kwargs)
        
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'model_dim' : self._get_hp(None, 'model_dim', hp),
            'warmup_steps' : self._get_hp(None, 'warmup_steps', hp),
        })
        return config
    
    def build(self, hp):
        return tf.keras.optimizers.Adam(NoamSchedule(**self.get_parameters(hp)), beta_1=0.9, beta_2=0.98, epsilon=1e-9)