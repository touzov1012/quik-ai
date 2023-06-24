import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class LogProbLoss(tf.keras.losses.Loss):
    def __init__(self, response_noise, log_response, name='log_prob', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.response_noise = response_noise
        self.log_response = log_response

    def call(self, response, model):
        response = tf.cast(tf.reshape(response, tf.shape(model)), tf.float32)
        response = response + tf.random.uniform(tf.shape(response), -self.response_noise, self.response_noise)
        response = tf.math.log(tf.math.maximum(response,1.0)) if self.log_response else response
        return -model.log_prob(response)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'response_noise' : self.response_noise,
            'log_response' : self.log_response,
        })
        return config