import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class LogProbMetric(tf.keras.metrics.Metric):
    def __init__(self, log_response, name='log_prob', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.log_response = log_response
        
        self.total = self.add_weight(name='tp', initializer='zeros')
        self.count = self.add_weight(name='tc', initializer='zeros')

    def loss(self, response, model, w):
        response = tf.cast(response, tf.float32)
        response = tf.math.log(tf.math.maximum(response,1.0)) if self.log_response else response
        return -model.log_prob(response) if w is None else -model.log_prob(response) * w
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        values = tf.cast(self.loss(y_true, y_pred, sample_weight), self.dtype)
        num_values = tf.cast(tf.size(values), self.dtype)
        
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(num_values)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'log_response' : self.log_response,
        })
        return config
