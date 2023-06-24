import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class MeanSquaredErrorMetric(tf.keras.metrics.Metric):
    def __init__(self, name='mean_squared_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        squared_error = tf.math.squared_difference(y_true, y_pred)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, y_true.dtype)
            squared_error = tf.multiply(squared_error, sample_weight)
        num_values = tf.cast(tf.size(y_true), self.dtype)
        
        self.total.assign_add(tf.reduce_sum(squared_error))
        self.count.assign_add(num_values)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0.)
        self.count.assign(0.)

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class LogProbMetric(tf.keras.metrics.Metric):
    def __init__(self, log_response, name='log_prob', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.log_response = log_response
        
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def loss(self, response, model, sample_weight):
        model_shape = tf.shape(model)
        response = tf.cast(tf.reshape(response, model_shape), tf.float32)
        response = tf.math.log(tf.math.maximum(response,1.0)) if self.log_response else response
        response = tf.reshape(model.log_prob(response), model_shape)
        return -response if sample_weight is None else -tf.multiply(response, sample_weight)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(self.loss(y_true, y_pred, sample_weight), self.dtype)
        num_values = tf.cast(tf.size(values), self.dtype)

        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(num_values)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    
    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0.)
        self.count.assign(0.)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'log_response' : self.log_response,
        })
        return config
