from quik_ai import tuning
from quik_ai import layers

import logging

import numpy as np
import tensorflow as tf

class Predictor(tuning.Tunable):
    def __init__(self, name, numeric=True, drop=False, **kwargs):
        super().__init__(name, **kwargs)
        self.numeric = numeric
        self.drop = drop
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'drop' : self._get_hp(None, 'drop', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        if self.get_parameters(hp)['drop']:
            return None
        
        return inputs
    
class NumericalPredictor(Predictor):
    def __init__(self, name, normalize=False, **kwargs):
        super().__init__(name, **kwargs)
        self.normalize = normalize
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'normalize' : self._get_hp(None, 'normalize', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        if self.get_parameters(hp)['normalize']:
            normal_layer = tf.keras.layers.Normalization()
            normal_layer.adapt(driver.get_training_data(self.name))
            inputs = normal_layer(inputs)
        
        return inputs

class PeriodicPredictor(NumericalPredictor):
    def __init__(self, name, period, **kwargs):
        super().__init__(name, **kwargs)
        self.period = period
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'period' : self._get_hp(None, 'period', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        theta = 2 * np.pi * inputs / self.get_parameters(hp)['period']
        
        return tf.concat([tf.math.sin(theta), tf.math.cos(theta)], axis=-1)
    
class TimeMaskedPredictor(NumericalPredictor):
    def __init__(self, name, mask_n=1, **kwargs):
        super().__init__(name, **kwargs)
        self.mask_n = mask_n
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        unmasked, masked = tf.split(inputs, [-1, self.mask_n], axis=1)
        masked = masked * 0.0
        
        return tf.concat([unmasked, masked], axis=1)

class LambdaPredictor(NumericalPredictor):
    def __init__(self, name, lambdas=None, **kwargs):
        super().__init__(name, **kwargs)
        
        if not isinstance(lambdas, (list, tuple)):
            logging.error('Expected lambdas as a list of functions!')
            self.lambda_count = 0
            return
        
        self.lambda_count = len(lambdas)
        
        for i in range(self.lambda_count):
            lbda = lambdas[i]
            if isinstance(lbda, (list, tuple)):
                setattr(self, 'lambda_%s' % i, lbda[0])
                setattr(self, 'lambda_%s_drop' % i, lbda[1])
            elif callable(lbda):
                setattr(self, 'lambda_%s' % i, lbda)
                setattr(self, 'lambda_%s_drop' % i, False)
            else:
                logging.error('Each lambda in lambdas should be a function or a tuple of (function, boolean)')
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        
        with self._condition_on_parent(hp, 'drop', [False], scope=None) as scope:
            for i in range(self.lambda_count):
                config['lambda_%s_drop' % i] = self._get_hp(scope, 'lambda_%s_drop' % i, hp)
        
        return config
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        if self.lambda_count == 0:
            return None
        
        config = self.get_parameters(hp)
        
        res = []
        for i in range(self.lambda_count):
            if not config['lambda_%s_drop' % i]:
                lbda = getattr(self, 'lambda_%s' % i)
                res.append(lbda(inputs))
        
        if len(res) == 0:
            return None
        
        return tf.concat(res, axis=-1)

class CategoricalPredictor(Predictor):
    def __init__(
        self, 
        name, 
        dropout=tuning.HyperFloat(min_value=0.0, max_value=0.4, step=0.1),
        use_one_hot=tuning.HyperBoolean(),
        embed_dim=tuning.HyperInt(min_value=8, max_value=32, step=8),
        embed_l2_regularizer=tuning.HyperFloat(min_value=0.0, max_value=0.2, step=0.1),
        dropout_token='[UNK]',
        seed=None,
        **kwargs
    ):
        super().__init__(name, numeric=False, **kwargs)
        self.dropout = dropout
        self.use_one_hot = use_one_hot
        self.embed_dim = embed_dim
        self.embed_l2_regularizer = embed_l2_regularizer
        self.dropout_token = dropout_token
        self.seed = seed
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        
        with self._condition_on_parent(hp, 'drop', [False], scope=None) as drop_scope:
            config['dropout'] = self._get_hp(drop_scope, 'dropout', hp)
            config['use_one_hot'] = self._get_hp(drop_scope, 'use_one_hot', hp)
            with self._condition_on_parent(hp, 'use_one_hot', [False], scope=drop_scope) as one_hot_scope:
                config['embed_dim'] = self._get_hp(one_hot_scope, 'embed_dim', hp)
                config['embed_l2_regularizer'] = self._get_hp(one_hot_scope, 'embed_l2_regularizer', hp)
        
        return config
    
    def transform(self, inputs, driver, hp):
        inputs = super().transform(inputs, driver, hp)
        
        if inputs is None:
            return None
        
        config = self.get_parameters(hp)
        
        dropout = config['dropout']
        use_one_hot = config['use_one_hot']
        
        input_dim = len(inputs.shape)
        
        if input_shape > 3:
            logging.error('Categorical predictor input must be (1) dims for flat data, or (2) dims for time-series')
        
        # flatten time series
        if input_shape == 3:
            inputs = tf.squeeze(inputs, axis=2)
        
        # dropout
        inputs = layers.CategoricalDropout(
            dropout=dropout, 
            dropout_token=self.dropout_token,
            seed=self.seed,
        )(inputs)
        
        encode_layer = tf.keras.layers.StringLookup(oov_token=self.dropout_token)
        encode_layer.adapt(driver.get_training_data(self.name))
        inputs = encode_layer(inputs)
        
        vocab_size = len(encode_layer.get_vocabulary())
        
        # if one hot encoding
        if use_one_hot:
            return tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=vocab_size,
                embeddings_initializer=tf.keras.initializers.Identity(),
                trainable=False,
            )(inputs)
        
        embed_dim = config['embed_dim']
        embed_l2_regularizer = config['embed_l2_regularizer']
        
        # if we embed the categories
        return tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_regularizer=tf.keras.regularizers.L2(embed_l2_regularizer),
        )(inputs)