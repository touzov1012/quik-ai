from quik_ai import tuning
from quik_ai import layers

import numpy as np
import tensorflow as tf

class Predictor(tuning.Tunable):
    def __init__(self, names, drop=False, **kwargs):
        super().__init__(names, **kwargs)
        self.names = names if isinstance(names, (list, tuple)) else [names]
        self.drop = drop
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'drop' : self._get_hp(None, 'drop', hp)
        })
        return config
    
    def transform(self, inputs, driver, hp):
        return None

class Lambda(Predictor):
    def __init__(self, names, lambdas=None, **kwargs):
        super().__init__(names, **kwargs)
        
        if callable(lambdas):
            lambdas = [lambdas]
        
        if not isinstance(lambdas, (list, tuple)):
            raise ValueError('Expected lambdas as a list of functions')
        
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
                raise ValueError('Each lambda in lambdas should be a function or a tuple of (function, boolean)')
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        
        with self._condition_on_parent(hp, 'drop', [False], scope=None) as scope:
            for i in range(self.lambda_count):
                config['lambda_%s_drop' % i] = self._get_hp(scope, 'lambda_%s_drop' % i, hp)
        
        return config
    
    def transform(self, inputs, driver, hp):
        config = self.get_parameters(hp)
        
        if config['drop']:
            return None
        
        if self.lambda_count == 0:
            return None
        
        res = []
        for i in range(self.lambda_count):
            if not config['lambda_%s_drop' % i]:
                lbda = getattr(self, 'lambda_%s' % i)
                res.append(lbda(inputs, driver, **config))
        
        if len(res) == 0:
            return None
        
        return tf.concat(res, axis=-1)
    
class Numerical(Lambda):
    def __init__(self, names, normalize=False, **kwargs):
        super().__init__(names, lambdas=self.body, **kwargs)
        self.normalize = normalize
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        
        with self._condition_on_parent(hp, 'drop', [False], scope=None) as scope:
            config['normalize'] = self._get_hp(scope, 'normalize', hp)
        
        return config
    
    def body(self, inputs, driver, normalize, **kwargs):  
        
        outputs = []
        
        if normalize:
            for name in self.names:
                normal_layer = tf.keras.layers.Normalization()
                normal_layer.adapt(driver.get_data_tensor(driver.training_data, name))
                outputs.append(normal_layer(inputs[name]))
        else:
            for name in self.names:
                outputs.append(inputs[name])
        
        return tf.concat(outputs, axis=-1)

class Periodic(Lambda):
    def __init__(self, names, period, **kwargs):
        super().__init__(names, lambdas=self.body, **kwargs)
        self.period = period
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        
        with self._condition_on_parent(hp, 'drop', [False], scope=None) as scope:
            config['period'] = self._get_hp(scope, 'period', hp)
        
        return config
    
    def body(self, inputs, driver, period, **kwargs):
        outputs = [inputs[name] for name in self.names]
        
        outputs = tf.concat(outputs, axis=-1)
        
        theta = 2 * np.pi * outputs / period
        
        return tf.concat([tf.math.sin(theta), tf.math.cos(theta)], axis=-1)
    
class TimeMasked(Lambda):
    def __init__(self, names, mask_n=1, **kwargs):
        super().__init__(names, lambdas=self.body, **kwargs)
        self.mask_n = mask_n
    
    def body(self, inputs, driver, **kwargs):
        outputs = [inputs[name] for name in self.names]
        
        outputs = tf.concat(outputs, axis=-1)
        
        if len(outputs.shape) < 3:
            raise ValueError('Time masked predictor must have at least (3) dimensions')
        
        unmasked, masked = tf.split(outputs, [-1, self.mask_n], axis=1)
        masked = masked * 0.0
        
        return tf.concat([unmasked, masked], axis=1)

class Categorical(Lambda):
    def __init__(
        self, 
        names, 
        dropout=tuning.HyperFloat(min_value=0.0, max_value=0.4, step=0.1),
        use_one_hot=tuning.HyperBoolean(),
        embed_dim=tuning.HyperInt(min_value=8, max_value=32, step=8),
        embed_l2_regularizer=tuning.HyperFloat(min_value=0.0, max_value=0.2, step=0.1),
        dropout_token='[UNK]',
        seed=None,
        **kwargs
    ):
        super().__init__(names, lambdas=self.body, **kwargs)
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
    
    def body(self, inputs, driver, dropout, use_one_hot, embed_dim, embed_l2_regularizer, **kwargs):
        outputs = [inputs[name] for name in self.names]
        
        # format to flat tensor for each input, this will be
        # expanded by a dimension in the end to (3)
        for i in range(len(outputs)):
            input_shape = len(outputs[i].shape)
            
            if input_shape > 3:
                raise ValueError('Categorical predictor input must be (1) dims for flat data, or (2) dims for time-series')

            # flatten time series
            if input_shape == 3:
                outputs[i] = tf.squeeze(outputs[i], axis=2)
        
        # create dropout layer
        dropout_layer = layers.CategoricalDropout(
            dropout=dropout, 
            dropout_token=self.dropout_token,
            seed=self.seed,
        )
        
        # apply dropout to each input
        for i in range(len(outputs)):
            outputs[i] = dropout_layer(outputs[i])
            
            # encode to numeric
            encode_layer = tf.keras.layers.StringLookup(oov_token=self.dropout_token)
            encode_layer.adapt(np.unique(driver.get_data_tensor(driver.training_data, self.names[i])))
            outputs[i] = encode_layer(outputs[i])
            
            # get the vocab size for this input
            vocab_size = len(encode_layer.get_vocabulary())
        
            # if one hot encoding
            if use_one_hot:
                outputs[i] = tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=vocab_size,
                    embeddings_initializer=tf.keras.initializers.Identity(),
                    trainable=False,
                )(outputs[i])
            else: # we are embedding categories
                outputs[i] = tf.keras.layers.Embedding(
                    input_dim=vocab_size,
                    output_dim=embed_dim,
                    embeddings_regularizer=tf.keras.regularizers.L2(embed_l2_regularizer),
                )(outputs[i])
        
        return tf.concat(outputs, axis=-1)