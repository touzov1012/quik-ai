from quik_ai import tuning
from quik_ai import layers

import tensorflow as tf
import keras_tuner as kt

class HyperModel(kt.HyperModel, tuning.Tunable):
    
    def __init__(
        self, 
        name, 
        head,
        driver,
        time_window=1,
        time_dropout=tuning.HyperFloat(min_value=0.0, max_value=0.4, step=0.1),
        **kwargs
    ):
        kt.HyperModel.__init__(self, name, **kwargs)
        tuning.Tunable.__init__(self, name, **kwargs)
        
        self.head = head
        self.driver = driver
        
        self.time_window = time_window
        self.time_dropout = time_dropout
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'time_window' : self._get_hp(None, 'time_window', hp),
            'time_dropout' : self._get_hp(None, 'time_dropout', hp),
        })
        return config
    
    def build(self, hp):
        
        # if we want to build without tuning fill out a
        # dummy instance of hp so we can sample some params
        if hp is None:
            hp = kt.HyperParameters()
        
        # build the input tensor from all input layers
        inputs = self.driver.build_input_tensor()
        
        # apply the model body
        outputs = self.body(inputs, **self.get_parameters(hp))
        
        # transform using output head
        outputs = self.head.transform(hp, outputs)
        
        # construct the model
        model = tf.keras.Model(
            inputs=self.driver.get_input_layers(), 
            outputs=outputs,
        )
        
        # compile
        model.compile(
            loss=self.head.loss(), 
            optimizer=self.driver.optimizer.build(hp), 
            weighted_metrics=self.head.metrics(),
        )
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return driver.fit(hp, model, *args, **kwargs)

class ResNet(HyperModel):
    
    def __init__(
        self,
        head,
        driver,
        model_dim=tuning.HyperInt(min_value=32, max_value=512, step=32),
        blocks=tuning.HyperInt(min_value=0, max_value=6),
        activation=tuning.HyperChoice(['relu','gelu']),
        dropout=tuning.HyperFloat(min_value=0.0, max_value=0.4, step=0.1),
        projection_scale=tuning.HyperInt(min_value=1, max_value=4),
        **kwargs
    ):
        super().__init__('ResNet', head, driver, **kwargs)
        
        self.model_dim = model_dim
        self.blocks = blocks
        self.activation = activation
        self.dropout = dropout
        self.projection_scale = projection_scale
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'model_dim' : self._get_hp(None, 'model_dim', hp),
            'blocks' : self._get_hp(None, 'blocks', hp),
            'activation' : self._get_hp(None, 'activation', hp),
            'dropout' : self._get_hp(None, 'dropout', hp),
            'projection_scale' : self._get_hp(None, 'projection_scale', hp),
        })
        return config
    
    def body(self, inputs, model_dim, blocks, activation, dropout, projection_scale, **kwargs):
        
        # flatten dimensions
        if len(inputs.shape) > 2:
            inputs = tf.keras.layers.Flatten()(inputs)
        
        # project into model dimension
        inputs = tf.keras.layers.Dense(model_dim)(inputs)
        
        # apply resnet blocks
        for _ in range(blocks):
            inputs = layers.ResNetBlock(
                activation=activation, 
                dropout=dropout, 
                projection_scale=projection_scale,
            )(inputs)
        
        # return this body output
        return inputs