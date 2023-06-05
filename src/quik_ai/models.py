from quik_ai import tuning
from quik_ai import layers

import tensorflow as tf
import keras_tuner as kt

class HyperModel(kt.HyperModel, tuning.Tunable):
    
    def __init__(
        self, 
        name, 
        response,
        head,
        predictors,
        driver,
        time_window=1,
        time_dropout=0.05,
        seed=None,
        run_eagerly=None,
        **kwargs
    ):
        kt.HyperModel.__init__(self, name, **kwargs)
        tuning.Tunable.__init__(self, name, **kwargs)
        
        self.response = response
        self.head = head
        self.predictors = predictors
        self.driver = driver
        
        self.time_window = time_window
        self.time_dropout = time_dropout
        self.seed = seed
        self.run_eagerly = run_eagerly
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'time_window' : self._get_hp(None, 'time_window', hp),
            'time_dropout' : self._get_hp(None, 'time_dropout', hp),
            'seed' : self._get_hp(None, 'seed', hp),
        })
        return config
    
    def get_dependent_tunables(self):
        tunables = super().get_dependent_tunables()
        tunables.extend(self.driver.get_dependent_tunables())
        tunables.extend(self.head.get_dependent_tunables())
        for predictor in self.predictors:
            tunables.extend(predictor.get_dependent_tunables())
        return tunables
    
    def get_input_names(self):
        names = []
        uniques = set()
        for predictor in self.predictors:
            for name in predictor.names:
                if name not in uniques:
                    uniques.add(name)
                    names.append(name)
        return names
    
    def __build_input_layers(self, time_window, **kwargs):
        input_layers = {}
        
        input_names = self.get_input_names()
        
        for name in input_names:
            dtype = self.driver.get_input_dtype(name)
            shape = self.driver.get_input_shape(name)

            # append time dimension
            if time_window > 1:
                shape = (time_window,) + shape

            input_layers[name] = tf.keras.Input(name=name, dtype=dtype, shape=shape)
        
        return input_layers
    
    def __build_input_tensor(self, hp, input_layers, time_window, time_dropout, seed, **kwargs):
        
        # if we have time, we need to apply history dropout
        if time_window > 1:
            dropped_inputs = layers.HistoryDropout(time_dropout, seed)(input_layers.values())
            input_layers = {k: v for k, v in zip(input_layers, dropped_inputs)}
        
        # process each predictor transform
        inputs = []
        for predictor in self.predictors:
            inputs.append(predictor.transform(input_layers, self.driver, hp))
        
        # remove dropped inputs
        inputs = [x for x in inputs if x is not None]
        
        # unify dimensions if we have no time
        if time_window <= 1:
            inputs = [tf.keras.layers.Flatten()(x) for x in inputs]
        
        # if all inputs are dropped, we replace with a constant
        if not inputs:
            inputs = []
            for x in input_layers.values():
                inputs.append(tf.ones_like(tf.keras.layers.Flatten()(x), dtype=tf.float32))
            inputs = [tf.keras.layers.GlobalAveragePooling1D()(tf.expand_dims(tf.concat(inputs, -1), -1))]
        
        # will be either shape (batch, time, features) or (batch, features)
        return tf.concat(inputs, axis=-1)
    
    def build(self, hp):
        
        # if we want to build without tuning fill out a
        # dummy instance of hp so we can sample some params
        if hp is None:
            hp = kt.HyperParameters()
        
        # get parameters
        config = self.get_parameters(hp)
        
        # get input layers as a dict
        inputs = self.__build_input_layers(**config)
        
        # build the input tensor from all input layers
        outputs = self.__build_input_tensor(hp, inputs, **config)
        
        # apply the model body
        outputs = self.body(outputs, **config)
        
        # transform using output head
        outputs = self.head.transform(hp, outputs)
        
        # construct the model
        model = tf.keras.Model(
            inputs=inputs.values(), 
            outputs=outputs,
        )
        
        # compile
        model.compile(
            loss=self.head.loss(), 
            optimizer=self.driver.get_optimizer(hp), 
            weighted_metrics=self.head.metrics(),
            run_eagerly=self.run_eagerly,
        )
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        # if we want to build without tuning fill out a
        # dummy instance of hp so we can sample some params
        if hp is None:
            hp = kt.HyperParameters()
        
        config = self.get_parameters(hp)
        input_names = self.get_input_names()
        
        return model.fit(
            self.driver.get_training_tensorflow_dataset(input_names, self.response, config['time_window'], hp),
            *args, 
            steps_per_epoch=self.driver.get_training_steps_per_epoch(hp), 
            validation_data=self.driver.get_validation_tensorflow_dataset(input_names, self.response, config['time_window'], hp), 
            validation_steps=self.driver.get_validation_steps_per_epoch(hp), 
            **kwargs
        )

class ResNet(HyperModel):
    
    def __init__(
        self,
        response,
        head,
        predictors,
        driver,
        model_dim=tuning.HyperInt(min_value=32, max_value=512, step=32),
        blocks=tuning.HyperInt(min_value=0, max_value=6),
        activation=tuning.HyperChoice(['relu','gelu']),
        dropout=tuning.HyperFloat(min_value=0.0, max_value=0.4, step=0.1),
        projection_scale=tuning.HyperInt(min_value=1, max_value=4),
        **kwargs
    ):
        super().__init__('ResNet', response, head, predictors, driver, **kwargs)
        
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