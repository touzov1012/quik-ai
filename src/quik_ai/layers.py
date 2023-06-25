import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from keras import backend
from keras.engine import base_layer
from keras.utils import control_flow_util

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class FeatureFlatten(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.flatten_dims = np.prod(input_shape[1:-1])
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, (-1, self.flatten_dims, shape[-1]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.flatten_dims, input_shape[-1])

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class TimeFlatten(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.flatten_dims = np.prod(input_shape[2:])
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, (-1, shape[1], self.flatten_dims))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.flatten_dims)

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
                # broadcast to correct number of dims
                sub_shape_new = [-1] + sub_switchs.shape.as_list()[1:] + [1] * (len(out.shape) - len(sub_switchs.shape))
                shape_new = [-1] + switchs.shape.as_list()[1:] + [1] * (len(out.shape) - len(sub_switchs.shape))
                
                # broadcast the switches
                b_sub_switch = tf.reshape(sub_switchs, sub_shape_new)
                b_switch = tf.reshape(switchs, shape_new)
                
                sub_sample = tf.where(b_sub_switch, tf.identity(out), tf.zeros_like(out, dtype=out.dtype))
                samples.append(tf.where(b_switch, sub_sample, tf.identity(out)))
            
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

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class ChunkEmbedding(tf.keras.layers.Layer):
    def __init__(self, model_dim, chunk_size=None, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.chunk_size = chunk_size
    
    def build(self, input_shape):
        self.projection = tf.keras.layers.Dense(self.model_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=input_shape[1], output_dim=self.model_dim)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        super().build(input_shape)

    def call(self, inputs):
        # a counter to get the index of the embedding position
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)

        # project into the embedding dimension
        outputs = self.projection(inputs)

        # flatten and add positional encoding
        outputs = outputs + self.position_embedding(positions)

        # normalize the embedding vectors
        outputs = self.layernorm(outputs)

        # apply chunking if we have large sequences
        if self.chunk_size is not None:
            outputs = tf.reshape(outputs, (-1, tf.shape(outputs)[1] // self.chunk_size, self.chunk_size, self.model_dim))

        return outputs
    
    def compute_output_shape(self, input_shape):
        if self.chunk_size is not None:
            return (input_shape[0], input_shape[1] // self.chunk_size, self.chunk_size, self.model_dim)
        else:
            return input_shape[0], input_shape[1], self.model_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'chunk_size': self.chunk_size
        })
        return config

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, dropout, multi_head_attention=None, **kwargs):
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.dropout = dropout
        self.multi_head_attention = multi_head_attention

    def build(self, input_shape):
        
        # input normalization layers
        self.query_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.key_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.value_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        
        # we need to set the initializer to fix the weight loading
        kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0)
        
        # main multi-head attention layer
        if self.multi_head_attention is None:
            self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=input_shape[-1],
                dropout=self.dropout,
                kernel_initializer=kernel_initializer,
            )

        self.attention_scores = None
        
        super().build(input_shape)
        
    def call(self, input_query, key, value):
        # apply normalization of inputs
        query = self.query_layernorm(input_query)
        key = self.key_layernorm(key)
        value = self.value_layernorm(value)
        
        # multihead attention
        (attention_outputs, attention_scores) = self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            return_attention_scores=True,
        )

        # save attention scores for visualization
        self.attention_scores = attention_scores

        # apply skip connection around attention block
        return input_query + attention_outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads' : self.num_heads,
            'dropout' : self.dropout,
            'multi_head_attention' : tf.keras.layers.serialize(self.multi_head_attention),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config['multi_head_attention'] = tf.keras.layers.deserialize(config['multi_head_attention'])
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        ffn_activation,
        ffn_dropout,
        ffn_projection_scale,
        num_heads,
        attn_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.ffn_activation = ffn_activation
        self.ffn_dropout = ffn_dropout
        self.ffn_projection_scale = ffn_projection_scale
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.attention = None
        
    def build(self, input_shape):
        
        # create the base multi head attention
        if self.attention is None:
            self.attention = BaseAttention(
                num_heads=self.num_heads,
                dropout=self.attn_dropout,
            )
        
        # followed by a resnet block
        self.ffn = ResNetBlock(
            activation=self.ffn_activation,
            dropout=self.ffn_dropout, 
            projection_scale=self.ffn_projection_scale,
        )

        self.attention_scores = None
        
        super().build(input_shape)

    def call(self, query, key, value):
        # apply the attention
        outputs = self.attention(query, key, value)

        # save the attention scores
        self.attention_scores = self.attention.attention_scores

        # run through the ffn layers
        return self.ffn(outputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'ffn_activation' : self.ffn_activation,
            'ffn_dropout' : self.ffn_dropout,
            'ffn_projection_scale' : self.ffn_projection_scale,
            'num_heads' : self.num_heads,
            'attn_dropout' : self.attn_dropout,
            'attention': self.attention.get_config(),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        attention = config.pop('attention')
        
        layer = cls(**config)
        layer.attention = BaseAttention.from_config(attention)
        
        return layer

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class Transformer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        ffn_activation,
        ffn_dropout,
        ffn_projection_scale,
        num_heads,
        attn_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.num_layers = num_layers
        self.ffn_activation = ffn_activation
        self.ffn_dropout = ffn_dropout
        self.ffn_projection_scale = ffn_projection_scale
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.perceptual_module = None
        
    def build(self, input_shape):

        self.attention_scores = []

        # build series of transformer blocks
        if self.perceptual_module is None:
            self.perceptual_module = []
            for layer_idx in range(self.num_layers):
                self.perceptual_module.append(TransformerBlock(
                    ffn_activation=self.ffn_activation,
                    ffn_dropout=self.ffn_dropout,
                    ffn_projection_scale=self.ffn_projection_scale,
                    num_heads=self.num_heads,
                    attn_dropout=self.attn_dropout,
                ))
        
        super().build(input_shape)

    def call(self, inputs):
        for layer in self.perceptual_module:
            inputs = layer(query=inputs, key=inputs, value=inputs)
            self.attention_scores.append(layer.attention_scores)

        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers' : self.num_layers,
            'ffn_activation' : self.ffn_activation,
            'ffn_dropout' : self.ffn_dropout,
            'ffn_projection_scale' : self.ffn_projection_scale,
            'num_heads' : self.num_heads,
            'attn_dropout' : self.attn_dropout,
            'perceptual_module' : [mod.get_config() for mod in self.perceptual_module],
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        perceptual_module = config.pop('perceptual_module')
        
        layer = cls(**config)
        layer.perceptual_module = [TransformerBlock.from_config(mod) for mod in perceptual_module]
        
        return layer

@tf.keras.utils.register_keras_serializable(package="quik_ai")
class TransformerRecurrentCell(tf.keras.layers.Layer):
    def __init__(
        self,
        chunk_size,
        model_dim,
        xattn_rate,
        num_layers,
        ffn_activation,
        ffn_dropout,
        ffn_projection_scale,
        num_heads,
        attn_dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.chunk_size = chunk_size
        self.model_dim = model_dim
        self.xattn_rate = xattn_rate
        self.num_layers = num_layers
        self.ffn_activation = ffn_activation
        self.ffn_dropout = ffn_dropout
        self.ffn_projection_scale = ffn_projection_scale
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.perceptual_module = None
        self.tlb_module = None
        
        # create necessary state and output size for the recurrent cell
        self.state_size = tf.TensorShape([self.chunk_size, model_dim])
        self.output_size = tf.TensorShape([self.chunk_size, model_dim])
        
    def build(self, input_shape):

        self.attention_scores = []

        # perceptual module
        if self.perceptual_module is None:
            self.perceptual_module = []
            for layer_idx in range(self.num_layers):
                self.perceptual_module.append(TransformerBlock(
                    ffn_activation=self.ffn_activation,
                    ffn_dropout=self.ffn_dropout,
                    ffn_projection_scale=self.ffn_projection_scale,
                    num_heads=self.num_heads,
                    attn_dropout=self.attn_dropout,
                ))
                if layer_idx % self.xattn_rate == 0:
                    self.perceptual_module.append(TransformerBlock(
                        ffn_activation=self.ffn_activation,
                        ffn_dropout=self.ffn_dropout,
                        ffn_projection_scale=self.ffn_projection_scale,
                        num_heads=self.num_heads,
                        attn_dropout=self.attn_dropout,
                    ))

        # temporal latent bottleneck module
        if self.tlb_module is None:
            self.tlb_module = TransformerBlock(
                ffn_activation=self.ffn_activation,
                ffn_dropout=self.ffn_dropout,
                ffn_projection_scale=self.ffn_projection_scale,
                num_heads=self.num_heads,
                attn_dropout=self.attn_dropout,
            )
        
        super().build(input_shape)

    def call(self, inputs, states):
        # inputs => (batch, chunk_size, dims)
        # states => [(batch, chunk_size, units)]
        slow_stream = states[0]
        fast_stream = inputs

        for layer_idx, layer in enumerate(self.perceptual_module):
            fast_stream = layer(query=fast_stream, key=fast_stream, value=fast_stream)

            if layer_idx % self.xattn_rate == 0:
                fast_stream = layer(
                    query=fast_stream, key=slow_stream, value=slow_stream
                )

        slow_stream = self.tlb_module(
            query=slow_stream, key=fast_stream, value=fast_stream
        )

        # save attention scores for visualization
        self.attention_scores.append(self.tlb_module.attention_scores)

        return fast_stream, [slow_stream]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'chunk_size' : self.chunk_size,
            'model_dim' : self.model_dim,
            'xattn_rate' : self.xattn_rate,
            'num_layers' : self.num_layers,
            'ffn_activation' : self.ffn_activation,
            'ffn_dropout' : self.ffn_dropout,
            'ffn_projection_scale' : self.ffn_projection_scale,
            'num_heads' : self.num_heads,
            'attn_dropout' : self.attn_dropout,
            'perceptual_module' : [mod.get_config() for mod in self.perceptual_module],
            'tlb_module' : self.tlb_module.get_config(),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        perceptual_module = config.pop('perceptual_module')
        tlb_module = config.pop('tlb_module')
        
        layer = cls(**config)
        layer.perceptual_module = [TransformerBlock.from_config(mod) for mod in perceptual_module]
        layer.tlb_module = TransformerBlock.from_config(tlb_module)
        
        return layer