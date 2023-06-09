from quik_ai import tuning
from quik_ai import optimizers
from quik_ai import backend

import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd

class Driver(tuning.Tunable):
    
    def __init__(
        self, 
        training_data, 
        validation_data, 
        testing_data, 
        optimizer='adam',
        max_steps_per_epoch=None,
        weights_column=None,
        time_group_column=None,
        shuffle=True,
        batch_size=tuning.HyperChoice([64, 128, 256, 512, 1024, 2048, 4096]),
        **kwargs
    ):
        super().__init__('Driver', **kwargs)
        
        self.training_data = training_data
        self.validation_data = validation_data 
        self.testing_data = testing_data
        self.optimizer = optimizer
        self.max_steps_per_epoch = max_steps_per_epoch
        self.weights_column = weights_column
        self.time_group_column = time_group_column
        
        self.shuffle = shuffle
        self.batch_size = batch_size
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'shuffle' : self._get_hp(None, 'shuffle', hp),
            'batch_size' : self._get_hp(None, 'batch_size', hp),
        })
        return config
    
    def get_dependent_tunables(self):
        tunables = super().get_dependent_tunables()
        if isinstance(self.optimizer, tuning.Tunable):
            tunables.extend(self.optimizer.get_dependent_tunables())
        return tunables
    
    def get_training_data(self, columns=None):
        if columns is None:
            return self.training_data
        return self.training_data[columns]
    
    def get_validation_data(self, columns=None):
        if columns is None:
            return self.validation_data
        return self.validation_data[columns]
    
    def get_testing_data(self, columns=None):
        if columns is None:
            return self.testing_data
        return self.testing_data[columns]
    
    def get_input_dtype(self, column):
        return tf.float32 if self.training_data[column].dtype.kind in 'iufcb' else tf.string
    
    def get_input_shape(self, column):
        # todo: add vector support
        return (1,)
    
    def get_optimizer(self, hp):
        if isinstance(self.optimizer, optimizers.Optimizer):
            return self.optimizer.build(hp)
        
        return self.optimizer
    
    def get_training_steps_per_epoch(self, hp):
        steps_per_epoch = max(self.training_data.shape[0] // self.get_parameters(hp)['batch_size'], 1)
        if self.max_steps_per_epoch is not None:
            return min(self.max_steps_per_epoch, steps_per_epoch)
        return steps_per_epoch
    
    def get_validation_steps_per_epoch(self, hp):
        return max(self.validation_data.shape[0] // self.get_parameters(hp)['batch_size'], 1)
    
    def __get_phantom_time_data(self, data, time_window):
        
        # we have no time series data
        if time_window <= 1:
            return []
        
        # we dont have a time group, but want to make sure there is enough data in our time window
        if self.time_group_column is None:
            extras = backend.nan_copy_of_size(time_window - 1, data)
            extras = backend.fillna(extras)
            return [extras]
        
        to_append = []
        group_counts = data.groupby(self.time_group_column).size().to_dict()
        for key, value in group_counts.items():
            # append empty data
            extras = backend.nan_copy_of_size(time_window - 1, data)
            extras = backend.fillna(extras)
            extras[self.time_group_column] = key
            to_append.append(extras)
            
        return to_append
    
    def __get_partitioned_data(self, data, input_names, response):
        
        # cache the names of the columns
        names_flt = []
        names_str = []
        for name in input_names:
            if self.get_input_dtype(name) == tf.string:
                names_str.append(name)
            else:
                names_flt.append(name)
        
        # get the str and float parts of the new_data
        Xf = data[names_flt].astype(np.float32).reset_index(drop=True)
        Xs = data[names_str].astype(str).reset_index(drop=True)
        
        # generate Y and W if we have them
        Y = data[response].reset_index(drop=True) if response is not None else None
        W = data[self.weights_column].reset_index(drop=True) if self.weights_column is not None else None
        
        # remove W if no response
        W = None if Y is None else W
        
        return names_flt, Xf, names_str, Xs, Y, W
    
    def __build_data_time_groups(self, data, Xf, Xs, Y, W):
        # mapping for all data arrays
        indices = { '__ALL__' : Xf[[]] }
        Xf_dict = { '__ALL__' : Xf.to_numpy() }
        Xs_dict = { '__ALL__' : Xs.to_numpy() }
        Y_dict = { '__ALL__' : Y.to_numpy() if Y is not None else None }
        W_dict = { '__ALL__' : W.to_numpy() if W is not None else None }
        
        # no time groups
        if self.time_group_column is None:
            return indices, Xf_dict, Xs_dict, Y_dict, W_dict, None
        
        time_group = data[self.time_group_column].reset_index(drop=True)
        for val in time_group.unique():
            flags = time_group == val
            indices[val] = indices['__ALL__'][flags]
            Xf_dict[val] = Xf_dict['__ALL__'][flags]
            Xs_dict[val] = Xs_dict['__ALL__'][flags]
            Y_dict[val] = Y_dict['__ALL__'][flags] if Y_dict['__ALL__'] is not None else None
            W_dict[val] = W_dict['__ALL__'][flags] if W_dict['__ALL__'] is not None else None
            
        return indices, Xf_dict, Xs_dict, Y_dict, W_dict, time_group
    
    def __get_tensorflow_generator(
        self, 
        data, 
        input_names, 
        response, 
        run_forever,
        time_window, 
        batch_size, 
        shuffle, 
        **kwargs
    ):
        
        # get additional buffer data for our time series
        to_append = self.__get_phantom_time_data(data, time_window)
        
        # add the buffer
        original_data_count = data.shape[0]
        if to_append:
            to_append.append(data)
            data = pd.concat(to_append, ignore_index=True)
        
        # split the data into different types
        Xf_names, Xf, Xs_names, Xs, Y, W = self.__get_partitioned_data(data, input_names, response)
        
        # build numpy tensors which will be read in generator, key by time group
        indices, Xf_dict, Xs_dict, Y_dict, W_dict, time_group = self.__build_data_time_groups(data, Xf, Xs, Y, W)
        
        # generator for getting data from the dataframe
        ends = np.arange(data.shape[0] - original_data_count, data.shape[0])
        
        # build the generator
        def generator():
            while True:
                # shuffle if we need to
                if shuffle:
                    np.random.shuffle(ends)

                # yield loop to iterate over the data
                for end in ends:

                    # cut off data after end
                    Xf = Xf_dict['__ALL__'][:end+1]
                    Xs = Xs_dict['__ALL__'][:end+1]
                    Y = Y_dict['__ALL__'][:end+1] if Y_dict['__ALL__'] is not None else None
                    W = W_dict['__ALL__'][:end+1] if W_dict['__ALL__'] is not None else None

                    # filter to sub-array if we have a time group
                    if self.time_group_column is not None:
                        group_type = time_group[end]
                        end_loc = indices[group_type].index.get_loc(end)
                        Xf = Xf_dict[group_type][:end_loc+1]
                        Xs = Xs_dict[group_type][:end_loc+1]
                        Y = Y_dict[group_type][:end_loc+1] if Y is not None else None
                        W = W_dict[group_type][:end_loc+1] if W is not None else None

                    # different fetch for time series and flat array
                    if time_window >= 2:
                        Xf = np.squeeze(np.lib.stride_tricks.sliding_window_view(Xf[-time_window:], (time_window, Xf.shape[1])), axis=(0,1))
                        Xs = np.squeeze(np.lib.stride_tricks.sliding_window_view(Xs[-time_window:], (time_window, Xs.shape[1])), axis=(0,1))
                    else:
                        Xf = Xf[-1]
                        Xs = Xs[-1]

                    X_keys = []
                    X_vals = []
                    if len(Xf_names) > 0:
                        X_keys += Xf_names
                        X_vals += np.split(Xf, Xf.shape[-1], axis=-1)
                    if len(Xs_names) > 0:
                        X_keys += Xs_names
                        X_vals += np.split(Xs, Xs.shape[-1], axis=-1)
                    if Y is None:
                        yield dict(zip(X_keys, X_vals))
                    else:
                        if W is None:
                            yield (dict(zip(X_keys, X_vals)), Y[-1])
                        else:
                            yield (dict(zip(X_keys, X_vals)), Y[-1], W[-1])
                
                # should we terminate the forever loop for a single data pass?
                if not run_forever:
                    break
        
        # we either have a time series or not, if we have a time series
        # we will use the generator to get data to avoid memory overflow
        # otherwise we can use the full data without processing
        input_tensor_specs = []
        for name in input_names:
            dtype = self.get_input_dtype(name)
            shape = self.get_input_shape(name)

            # append time dimension
            if time_window > 1:
                shape = (time_window,) + shape
            
            input_tensor_specs.append(tf.TensorSpec(shape=shape, dtype=dtype))
        
        # build the signature
        output_signature = dict(zip(input_names, input_tensor_specs))

        # append the response and weights
        if Y is not None:
            if W is None:
                output_signature = (output_signature, tf.TensorSpec(shape=(), dtype=tf.float32))
            else:
                output_signature = (output_signature, tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32))
        
        # return the built generator
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
    
    def get_tensorflow_dataset(self, data, input_names, response, run_forever, time_window, hp, shuffle=None):
        config = self.get_parameters(hp)
        
        # option added to not shuffle if we want to do inference
        if shuffle is not None:
            config['shuffle'] = shuffle
        
        tdf = self.__get_tensorflow_generator(
            data, 
            input_names, 
            response, 
            run_forever=run_forever,
            time_window=time_window,
            **config
        )
        
        return tdf.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    def get_training_tensorflow_dataset(self, input_names, response, time_window, hp):
        return self.get_tensorflow_dataset(self.training_data, input_names, response, True, time_window, hp)
    
    def get_validation_tensorflow_dataset(self, input_names, response, time_window, hp):
        return self.get_tensorflow_dataset(self.validation_data, input_names, response, True, time_window, hp)