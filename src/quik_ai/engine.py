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
    
    def __get_partitioned_data(self, data, input_names, response):
        
        # cache the names of the columns
        names_flt = []
        names_str = []
        for name in input_names:
            if self.get_input_dtype(name) == tf.string:
                names_str.append(name)
            else:
                names_flt.append(name)
        
        # the array may be shuffled, but the index of the true order remains
        # we need to build a map to the locations of the new ordered elements
        row_order = data.index.to_numpy()
        
        # create a dictionary from the shuffled list with numbers as keys and their indices as values
        row_order = {number: index for index, number in enumerate(row_order)}

        # create a new list with the indices of the numbers from 0 to n-1 in the shuffled list
        row_order = [row_order[i] for i in range(data.shape[0])]
        
        # cache the group order
        group_order = data.__group_id__.to_numpy() if self.time_group_column is not None else None
        
        # get the str and float parts of the new_data
        x_flt = data[names_flt].astype(np.float32).to_numpy()
        x_str = data[names_str].astype(str).to_numpy()
        
        # generate y and w if we have them
        y = data[response].to_numpy() if response is not None else None
        w = data[self.weights_column].to_numpy() if self.weights_column is not None else None
        
        # remove w if no response
        w = None if y is None else w
        
        return {
            'row_order' : row_order,
            'group_order' : group_order,
            'names_flt' : names_flt,
            'names_str' : names_str,
            'x_flt' : x_flt,
            'x_str' : x_str,
            'y' : y,
            'w' : w,
        }
    
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
        # presort by index
        data = data.sort_index()
        
        # if we have a time group, first sort by group
        if self.time_group_column is not None:
            data = data.reset_index().rename(columns={'index': '__group_id__'})
            data = data.sort_values(by=[self.time_group_column, '__group_id__'])
            data['__group_id__'] = data.groupby(self.time_group_column).cumcount()
        
        # split the data into different type tensors and get the
        # order of the rows as well as the element of the row in
        # each group if we have a time group
        cache = self.__get_partitioned_data(data, input_names, response)
        
        # get the order of our results
        row_order = cache['row_order']
        
        # build the generator
        def generator():
            while True:
                # shuffle if we need to
                if shuffle:
                    np.random.shuffle(row_order)

                # yield loop to iterate over the data
                for end in row_order:
                    
                    # filter to sub-array depending on if we have a time group
                    if self.time_group_column is None:
                        x_flt = cache['x_flt'][:end+1]
                        x_str = cache['x_str'][:end+1]
                        y = cache['y'][:end+1] if cache['y'] is not None else None
                        w = cache['w'][:end+1] if cache['w'] is not None else None
                    else:
                        group_id = cache['group_order'][end]
                        x_flt = cache['x_flt'][end-group_id:end+1]
                        x_str = cache['x_str'][end-group_id:end+1]
                        y = cache['y'][end-group_id:end+1] if cache['y'] is not None else None
                        w = cache['w'][end-group_id:end+1] if cache['w'] is not None else None

                    # different fetch for time series and flat array, we may need to pad
                    # the time series if it there is not enough history
                    if time_window >= 2:
                        x_flt = backend.get_k_from_end(x_flt, time_window, fill=0)
                        x_str = backend.get_k_from_end(x_str, time_window, fill='[UNK]')
                    else:
                        x_flt = x_flt[-1]
                        x_str = x_str[-1]
                        
                    # combine the predictor names
                    x_keys = [*cache['names_flt'], *cache['names_str']]
                    
                    # combine the predictor vectors
                    x_flt_vals = [] if x_flt.shape[-1] == 0 else np.split(x_flt, x_flt.shape[-1], axis=-1)
                    x_str_vals = [] if x_str.shape[-1] == 0 else np.split(x_str, x_str.shape[-1], axis=-1)
                    x_vals = [*x_flt_vals, *x_str_vals]
                    
                    if y is None:
                        yield dict(zip(x_keys, x_vals))
                    else:
                        if w is None:
                            yield (dict(zip(x_keys, x_vals)), y[-1])
                        else:
                            yield (dict(zip(x_keys, x_vals)), y[-1], w[-1])
                
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
        if cache['y'] is not None:
            if cache['w'] is None:
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