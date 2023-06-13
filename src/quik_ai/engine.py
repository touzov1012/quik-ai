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
    
    def get_data_tensor(self, data, columns):
        '''
        Build a tensor from the input, the result will 
        be shape (n_rows, n_columns, common_element_shape)
        
        data:      the pandas array
        columns:   the columns to extract from the array
        '''
        
        if not isinstance(columns, (list, tuple)):
            columns = [columns]

        # convert to numpy
        arr = data[columns].to_numpy()

        # first, we reshape the 2D array to a 1D array
        arr_reshaped = arr.reshape(-1)

        # then, we stack the 2D arrays along new dimensions
        result = np.stack(arr_reshaped)

        # reshape the result back to the desired shape
        entry_shape = arr[0,0].shape if not isinstance(arr[0,0], str) else ()
        result = result.reshape(*arr.shape, *entry_shape)

        return result
    
    def get_input_dtype(self, column):
        cell = self.training_data[column].iloc[0]
        if isinstance(cell, str):
            return tf.string
        return tf.float32 if cell.flatten()[0].dtype.kind in 'iufcb' else tf.string
    
    def get_input_shape(self, column, time_window):
        cell = self.training_data[column].iloc[0]
        if isinstance(cell, str):
            return (1,) if time_window <= 1 else (time_window, 1)
        shape = cell.shape if len(cell.shape) > 0 else (1,)
        return shape if time_window <= 1 else (time_window,) + shape
    
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
    
    def __get_partitioned_data(self, data, input_names, response, time_window):
        
        # cache the names of the columns corresponding to each tensor type
        names_map = {}
        for name in input_names:
            key = (self.get_input_dtype(name), self.get_input_shape(name, time_window))
            array = names_map.get(key,[])
            array.append(name)
            names_map[key] = array
        
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
        tensor_map = {}
        for key, value in names_map.items():
            tensor_map[key] = self.get_data_tensor(data, value)
        
        # generate y and w if we have them
        y = np.squeeze(self.get_data_tensor(data, response), axis=1) if response is not None else None
        w = data[self.weights_column].to_numpy() if self.weights_column is not None else None
        
        # remove w if no response
        w = None if y is None else w
        
        return {
            'row_order' : row_order,
            'group_order' : group_order,
            'names_map' : names_map,
            'tensor_map' : tensor_map,
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
        cache = self.__get_partitioned_data(data, input_names, response, time_window)
        
        # get the order of our results
        row_order = cache['row_order']
        
        # sliced tensor input
        sliced_tensors = {}
        
        # build the generator
        def generator():
            while True:
                # shuffle if we need to
                if shuffle:
                    np.random.shuffle(row_order)

                # yield loop to iterate over the data
                for end in row_order:
                    
                    # clear the previous slices
                    sliced_tensors.clear()
                    
                    # filter to sub-array depending on if we have a time group
                    if self.time_group_column is None:
                        for key, value in cache['tensor_map'].items():
                            sliced_tensors[key] = value[:end+1]
                    else:
                        group_id = cache['group_order'][end]
                        for key, value in cache['tensor_map'].items():
                            sliced_tensors[key] = value[end-group_id:end+1]

                    # different fetch for time series and flat array, we may need to pad
                    # the time series if it there is not enough history
                    for key in sliced_tensors.keys():
                        fill_type = '[UNK]' if key[0] == tf.string else 0
                        sliced_tensors[key] = backend.get_k_from_end(sliced_tensors[key], time_window, fill=fill_type)
                        
                    # combine the predictor names
                    x_keys = []
                    for value in cache['names_map'].values():
                        x_keys += value
                    
                    # combine the predictor vectors
                    x_values = []
                    for key, value in sliced_tensors.items():
                        splits = np.split(value, value.shape[1], axis=1)
                        for split in splits:
                            squeezed = np.squeeze(split, axis=(0,1)) if time_window <= 1 else np.squeeze(split, axis=1)
                            x_values.append(np.reshape(squeezed, key[1]))
                    
                    if cache['y'] is None:
                        yield dict(zip(x_keys, x_values))
                    else:
                        if cache['w'] is None:
                            yield (dict(zip(x_keys, x_values)), cache['y'][end])
                        else:
                            yield (dict(zip(x_keys, x_values)), cache['y'][end], cache['w'][end])
                
                # should we terminate the forever loop for a single data pass?
                if not run_forever:
                    break
        
        # we either have a time series or not, if we have a time series
        # we will use the generator to get data to avoid memory overflow
        # otherwise we can use the full data without processing
        input_tensor_specs = []
        for name in input_names:
            dtype = self.get_input_dtype(name)
            shape = self.get_input_shape(name, time_window)
            
            input_tensor_specs.append(tf.TensorSpec(shape=shape, dtype=dtype))
        
        # build the signature
        output_signature = dict(zip(input_names, input_tensor_specs))

        # append the response and weights
        if cache['y'] is not None:
            # get the response shape
            response_shape = self.get_input_shape(response, 1)
            response_shape = () if response_shape == (1,) else response_shape
            
            # append to the signature
            output_signature = (output_signature, tf.TensorSpec(shape=response_shape, dtype=self.get_input_dtype(response)))
        
        # append the weights to the signature
        if cache['w'] is not None:
            output_signature += (tf.TensorSpec(shape=(), dtype=tf.float32),)
        
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