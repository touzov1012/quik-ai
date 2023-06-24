from quik_ai import tuning
from quik_ai import optimizers
from quik_ai import backend

import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
import os

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
        working_dir='./tmp', 
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
        self.working_dir = working_dir
        self.file_dir = backend.create_unique_dir(working_dir=self.working_dir)
    
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
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < steps_per_epoch:
            return self.max_steps_per_epoch
        return None
    
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
        y = np.float32(np.squeeze(self.get_data_tensor(data, response), axis=1)) if response is not None else None
        w = data[self.weights_column].to_numpy(dtype=np.float32) if self.weights_column is not None else None
        
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
    
    def __bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))
    
    def __float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    def __serialize_example(self, x, y=None, w=None):
        input_features = {name: self.__bytes_feature(tensor) for name, tensor in x.items()}
        if y is not None:
            input_features['__response__'] = self.__bytes_feature(y)
        if w is not None:
            input_features['__weights__'] = self.__float_feature(w)
        example_proto = tf.train.Example(features=tf.train.Features(feature=input_features))
        return example_proto.SerializeToString()
    
    def __get_tensorflow_generator(
        self, 
        data, 
        input_names, 
        response, 
        time_window, 
        batch_size,
        shuffle, 
        name,
        reinit=True,
        **kwargs
    ):  
        # file name
        filepath = backend.join_path(self.file_dir, '%s.tfrecords' % name)
        
        # check if we already have this record dataset
        # if we do, and we want to reinit, then delete it
        if not os.path.exists(filepath) or reinit:
            
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

            # shuffle if we need to
            if shuffle:
                np.random.shuffle(row_order)

            # combine the predictor names
            x_keys = []
            for value in cache['names_map'].values():
                x_keys += value
                
            # yield loop to iterate over the data
            with tf.io.TFRecordWriter(filepath) as writer:
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

                    # combine the predictor vectors
                    x_values = []
                    for key, value in sliced_tensors.items():
                        splits = np.split(value, value.shape[1], axis=1)
                        for split in splits:
                            squeezed = np.squeeze(split, axis=(0,1)) if time_window <= 1 else np.squeeze(split, axis=1)
                            squeezed = np.reshape(squeezed, key[1])
                            x_values.append(tf.convert_to_tensor(squeezed, dtype=key[0]))

                    if cache['y'] is None:
                        example = self.__serialize_example(dict(zip(x_keys, x_values)))
                    elif cache['w'] is None:
                        example = self.__serialize_example(dict(zip(x_keys, x_values)), cache['y'][end])
                    else:
                        example = self.__serialize_example(dict(zip(x_keys, x_values)), cache['y'][end], cache['w'][end])
                    
                    # write the observation
                    writer.write(example)
            
        # create the raw data
        raw_dataset = tf.data.TFRecordDataset(filepath)
        
        # create the description of each feature
        feature_description = {key: tf.io.FixedLenFeature([], tf.string) for key in input_names}
        if response is not None:
            feature_description['__response__'] = tf.io.FixedLenFeature([], tf.string)
        if self.weights_column is not None and response is not None:
            feature_description['__weights__'] = tf.io.FixedLenFeature([], tf.float32)
        
        # create a mapping to parse our serialized data
        def _parse_function(example_proto):
            features = tf.io.parse_single_example(example_proto, feature_description)
            inputs = {name: tf.io.parse_tensor(features[name], out_type=self.get_input_dtype(name)) 
                      for name in features if name != '__response__' and name != '__weights__'}
            if response is None:
                return inputs
            elif self.weights_column is None:
                return inputs, tf.io.parse_tensor(features['__response__'], out_type=tf.float32)
            else:
                return inputs, tf.io.parse_tensor(features['__response__'], out_type=tf.float32), features['__weights__']
        
        # return the deserialized dataset of observations
        tdf = raw_dataset.map(_parse_function)
        
        # if we shuffle
        if shuffle:
            tdf = tdf.shuffle(2048, seed=1337)
        
        return tdf.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def get_tensorflow_dataset(
        self, 
        data, 
        input_names, 
        response,
        time_window, 
        hp, 
        shuffle=None, 
        name='testing',
        reinit=True
    ):
        # we may have tuning parameters
        config = self.get_parameters(hp)
        
        # option added to not shuffle if we want to do inference
        if shuffle is not None:
            config['shuffle'] = shuffle
        
        return self.__get_tensorflow_generator(
            data, 
            input_names, 
            response, 
            time_window=time_window,
            name=name,
            reinit=reinit,
            **config
        )
    
    def get_training_tensorflow_dataset(self, input_names, response, time_window, hp):
        return self.get_tensorflow_dataset(
            self.training_data, 
            input_names, 
            response, 
            time_window, 
            hp, 
            name='training', 
            reinit=False
        )
    
    def get_validation_tensorflow_dataset(self, input_names, response, time_window, hp):
        return self.get_tensorflow_dataset(
            self.validation_data, 
            input_names, 
            response, 
            time_window, 
            hp, 
            name='validation', 
            reinit=False
        )