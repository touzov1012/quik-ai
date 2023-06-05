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
        optimizer,
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
        return (1,)
    
    def get_optimizer(self, hp):
        if isinstance(self.optimizer, optimizers.Optimizer):
            return self.optimizer.build(hp)
        
        return self.optimizer
    
    def get_training_steps_per_epoch(self, hp):
        steps_per_epoch = self.training_data.shape[0] // self.get_parameters(hp)['batch_size']
        if self.max_steps_per_epoch is not None:
            return min(self.max_steps_per_epoch, steps_per_epoch)
        return steps_per_epoch
    
    def get_validation_steps_per_epoch(self, hp):
        return self.validation_data.shape[0] // self.get_parameters(hp)['batch_size']
    
    def __get_phantom_time_data(self, data, time_window):
        
        # we dont have a time group, but want to make sure there is enough data in our time window
        if self.time_group_column is None:
            extra_needed = time_window - data.shape[0]
            if extra_needed > 0:
                extras = backend.nan_copy_of_size(extra_needed, data)
                extras = backend.fillna(extras)
                return [extras]
            return []
        
        to_append = []
        group_counts = data.groupby(self.time_group_column).size().to_dict()
        for key, value in group_counts.items():
            extra_needed = time_window - value
            if extra_needed <= 0:
                continue

            # append empty data
            extras = backend.nan_copy_of_size(extra_needed, data)
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
        if to_append:
            to_append.append(data)
            data = pd.concat(to_append, ignore_index=True)
        
        # split the data into different types
        Xf_names, Xf, Xs_names, Xs, Y, W = self.__get_partitioned_data(data, input_names, response)
        
        # build numpy tensors which will be read in generator, key by time group
        indices, Xf_dict, Xs_dict, Y_dict, W_dict, time_group = self.__build_data_time_groups(data, Xf, Xs, Y, W)
        
        # generator for getting data from the dataframe
        starts = np.arange(data.shape[0] - time_window + 1)
        
        # build the generator
        def generator():
            while True:
                # shuffle if we need to
                if shuffle:
                    np.random.shuffle(starts)

                # yield loop to iterate over the data
                for start in starts:

                    # cut off data prior to start
                    Xf = Xf_dict['__ALL__'][start:]
                    Xs = Xs_dict['__ALL__'][start:]
                    Y = Y_dict['__ALL__'][start:] if Y_dict['__ALL__'] is not None else None
                    W = W_dict['__ALL__'][start:] if W_dict['__ALL__'] is not None else None

                    # filter to sub-array if we have a time group
                    if self.time_group_column is not None:
                        group_type = time_group[start]
                        start_loc = indices[group_type].index.get_loc(start)
                        Xf = Xf_dict[group_type][start_loc:]
                        Xs = Xs_dict[group_type][start_loc:]
                        Y = Y_dict[group_type][start_loc:] if Y is not None else None
                        W = W_dict[group_type][start_loc:] if W is not None else None

                    # if we dont have enough data
                    if time_window > Xf.shape[0]:
                        continue

                    # different fetch for time series and flat array
                    if time_window >= 2:
                        Xf = np.squeeze(np.lib.stride_tricks.sliding_window_view(Xf[:time_window], (time_window, Xf.shape[1])), axis=(0,1))
                        Xs = np.squeeze(np.lib.stride_tricks.sliding_window_view(Xs[:time_window], (time_window, Xs.shape[1])), axis=(0,1))
                    else:
                        Xf = Xf[0]
                        Xs = Xs[0]

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
                            yield (dict(zip(X_keys, X_vals)), Y[time_window-1])
                        else:
                            yield (dict(zip(X_keys, X_vals)), Y[time_window-1], W[time_window-1])
                
                # should we terminate the forever loop for a single data pass?
                if not run_forever:
                    break
        
        # we either have a time series or not, if we have a time series
        # we will use the generator to get data to avoid memory overflow
        # otherwise we can use the full data without processing
        X_vals = []
        for name in input_names:
            dtype = self.driver.get_input_dtype(name)
            shape = self.driver.get_input_shape(name)

            # append time dimension
            if time_window > 1:
                shape = (time_window,) + shape
            
            X_vals.append(tf.TensorSpec(shape=shape, dtype=dtype))
        
        # build the signature
        output_signature = dict(zip(X_keys, X_vals))

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
    
    def get_tensorflow_dataset(self, data, input_names, response, run_forever, time_window, hp):
        config = self.get_parameters(hp)
        
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
    
    
def build(
    hyper_model,
    tuner,
    early_stopping_tune=10, 
    early_stopping_full=10,
    full_rounds=5,
    working_dir='.', 
    verbose=1
):
    # create working directory for the build
    build_dir = backend.create_unique_dir(working_dir=working_dir)

    backend.info('Checking for hyper-parameters to tune ...', verbose)

    # set up the keras tuner parameters
    checkpoint_monitor = 'val_' + hyper_model.head.monitor()

    # create the tuner
    tuner = self.tuner(
        hyper_model, 
        objective=kt.Objective(checkpoint_monitor, direction=hyper_model.head.objective_direction),
        directory=backend.join_path(build_dir, 'tuner'),
        project_name='kt_tuner', 
        **self.tuner.get_tuner_params()
    )

    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=checkpoint_monitor, 
        patience=early_stopping_tune,
        mode=hyper_model.head.objective_direction
    )

    # search the space
    tuner.search(
        epochs=self.tuner.get_tuner_epochs(), 
        callbacks=[tf.keras.callbacks.TerminateOnNaN(), early_stopping], 
        verbose=backend.clamp(verbose)
    )

    # get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)
    if best_hps is not None and len(best_hps) > 0:
        best_hps = best_hps[0].values

        # apply to all parameters tuned through model
        tunables = hyper_model.get_dependent_tunables()
        for tunable in tunables:
            tunable._apply_hp(best_hps)

    # early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=checkpoint_monitor, 
        patience=early_stopping_full,
        mode=hyper_model.head.objective_direction
    )

    # checkpoint to catch best validation
    checkpoint_filepath = backend.join_path(build_dir, 'cp')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=checkpoint_monitor,
        mode=hyper_model.head.objective_direction,
        save_best_only=True,
        verbose=backend.clamp(verbose-1)
    )

    # full training
    for i in range(full_rounds):

        # clean up last model
        if i > 0:
            del model
            backend.clean_tensorflow()

        # re-initialize the model
        model = hyper_model.build(None)

        # fit the model
        history = hyper_model.fit(
            None, 
            model, 
            epochs=1_000_000, 
            callbacks=[tf.keras.callbacks.TerminateOnNaN(), model_checkpoint_callback, early_stopping], 
            verbose=backend.clamp(verbose-2)
        )

        # process history to get extrema score
        scores = history.history[checkpoint_monitor]
        curr_score = min(scores) if hyper_model.head.objective_direction == 'min' else max(scores)

        # log the current run
        backend.info('Round %s best score: %.4f' % (i+1, curr_score), verbose)

    # load the best weights
    model.load_weights(checkpoint_filepath)

    return model