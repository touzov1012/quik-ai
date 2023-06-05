import os
import gc
import uuid
import threading
import logging

import pandas as pd
import numpy as np
import tensorflow as tf

lock = threading.Lock()

def clean_tensorflow():
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect();

def create_unique_dir(working_dir='.'):
    with lock:
        unique_dir_name = str(uuid.uuid4())
        new_dir_path = os.path.join(working_dir, unique_dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
    return new_dir_path

def join_path(path, *paths):
    return os.path.join(path, *paths)

def info(message, verbose):
    if verbose < 1:
        return
    logging.info(message)

def warning(message):
    logging.warning(message)

def error(message):
    logging.error(message)
    
def clamp(value, lower=0, upper=1):
    return min(max(value,0), 1)

def fillna(data, int_token=0, float_token=0.0, str_token='[UNK]', dt_token=pd.to_datetime(0)):
    
    for column in data:
        if data[column].dtype.kind in 'biu':
            data[column] = data[column].fillna(int_token)
        elif data[column].dtype.kind in 'fc':
            data[column] = data[column].fillna(float_token)
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].fillna(dt_token)
        elif data[column].dtype == object:
            data[column] = data[column].fillna(str_token)
    
    return data

def nan_copy_of_size(n, data):
    new_df = pd.DataFrame(np.nan, index=np.arange(n), columns=data.columns)
    
    for col in data.columns:
        if data[col].dtype.kind in 'biu':
            new_df[col] = new_df[col].fillna(0).astype(data[col].dtype)
        else:
            new_df[col] = new_df[col].astype(data[col].dtype)
    
    return new_df