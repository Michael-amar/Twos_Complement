import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from common import *
import numpy as np
import pandas as pd

def split(sample, win_size, overlapping_percentage):
    sample = sample[:(len(sample)//win_size)*win_size]
    if overlapping_percentage:
        overlap_size = int(np.floor(win_size * overlapping_percentage))
        step_size = win_size - overlap_size
        return window_split(sample,win_size,step_size)
    else:
        return sample.reshape(-1, win_size)

'Generates data for Keras'
class DataGenerator(Sequence):
    def __init__(self, samples_df, n_classes, max_running_time, min_running_time, win_size, sampling_rate, sigma):
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.batch_size = 8
        self.shuffle = False  # at epoch end
        self.overlapping_percentage=0.6
        self.samples_df = samples_df
        self.n_classes = n_classes
        self.loader=default_loader
        self.max_trace_len = int(max_running_time * sampling_rate)
        self.min_trace_len = int(min_running_time * sampling_rate)
        self.sigma = sigma
        self.len = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.samples_df.index) / self.batch_size))

    # Updates indexes after each epoch
    def on_epoch_end(self):
        if self.shuffle:
            self.samples_df = self.samples_df.sample(frac=1, random_state=1234)

    def __getitem__(self, index):
        if index == 0:
            self.history = []
            self.y_true = []
            self.indexes_history = []

        # Generate indexes of the batch
        if index == (self.len - 1):
            indexes = list(range(index*self.batch_size, len(self.samples_df.index)))
        else:
            indexes = list(range(index*self.batch_size,(index+1)*self.batch_size))

        self.indexes_history.append(indexes)

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
    
    def get_history(self):
        history = pd.DataFrame(self.history , columns=["sample_path", "numeric_class"])
        return history

    def __data_generation(self, indexes):
        X = []
        Y = []
        batch_df = self.samples_df.iloc[indexes]
        for i in batch_df.index:
            path = batch_df.loc[i]["path"]
            y = batch_df.loc[i]["numeric_class"]

            trace = self.loader(path, self.sampling_rate, self.sigma)
            trace = adjust_trace_len(trace, self.max_trace_len)
            trace = split(trace, self.win_size, self.overlapping_percentage)       
            X.append(trace)
            Y.append(y)
            self.history.append([(path), y])
        self.y_true.extend(Y)
        X = np.array(X)
        return X, to_categorical(Y, num_classes=self.n_classes)

    def get_shape(self):
        X,y = self.__getitem__(0)
        return X[0].shape