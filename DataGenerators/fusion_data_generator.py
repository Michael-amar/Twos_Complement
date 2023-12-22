import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
from common import *
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
class FusionDataGenerator(Sequence):
    def __init__(self, samples_df, mode, n_classes, max_running_time, min_running_time,  sampling_rate_power, sampling_rate_em, overlapping_percentage, em_noise_sigma, power_noise_sigma):
        self.mode = mode # 'early' /'intermediate'

        if self.mode == 'early':
            self.get_shape = self.get_shape_early
            self.__data_generation = self.__data_generation_early
        else:
            self.get_shape = self.get_shape_intermediate
            self.__data_generation = self.__data_generation_intermediate

        self.window_size_power = 30
        self.window_size_em = 3

        # frequency of points to load in each sample (up to 2.5GHz for power samples and 56Mhz for EM samples)
        self.sampling_rate_power = sampling_rate_power 
        self.sampling_rate_em = sampling_rate_em

        self.batch_size = 8
        self.shuffle = False  # at epoch end
        self.overlapping_percentage = overlapping_percentage

        self.max_trace_len_power = int(max_running_time * sampling_rate_power)
        self.max_trace_len_em = int(max_running_time * sampling_rate_em)
        self.min_trace_len_power = int(min_running_time * sampling_rate_power)
        self.min_trace_len_em = int(min_running_time * sampling_rate_em)

        self.samples_df = samples_df
        self.n_classes = n_classes
        self.loader= default_loader

        self.em_noise_sigma = em_noise_sigma
        self.power_noise_sigma = power_noise_sigma

        self.len = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.ceil(len(self.samples_df.index) / self.batch_size))
        return int(np.floor(len(self.samples_df.index) / self.batch_size))

    # Updates indexes after each epoch
    def on_epoch_end(self):
        if self.shuffle:
            self.samples_df = self.samples_df.sample(frac=1, random_state=1234)

    def __getitem__(self, index):
        if index == 0:
            # for debugging
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
        history = pd.DataFrame(self.history , columns=["POWER_path", "EM_path", "numeric_class"])
        return history
    
    def __data_generation_intermediate(self, indexes):
        POWER_X = []
        EM_X = []
        Y = []
        batch_df = self.samples_df.iloc[indexes]
        for i in batch_df.index:
            power_path = batch_df.loc[i]["POWER_path"]
            power_trace = self.loader(power_path, self.sampling_rate_power, self.power_noise_sigma)
            power_trace = adjust_trace_len(power_trace, self.max_trace_len_power)
            power_trace = split(power_trace, self.window_size_power, self.overlapping_percentage)
            POWER_X.append(power_trace)
            
            em_path = batch_df.loc[i]["EM_path"]
            em_trace = self.loader(em_path, self.sampling_rate_em, self.em_noise_sigma)
            em_trace = adjust_trace_len(em_trace, self.max_trace_len_em)
            em_trace = split(em_trace, self.window_size_em, self.overlapping_percentage)
            EM_X.append(em_trace)

            y = batch_df.loc[i]["numeric_class"]
            Y.append(y)

            self.history.append([power_path,em_path, y])
        self.y_true.extend(Y)
        POWER_X = np.array(POWER_X)
        EM_X = np.array(EM_X)
        Y = np.array(Y)
        y_categorical = to_categorical(Y, num_classes=self.n_classes)
        return [POWER_X, EM_X], y_categorical


    def get_shape_intermediate(self):
        [POWER_X, EM_X], y = self.__getitem__(0)
        return POWER_X[0].shape, EM_X[0].shape


    def __data_generation_early(self, indexes):
        CONCATENATED_X = []
        Y = []
        batch_df = self.samples_df.iloc[indexes]
        for i in batch_df.index:
            power_path = batch_df.loc[i]["POWER_path"]
            power_trace = self.loader(power_path, self.sampling_rate_power, self.power_noise_sigma)
            power_trace = adjust_trace_len(power_trace, self.max_trace_len_power)

            
            em_path = batch_df.loc[i]["EM_path"]
            em_trace = self.loader(em_path, self.sampling_rate_em, self.em_noise_sigma)
            em_trace = adjust_trace_len(em_trace, self.max_trace_len_em)

            concatenated_trace = np.concatenate([em_trace, power_trace])
            concatenated_trace = split(concatenated_trace, self.window_size_power, self.overlapping_percentage)
            CONCATENATED_X.append(concatenated_trace)
            y = batch_df.loc[i]["numeric_class"]
            Y.append(y)

            self.history.append([power_path,em_path, y])
        self.y_true.extend(Y)
        CONCATENATED_X = np.array(CONCATENATED_X)
        Y = np.array(Y)
        y_categorical = to_categorical(Y, num_classes=self.n_classes)
        return CONCATENATED_X, y_categorical


    def get_shape_early(self):
        X, y = self.__getitem__(0)
        return X[0].shape
