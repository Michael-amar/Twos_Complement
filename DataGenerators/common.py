import numpy as np

def default_loader(path, sampling_rate, sigma):
    
    # load the original trace
    item = np.load(path, encoding='latin1', allow_pickle=True).item()
    trace, running_time = item['trace'], item['running_time']
    
    trace = normalize(trace)
    trace = add_noise(trace, sigma)
    
    # adjust to desired sampling rate
    number_of_points_to_keep = int(running_time * sampling_rate)
    trace = trace[np.linspace(0, trace.size-1, number_of_points_to_keep, dtype=int)]

    return trace

def add_noise(trace, sigma):
    if sigma > 0:
        noise = np.random.normal(0,sigma, trace.shape)
        trace = trace + noise
        return trace
    return trace

def normalize(signal):
    if signal.std() == 0:
        return signal
    return (signal - signal.mean()) / signal.std()

def adjust_trace_len(trace, desired_length):
    if trace.size > desired_length:
         # this is possible beacause the attacks samples are 'unseen' and can be longer than the maximum trace seen in the training samples collection
        num_points_to_drop = trace.size - desired_length 
        indexes_to_drop = np.linspace(0, trace.size-1, num_points_to_drop, dtype=int)
        trace = np.delete(trace, indexes_to_drop)
    else:
        trace = interpolate(trace, desired_length)
    return trace


# interpolate a signal to match desired length
def interpolate(signal, desired_length):
    oversampled_time = np.linspace(0, signal.size-1, desired_length)
    oversampled_signal = np.interp(oversampled_time, np.arange(signal.size)*1, signal)
    return oversampled_signal

# if keep_short_tails is set to True, the slices shorter than window_size at the end of the result will be kept 
def window_split(x, window_size, stride):
    length = len(x)
    splits = []

    for slice_start in range(0, length - window_size + 1, stride):
        slice_end = slice_start + window_size
        splits.append(x[slice_start:slice_end])

    return np.stack(splits, axis=0)