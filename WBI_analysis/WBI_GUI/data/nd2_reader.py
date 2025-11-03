# filepath: nd2_neuron_gui/src/data/nd2_reader.py
import numpy as np
from pims import ND2Reader_SDK

def read_nd2_file(path):
    images = ND2Reader_SDK(path)
    images.bundle_axes = 'zyx'
    images.iter_axes = 'tc'
    return images

def read_neuron_data(path):
    raw_data = np.load(path, allow_pickle=True)
    n_data = raw_data.item()
    return n_data

def read_neuron_signals(path):
    signals = np.load(path, allow_pickle=True)
    return signals

def read_cluster(path):
    cluster_neuron = np.load(path, allow_pickle=True)
    return cluster_neuron