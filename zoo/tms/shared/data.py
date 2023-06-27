import copy
import os
import random
import time
from abc import ABC
from itertools import combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from IPython.display import clear_output
from scipy.spatial import ConvexHull
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SyntheticDataset(Dataset, ABC):
    def __init__(self, num_samples, f, s):
        """
        Initialize the  object.
        
        Args:
            num_samples: The number of samples to generate.
            f: The length of the feature vector.
            s: Sparsity: the probability that a given feature is zero.
        """
        self.num_samples = num_samples  # The number of samples in the dataset
        self.f = f  # The size of the feature vector for each sample
        self.s = s
        self.data = self.generate_data()  # Generate the synthetic data

    def generate_data(self):
        raise NotImplementedError

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
    

class SyntheticSparse(SyntheticDataset):       
    """
    This class creates a synthetic dataset where each sample is a vector which has indices which are zero with probability s and uniform between 0 and 1 otherwise
    """
    
    def generate_data(self):
        """
        Generate the synthetic data.
        
        Returns:
            A tensor of size (num_samples x f) with each row containing one random number 
            between 0 and 1 at a random index and the rest are all zeros.
        """
        data = torch.zeros((self.num_samples, self.f))  # Initialize the data tensor with zeros

        for i in range(self.num_samples):  # For each sample
            data[i] = torch.rand(self.f) * torch.bernoulli(torch.ones(self.f) * (1-self.s))
        
        return data

    

class SyntheticSparseNormalised(SyntheticDataset):
    """
    This class creates a synthetic dataset where each sample is a vector which has indices which are zero with probability s and 1 otherwise
    """
    def generate_data(self):
        """
        Generate the synthetic data.
        
        Returns:
            A tensor of size (num_samples x f) with each row containing one random number 
            between 0 and 1 at a random index and the rest are all zeros.
        """
        data = torch.zeros((self.num_samples, self.f))  # Initialize the data tensor with zeros
        for i in range(self.num_samples):  # For each sample
            data[i] = torch.bernoulli(torch.ones(self.f)*(1-self.s))
        return data
    
