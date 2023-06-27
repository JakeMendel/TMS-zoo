#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
import plotly.graph_objs as go
import matplotlib as mpl
import torch.nn.functional as F
import random
import plotly.graph_objects as go
import copy
from itertools import combinations
from torch.optim.lr_scheduler import _LRScheduler
from fancy_einsum import einsum
from scipy.spatial import ConvexHull
import os

device = t.device("cuda" if t.cuda.is_available() else "cpu")
#%%
# This class creates a synthetic dataset where each sample is a vector which has indices which are zero with probability s and uniform between 0 and 1 otherwise
class SyntheticSparse(Dataset):
    def __init__(self, num_samples, f, s):
        """
        Initialize the SyntheticDataset object.
        
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
        """
        Generate the synthetic data.
        
        Returns:
            A tensor of size (num_samples x f) with each row containing one random number 
            between 0 and 1 at a random index and the rest are all zeros.
        """
        data = torch.zeros((self.num_samples, self.f))  # Initialize the data tensor with zeros
        for i in range(self.num_samples):  # For each sample
            data[i] = torch.rand(self.f) * torch.bernoulli(torch.ones(self.f)*(1-self.s))
        return data
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]
    
#%%
# This class creates a synthetic dataset where each sample is a vector which has indices which are zero with probability s and 1 otherwise
class SyntheticSparseNormalised(Dataset):
    def __init__(self, num_samples, f, s):
        """
        Initialize the SyntheticDataset object.
        
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
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

#%%
# This class creates a synthetic dataset where each sample is a vector of zeros except 
# for a set of k random indices which are assigned random values between 0 and 1.
class SyntheticKHot(Dataset):
    def __init__(self, num_samples, f, k):
        """
        Initialize the SyntheticDataset object.
        
        Args:
            num_samples: The number of samples to generate.
            f: The length of the feature vector.
            k: The number of "hot" indices in each sample.
        """
        self.num_samples = num_samples  # The number of samples in the dataset
        self.f = f  # The size of the feature vector for each sample
        self.k = k
        self.data = self.generate_data()  # Generate the synthetic data
        
    def generate_data(self):
        """
        Generate the synthetic data.
        
        Returns:
            A tensor of size (num_samples x f) with each row containing one random number 
            between 0 and 1 at a random index and the rest are all zeros.
        """
        data = torch.zeros((self.num_samples, self.f))  # Initialize the data tensor with zeros
        for i in range(self.num_samples):  # For each sample
            indices = random.sample(range(self.f), self.k)
            for index in indices:
                data[i, index] = torch.rand(1)  # Assign a random value between 0 and 1 at the chosen index
        return data
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# This class creates a synthetic dataset where each sample is a vector with k indices set to 1 
# (i.e., "hot") and the rest set to 0. All possible combinations of f choose k are included in the dataset.
class SyntheticKHotNormalised(Dataset):
    def __init__(self, f, k):
        """
        Initialize the SyntheticKHot object.
        
        Args:
            f: The length of the feature vector.
            k: The number of "hot" indices in each sample.
        """
        self.f = f  # The length of the feature vector
        self.k = k  # The number of "hot" indices in each sample
        self.data = []  # Initialize the data list

        # Generate all possible combinations of f choose k
        for indices in combinations(range(f), k):
            vec = torch.zeros(f)  # Initialize a vector of zeros
            vec[list(indices)] = 1  # Set the indices in the combination to 1
            self.data.append(vec)  # Add the vector to the dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # If idx is a tensor, convert it to a list
            idx = idx.tolist()
        return self.data[idx]
