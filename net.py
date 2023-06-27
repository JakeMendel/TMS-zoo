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
from dataclasses import dataclass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
mpl.style.use('seaborn-v0_8')
mpl.rcParams['figure.figsize'] = (15,10)
fontsize = 40
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize

#%%

class Net(nn.Module):
    """
    Basic Network class for linear transformation with non-linear activations
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 tied = True,
                 final_bias = False,
                 hidden_bias = False,
                 nonlinearity = F.relu,
                 unit_weights=False,
                 learnable_scale_factor = False, 
                 standard_magnitude = False,
                 initial_scale_factor = 1.0,
                 initial_embed = None, 
                 initial_bias = None):
        
        super().__init__()

        # Set the dimensions and parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.final_bias = final_bias
        self.unit_weights = unit_weights
        self.learnable_scale_factor = learnable_scale_factor
        self.standard_magnitude = standard_magnitude

        # Define the input layer (embedding)
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias=hidden_bias)

        # Set initial embeddings if provided
        if initial_embed is not None:
            self.embedding.weight.data = initial_embed

        # Define the output layer (unembedding)
        self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)

        # Set initial bias if provided
        if initial_bias is not None:
            self.unembedding.bias.data = initial_bias

        # If standard magnitude is set, normalize weights and maintain average norm
        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim = 0).mean()
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm

        # If unit weights is set, normalize weights
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)

        # Tie the weights of embedding and unembedding layers
        if tied:
            self.unembedding.weight = torch.nn.Parameter(self.embedding.weight.transpose(0, 1))

        # Set learnable scale factor
        if self.learnable_scale_factor:
            self.scale_factor = nn.Parameter(torch.tensor(initial_scale_factor))
        else:
            self.scale_factor = initial_scale_factor

    def forward(self, x, hooked = False):
        """
        Forward pass through the network
        """
        # Apply the same steps for weights as done during initialization
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)
        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim = 0).mean()
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
        if self.tied:
            self.unembedding.weight.data = self.embedding.weight.data.transpose(0, 1)

        # In case we want to track the activations
        if hooked:
            activations = {}
            activations['res_pre'] = self.embedding(x)
            activations['unembed_pre'] = self.unembedding(activations['res_pre'])
            activations['output'] = self.scale_factor * self.nonlinearity(activations['unembed_pre'])
            return activations['output'], activations
        else:
            x = self.embedding(x)
            x = self.unembedding(x)
            x = self.nonlinearity(x)
            return self.scale_factor * x

#%%
class CustomScheduler(_LRScheduler):
    """
    Custom learning rate scheduler class that inherits from _LRScheduler.
    
    Parameters:
    optimizer (Optimizer): Optimizer.
    warmup_steps (int): Number of warmup steps.
    max_lr (float): Maximum learning rate after warmup.
    decay_factor (float): Decay factor for the exponential decay phase.
    last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_steps, max_lr, decay_factor, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.decay_factor = decay_factor
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate according to the number of steps: linear warmup followed by exponential decay.
        """
        if self.last_epoch < self.warmup_steps:  # warmup phase
            return [base_lr + self.last_epoch * ((self.max_lr - base_lr) / self.warmup_steps) for base_lr in self.base_lrs]
        else:  # decay phase
            return [self.max_lr * (self.decay_factor ** (self.last_epoch - self.warmup_steps)) for _ in self.base_lrs]


def train(model, loader, criterion, optimizer, epochs, logging_loss, plot_rate, store_rate, scheduler = None, lr_print_rate = 0):
    """
    Train function to train the model and print the loss.
    
    Parameters:
    model (nn.Module): The model to be trained.
    loader (DataLoader): The data loader.
    criterion (nn.Module): The loss function.
    optimizer (Optimizer): The optimizer.
    epochs (int): Number of epochs.
    logging_loss (bool): If True, log the loss value.
    plot_rate (int): The rate at which the loss plot is updated.
    store_rate (int): The rate at which model parameters are stored.
    scheduler (_LRScheduler, optional): The learning rate scheduler. If None, no scheduler is used.
    lr_print_rate (int): The rate at which learning rate is printed. If 0, learning rate is not printed.
    """
    weights_history = {k:[v.detach().numpy().copy()] for k,v in dict(model.named_parameters()).items()}  # Store the weights here
    model_history = {} #store model here
    losses = []  # List to store the losses

    # Training loop
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()  # Zero out any gradients.
            outputs = model(batch)
            loss = criterion(outputs, batch)  # Compute the loss.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update weights.
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)  # Compute average loss
        if logging_loss:  # If logging loss is True
            losses.append(avg_loss)
            if plot_rate > 0 and (epoch + 1) % plot_rate == 0:  # If plot rate is greater than zero and the epoch number is a multiple of the plot rate
                plt.figure(figsize=(5,5))
                plt.plot(losses)
                plt.show()
        if (epoch + 1) % store_rate == 0:  # If the epoch number is a multiple of the store rate
            for k,v in dict(model.named_parameters()).items():
                weights_history[k].append(v.detach().numpy().copy())
            model_history[epoch] = copy.deepcopy(model)  # Store the model state
        if scheduler is not None:
            scheduler.step()  # Step the learning rate scheduler
        if lr_print_rate > 0 and (epoch % lr_print_rate) == 0:  # If the learning rate print rate is greater than zero and the epoch number is a multiple of the learning rate print rate
            print(optimizer.param_groups[0]['lr'])
    return losses, weights_history, model_history  # Return the weights history

def set_seed(seed):
    """
    Set the seed for reproducibility.
    
    Parameters:
    seed (int): The seed to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
