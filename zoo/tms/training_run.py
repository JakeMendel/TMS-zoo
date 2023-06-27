#%%
import functions_and_classes as functions
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
import copy
from itertools import combinations
import os

#%%
# your chosen seed
chosen_seed = 12
functions.set_seed(chosen_seed)

#Checking for errors
lr_print_rate = 0


# Configure the hyperparameters
f = 6
k = 1
n = 2
MSE = True #else Crossentropy
nonlinearity = F.relu
tied = True
final_bias = False
hidden_bias = False
unit_weights = False
learnable_scale_factor = False
initial_scale_factor = 1 # (1/(1-np.cos(2*np.pi/f)))**0.5
standard_magnitude = False
initial_embed = None
initial_bias = None


epochs = 5000
logging_loss = True

#Scheduler params
max_lr = 2
initial_lr = 2
warmup_frac = 0.05
final_lr = 2
decay_factor=(final_lr/max_lr)**(1/(epochs * (1-warmup_frac)))
warmup_steps = int(epochs * warmup_frac)


store_rate = epochs//100
plot_rate=0 #epochs/5


# Instantiate synthetic dataset
dataset = functions.SyntheticKHot(f,k)
batch_size = len(dataset) #Full batch gradient descent
loader = functions.DataLoader(dataset, batch_size=batch_size, shuffle = True, num_workers=0)

#Define the Loss function
criterion = nn.MSELoss() if MSE else nn.CrossEntropyLoss() 

# Instantiate the model
# initial_embed = torch.tensor(np.array([1/(1-np.cos(2*np.pi/f))**0.5*np.array([np.cos(2*np.pi*i/f),np.sin(2*np.pi*i/f)]) for i in range(f)]),dtype=torch.float32).T * 0.5
# initial_bias = -torch.ones(f)*(1/(1-np.cos(2*np.pi/f))- 1)*0.25
model = functions.Net(f, n,
            tied = tied,
            final_bias = final_bias,
            hidden_bias = hidden_bias,
            nonlinearity=nonlinearity,
            unit_weights=unit_weights,
            learnable_scale_factor=learnable_scale_factor,
            standard_magnitude=standard_magnitude,
            initial_scale_factor = initial_scale_factor,
            initial_embed = initial_embed,
            initial_bias = initial_bias)

# Define loss function and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

#Define a learning rate schedule
scheduler = functions.CustomScheduler(optimizer, warmup_steps, max_lr, decay_factor)


# Train the model
losses, weights_history, model_history = functions.train(model, loader, criterion, optimizer, epochs, logging_loss, plot_rate, store_rate, scheduler, lr_print_rate)
# %%
functions.plot_weights_interactive(weights_history, store_rate=store_rate)

# %%
