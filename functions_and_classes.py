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
# This class creates a synthetic dataset where each sample is a vector of zeros except 
# for a random index which is assigned a random value between 0 and 1.
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, f):
        """
        Initialize the SyntheticDataset object.
        
        Args:
            num_samples: The number of samples to generate.
            f: The length of the feature vector.
        """
        self.num_samples = num_samples  # The number of samples in the dataset
        self.f = f  # The size of the feature vector for each sample
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
            index = torch.randint(0, self.f, (1,))  # Pick a random index
            data[i, index] = torch.rand(1)  # Assign a random value between 0 and 1 at the chosen index
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# This class creates a synthetic dataset where each sample is a vector with k indices set to 1 
# (i.e., "hot") and the rest set to 0. All possible combinations of f choose k are included in the dataset.
class SyntheticKHot(Dataset):
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
def force_numpy(matrix):
    if isinstance(matrix,np.ndarray):
        return matrix
    elif isinstance(matrix, torch.Tensor):
        return matrix.cpu().detach().numpy()
    else:
        raise ValueError

def plot_weights_interactive(weights_history, store_rate=1, dotsize = 5, with_labels = True, to_label = None, plot_size = 800):
    """
    Function to plot weights history in an interactive way.

    Args:
    weights_history (dict): Dictionary of weight history.
    store_rate (int): Rate of storing weights.
    dotsize (int): Size of dots in scatter plot.
    with_labels (bool): If True, plot includes labels.
    to_label (list): List of labels to include. If None, all labels are included.
    plot_size (int): Size of plot.

    Returns:
    None
    """
    for key, weight_list in weights_history.items():
        # Initialize figure for each weight list
        fig = go.Figure()
        weight_list = [force_numpy(weight_matrix) for weight_matrix in weight_list]

        # Find maximum absolute value in all weight matrices
        max_value = np.max([np.abs(weight_matrix).max() for weight_matrix in weight_list])

        # Check if weights are scalars
        if weight_list[0].ndim == 0:
            plt.plot(weight_list)
            continue

        weight_shape = min(weight_list[0].shape)
        is_bias = True if len(weight_list[0].squeeze().shape) == 1 else False
        
        # Create a scatter plot for each weight matrix
        for i, weight_matrix in enumerate(weight_list):
            weight_matrix = weight_matrix.squeeze()
            if is_bias:
                new_matrix = np.zeros((weight_matrix.shape[0],2))
                new_matrix[:,0] = weight_matrix
                weight_matrix = new_matrix
            if weight_matrix.shape[1] > weight_matrix.shape[0]:
                weight_matrix = weight_matrix.T 

            x_values = weight_matrix[:, 0]
            y_values = weight_matrix[:, 1] if weight_shape > 1 else None
            z_values = weight_matrix[:, 2] if weight_shape == 3 else None
            labels = list(range(len(x_values))) if with_labels else ['' for _ in range(len(x_values))]
            if to_label is not None:
                labels = [label if label in to_label else '' for label in labels]

            if z_values is None:
                scatter = go.Scatter(x=x_values, y=y_values, mode='markers+text', text=labels,
                                     textposition='top center', marker=dict(size=dotsize), visible=False, name=f'Epoch {i * store_rate}')
            else:
                scatter = go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='markers+text', text=labels,
                                       marker=dict(size=dotsize), visible=False, name=f'Epoch {i * store_rate}')

            fig.add_trace(scatter)

        fig.data[0].visible = True
        if z_values is not None:
                fig.update_layout(scene = dict(
                    xaxis=dict(range=[-max_value * 1.1,max_value * 1.1], title='X Value'),
                    yaxis=dict(range=[-max_value * 1.1,max_value * 1.1], title='Y Value'),
                    zaxis=dict(range=[-max_value * 1.1,max_value * 1.1], title='Z Value'),
                    aspectmode='cube'))
        else:
            fig.update_xaxes(title_text='X Value', range=[-max_value * 1.1, max_value * 1.1])
            fig.update_yaxes(title_text='Y Value', range=[-max_value * 1.1, max_value * 1.1])

        steps = []
        for i in range(len(weight_list)):
            step = dict(
                method='restyle',
                args=['visible', [False] * len(fig.data)],
                label=f'Epoch {i * store_rate}'
            )
            step['args'][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        slider = dict(
            active=0,
            currentvalue={"prefix": f"{key} - "},
            pad={"t": 50},
            steps=steps
        )

        fig.update_layout(sliders=[slider], width=plot_size, height=plot_size)

        fig.show()



def get_activation_history(model_history, f, included_keys=None):
    """
    Function to get activation history from model history.

    Args:
    model_history (dict): Dictionary of model history.
    f (int): Dimensionality of identity matrix for input.
    included_keys (list): List of keys to include in activation history.

    Returns:
    activation_history (dict): Dictionary of activation history.
    """
    out, activations = list(model_history.values())[0](torch.eye(f), hooked = True)
    if included_keys is None:
        activation_history = {k: [] for k in activations}
    else:
        assert all([k in activations for k in included_keys]), f'Valid keys are {activations.keys()}'
        activation_history = {k: [] for k in included_keys}
    for model in model_history.values():
        out, activations = model(torch.eye(f), hooked = True)
        for k in activation_history:
            activation_history[k].append(activations[k])
    return activation_history

def calculate_angles(tensor):
    """
    Function to calculate angles from tensor of shape (n, 2).

    Args:
    tensor (torch.Tensor): Input tensor of shape (n, 2).

    Returns:
    angles (torch.Tensor): Tensor of angles.
    """
    assert tensor.shape[1] == 2, "Input tensor must be of shape (n, 2)"
    return torch.atan2(tensor[:, 1], tensor[:, 0])

def linearity(x):
    """
    Function to implement a linearity transformation.

    Args:
    x (int/float/tensor): Input to the transformation.

    Returns:
    x (int/float/tensor): Output after the transformation.
    """
    return x

#%%
# PyTorch Imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

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

#%%
import numpy as np

def group_vectors(vectors, epsilon):
    """
    Group similar vectors together, where similarity is defined as being within
    an epsilon difference in direction.
    
    Args:
        vectors (list): List of numpy arrays to be grouped.
        epsilon (float): The threshold to decide whether vectors are similar or not.
    
    Returns:
        tuple: A tuple containing the groups of similar vectors and their respective directions.
    """

    # Initialize an empty list to store the groups of similar vectors.
    groups = []

    # Initialize empty lists to store the normalized vectors and their respective directions.
    norms = []
    directions = []

    # Iterate through all vectors in the input list.
    for v in vectors:

        # Normalize the current vector.
        v_norm = v / np.linalg.norm(v)
        
        # If the norm of the vector is less than 0.01, skip it.
        if np.linalg.norm(v) < 0.01:
            continue

        # Initialize a flag to check if the current vector has been added to any group.
        added_to_group = False

        # Iterate through each existing group to check if this vector belongs there.
        for i, group in enumerate(groups):

            # Use the first vector in the group as a representative.
            group_representative = group[0]
            group_representative_norm = group_representative / np.linalg.norm(group_representative)

            # Calculate the dot product between the normalized vectors to find the cosine of the angle between them.
            dot_product = np.dot(v_norm, group_representative_norm)

            # Check if the dot product is close enough to 1, which would mean the vectors are scalar multiples of each other.
            if np.abs(dot_product - 1) < epsilon:
                # If it is, add the vector to the group and update the flag.
                group.append(v)
                norms[i].append(v_norm)
                added_to_group = True
                break

        # If the vector has not been added to any group, create a new group for it.
        if not added_to_group:
            groups.append([v])
            norms.append([v_norm])

    # Compute the direction of each group by taking the mean of the normalized vectors in the group.
    for norm in norms:
        arr = np.array(norm)
        directions.append(np.mean(arr, axis=0))

    # Return the groups of similar vectors and their respective directions.
    return groups, directions

#%%
def visualize_matrices_with_slider(matrices, rate, const_colorbar=False, plot_size = 800):
    """
    Visualizes a list of matrices as heatmaps and adds a slider to toggle between the different heatmaps.

    The matrices are expected to represent the state of some 2D data at different epochs or timesteps,
    and the slider is labeled with 'Epoch * rate' to suggest its use for visualizing data over time.

    Parameters:
    matrices (list of 2D np.array): List of 2D numpy arrays to be visualized as heatmaps.
    rate (int): Factor to multiply with each step (matrix index) to get the epoch for the slider label.
    const_colorbar (bool, optional): If True, the colorbar range will be consistent across all heatmaps 
                                     based on the global min and max values. Default is False.
    plot_size (int, optional): The size of the plot in pixels (height and width). Default is 800.

    Returns:
    None. The function will plot and display the interactive plot with the slider.

    """
    if const_colorbar:
        global_min = np.min([np.min(matrix) for matrix in matrices])
        global_max = np.max([np.max(matrix) for matrix in matrices])

    # Create empty figure
    fig = go.Figure()

    # Add traces for each matrix
    for i, matrix in enumerate(matrices):
        # Create a heatmap for the matrix
        heatmap = go.Heatmap(
            z=matrix, 
            colorscale='magma', 
            showscale=True,
            zmin=global_min if const_colorbar else None,
            zmax=global_max if const_colorbar else None
        )
        fig.update_yaxes(autorange='reversed')
        # Add the heatmap to the figure, but only make it visible if it's the first one
        fig.add_trace(heatmap)
        fig.data[i].visible = (i == 0)
        fig.data[i].name = f'Epoch {i * rate}'
        
    # Create a slider
    steps = []
    for i in range(len(matrices)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(matrices)],
            label=f'Epoch {i * rate}'
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Displaying: "},
        pad={"t": 50},
        steps=steps
    )]

    # Add the slider to the figure
    fig.update_layout(
        sliders=sliders,
        height = plot_size,
        width = plot_size
    )

    fig.show()

#%%
def visualise_polyhedron(vertices, filled_faces = True, opacity = 1, with_labels = False):
    """
    This function takes the vertices of a polyhedron and produces a 3D visualization of the shape.
    
    Parameters:
    vertices (numpy array or list): A 2D array where each row represents the coordinates of a vertex.
        Each row should contain exactly 3 values for the x, y, and z coordinates.
    
    filled_faces (bool, optional): If set to True, the faces of the polyhedron will be shaded. 
        Default is True.
    
    opacity (float, optional): The opacity of the shaded faces, as a float between 0 (completely transparent) 
        and 1 (completely opaque). Default is 1.
    
    with_labels (bool, optional): If set to True, each vertex of the polyhedron will be labeled with its 
        index in the input list. Default is False.
    
    Returns:
    None. The function directly generates a 3D plot using plotly's interactive plotting functions.
    """
        # convert vertices list to numpy array for convenience
    if vertices.shape[1] > vertices.shape[0]:
                vertices = vertices.T 
    
    # scipy's ConvexHull will give us the simplices (triangles) that form the polyhedron
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    
    # initialize 3D plot
    fig = go.Figure()
    
    # add each simplex as a triangular face
    # add each simplex as a triangular face
    for s in hull.simplices:
        # Ensure vertices are in counterclockwise order
        cross_product = np.cross(vertices[s[1]] - vertices[s[0]], vertices[s[2]] - vertices[s[0]])
        dot_product = np.dot(cross_product, hull.equations[s[0], :-1])
        
        if dot_product < 0:
            s = s[[0, 2, 1]]  # Swap the last two elements to change the order to counterclockwise
        
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate for plotly
        if filled_faces:
            x = vertices[s, 0]
            y = vertices[s, 1]
            z = vertices[s, 2]
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=opacity))
        fig.add_trace(go.Scatter3d(x=vertices[s, 0], y=vertices[s, 1], z=vertices[s, 2],
                                mode='lines',
                                line=dict(color='blue', width=2)))

    # add each vertex in the hull as a label
    if with_labels:
        for i in hull.vertices:
            fig.add_trace(go.Scatter3d(x=[vertices[i, 0]], y=[vertices[i, 1]], z=[vertices[i, 2]],
                                    mode='text',
                                    text=[str(i)],  # or other string labels
                                    textposition='top center'))

        
    # set the 3d scene parameters
    fig.update_layout(showlegend = False,
                      scene = dict(xaxis_title='X',
                                   yaxis_title='Y',
                                   zaxis_title='Z',
                                   aspectmode='auto'),
                      width=700,
                      margin=dict(r=20, l=10, b=10, t=10))
    fig.show()
