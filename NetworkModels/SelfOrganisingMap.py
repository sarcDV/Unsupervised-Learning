"""
A self-organizing map (SOM) is a type of artificial neural network 
that can learn to map high-dimensional data onto a low-dimensional 
grid of neurons, preserving the topological structure of the data. 

SOMs can be used for clustering, dimensionality reduction, or visualization.

One way to implement SOMs in Python with PyTorch is to use the torch.nn module to 
define the network architecture and the torch.optim module to perform gradient-based optimization. 

Here is a code block that shows an example of a SOM class that can be trained on the MNIST dataset:
"""

import torch
import torch.nn as nn
import numpy as np

class SOM(nn.Module):
    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.m = m # number of rows in the grid
        self.n = n # number of columns in the grid
        self.dim = dim # dimensionality of the input data
        self.niter = niter # number of iterations for training
        self.alpha = alpha # learning rate
        self.sigma = sigma # neighborhood radius

        # initialize the weights randomly
        self.weights = nn.Parameter(torch.randn(m*n, dim))
        # initialize the locations of the neurons on the grid
        self.locations = nn.Parameter(torch.LongTensor(np.array(list(self.neuron_locations()))))
        # initialize the learning rate and neighborhood radius
        self.alpha = alpha if alpha else 0.3
        self.sigma = sigma if sigma else max(m, n) / 2.0

    def neuron_locations(self):
        # yield the coordinates of each neuron on the grid
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def get_distance(self, x1, x2):
        # compute the Euclidean distance between two tensors
        return torch.sqrt(torch.sum((x1 - x2) ** 2))

    def get_bmu(self, x):
        # find the best matching unit (BMU) for a given input vector x
        # BMU is the neuron with the smallest distance to x
        distances = torch.stack([self.get_distance(x, w) for w in self.weights])
        bmu_index = torch.argmin(distances)
        bmu_loc = self.locations[bmu_index]
        return bmu_index, bmu_loc

    def update(self, x, bmu_loc):
        # update the weights of the neurons based on their distance to the BMU
        num_neurons = self.m * self.n
        rate = 1.0 - np.arange(num_neurons).astype(float) / num_neurons # decay factor
        alpha_t = rate * self.alpha # decayed learning rate
        sigma_t = rate * self.sigma # decayed neighborhood radius

        # compute the distance between each neuron and the BMU on the grid
        distances = torch.stack([self.get_distance(bmu_loc.float(), l.float()) for l in self.locations])
        
        # compute the neighborhood function for each neuron based on its distance to the BMU
        h_t = torch.exp(-distances**2 / (2*sigma_t**2))

        # update the weights of each neuron by moving them closer to x weighted by h_t and alpha_t
        delta_w = h_t[:, None] * alpha_t[:, None] * (x - self.weights)
        new_weights = torch.add(self.weights, delta_w)
        
        # assign new values to the weights tensor
        self.weights.data.copy_(new_weights)

    def forward(self, x):
        # perform one step of training for a given input vector x
        bmu_index, bmu_loc = self.get_bmu(x)
        self.update(x, bmu_loc)

# create a SOM object with a 20x20 grid and 784 input dimensions (MNIST images)
som = SOM(20, 20, 784, 10000)

# create an optimizer object for the SOM parameters
optimizer = optim.Adam(som.parameters(), lr=0.01)

# train the SOM on the MNIST dataset for 10000 iterations
for i in range(10000):
    data = train_data[i % len(train_data)] # get a training sample
    data = data.view(-1) # flatten the image
    data = data / 255.0 # normalize the pixel
