"""
A Hopfield network is a type of recurrent neural network that can store and retrieve binary patterns as stable states.
"""
# Importing modules
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.01

# Loading MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Defining the Hopfield network model
class HopfieldNetwork(nn.Module):
    def __init__(self, n):
        super(HopfieldNetwork, self).__init__()
        # n is the number of neurons
        self.n = n
        # W is the weight matrix
        self.W = nn.Parameter(torch.zeros(n, n))
        # b is the bias vector
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        # x is a batch of binary patterns of shape (batch_size, n)
        # Applying the activation function to x * W + b
        return torch.sign(x @ self.W + self.b)

    def energy(self, x):
        # x is a batch of binary patterns of shape (batch_size, n)
        # Computing the energy function for each pattern in the batch
        return -0.5 * torch.sum(x * (x @ self.W + self.b), dim=1)

    def update(self):
        # Updating the weight matrix using Hebb's rule
        # W_ij = sum_k x_ki * x_kj for all i != j
        # W_ii = 0 for all i
        # b_i = sum_k x_ki for all i
        # where x_k is the k-th pattern in the training set
        X = torch.cat([data[0].view(-1, 28*28) for data in train_loader], dim=0).to(device)
        X[X < 0] = -1 # Binarizing the input patterns
        self.W.data = X.t() @ X / X.size(0) # Computing the outer product of X and dividing by the number of patterns
        self.W.data.fill_diagonal_(0) # Setting the diagonal elements to zero
        self.b.data = X.sum(dim=0) / X.size(0) # Computing the mean of each column of X

# Creating an instance of the model
model = HopfieldNetwork(28*28).to(device)

# Updating the model using Hebb's rule
model.update()

# Testing the model on corrupted input patterns
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        # Getting the inputs and labels
        inputs, _ = data
        # Flattening and binarizing the inputs
        inputs = inputs.view(-1, 28*28).to(device)
        inputs[inputs < 0] = -1 
        # Corrupting some pixels randomly by flipping their signs
        noise = torch.rand(inputs.size()) > 0.9 # Creating a mask of noisy pixels with 10% probability
        inputs[noise] *= -1 # Flipping the signs of noisy pixels
        # Forward pass to retrieve the original patterns
        outputs = model(inputs)
        # Computing the loss as the number of mismatched pixels per pattern
        loss = torch.mean(torch.sum(inputs != outputs, dim=1).float())
        # Updating the test loss 
        test_loss += loss.item()
    # Printing the average test loss 
    print(f'Test Loss: {test_loss/len(test_loader)}')
