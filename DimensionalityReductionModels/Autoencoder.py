"""
An autoencoder is a type of neural network that learns to encode input data into a 
smaller feature vector and then reconstruct it using a decoder network.
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

# Defining the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Creating an instance of the model
model = Autoencoder().to(device)

# Defining the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    train_loss = 0.0
    for data in train_loader:
        # Getting the inputs and labels
        inputs, _ = data
        # Flattening the inputs
        inputs = inputs.view(-1, 28*28).to(device)
        # Zeroing the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Computing the loss
        loss = criterion(outputs, inputs)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Updating the train loss
        train_loss += loss.item()
    # Printing the average train loss for each epoch
    print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}')

# Testing the model
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        # Getting the inputs and labels
        inputs, _ = data
        # Flattening the inputs
        inputs = inputs.view(-1, 28*28).to(device)
        # Forward pass
        outputs = model(inputs)
        # Computing the loss
        loss = criterion(outputs, inputs)
        # Updating the test loss
        test_loss += loss.item()
    # Printing the average test loss 
    print(f'Test Loss: {test_loss/len(test_loader)}')
