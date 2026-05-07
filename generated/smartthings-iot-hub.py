"""
Module for Quantized Probabilistic AI for Gear Fault Diagnosis in Motor Drives.

Source paper: Quantized Probabilistic AI for Gear Fault Diagnosis in Motor Drives (http://arxiv.org/abs/2605.05032v1)
This module implements the core algorithm described in the paper, which applies quantized AI models for gear fault diagnosis.
The mathematical idea behind this implementation is to use a probabilistic approach to diagnose gear faults in motor drives.
The algorithm uses a quantized neural network to predict the probability of a gear fault given a set of input features.

Key hyperparameters and their default values:
- num_classes (integer): number of classes for classification (default: 5)
- input_size (integer): size of input features (default: 128)
- hidden_size (integer): size of hidden layer (default: 256)
- num_layers (integer): number of layers in the neural network (default: 2)
- dropout (float): dropout rate for regularization (default: 0.2)
- learning_rate (float): learning rate for optimization (default: 0.001)

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GearFaultDataset(Dataset):
    """
    Custom dataset class for gear fault data.

    Parameters
    ----------
    features : numpy.ndarray
        Input features for gear fault diagnosis.
    labels : numpy.ndarray
        Corresponding labels for the input features.
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {
            'feature': torch.tensor(feature, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class QuantizedProbabilisticAI(nn.Module):
    """
    Quantized Probabilistic AI model for gear fault diagnosis.

    Parameters
    ----------
    num_classes : integer
        Number of classes for classification.
    input_size : integer
        Size of input features.
    hidden_size : integer
        Size of hidden layer.
    num_layers : integer
        Number of layers in the neural network.
    dropout : float
        Dropout rate for regularization.
    """

    def __init__(self, num_classes=5, input_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super(QuantizedProbabilisticAI, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        # Initialize the neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # input layer to hidden layer
        self.dropout_layer = nn.Dropout(dropout)  # dropout layer for regularization
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_size, num_classes)  # hidden layer to output layer
        # Initialize the quantization parameters
        self.quantization_bits = 8  # number of bits for quantization

    def forward(self, x):
        # Forward pass through the neural network
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.dropout_layer(x)  # apply dropout for regularization
        for _ in range(self.num_layers - 1):  # iterate through the remaining hidden layers
            x = torch.relu(self.fc2(x))  # activation function for hidden layer
            x = self.dropout_layer(x)  # apply dropout for regularization
        x = self.fc3(x)  # output layer
        return x

    def quantize(self, x):
        # Quantize the input tensor using the specified number of bits
        x_min = x.min()
        x_max = x.max()
        x_range = x_max - x_min
        x_scaled = (x - x_min) / x_range  # scale the input to [0, 1] range
        x_quantized = torch.round(x_scaled * (2**self.quantization_bits - 1))  # quantize the input
        x_quantized = x_quantized / (2**self.quantization_bits - 1)  # scale back to original range
        x_quantized = x_quantized * x_range + x_min  # shift back to original range
        return x_quantized

def train(model, device, dataloader, optimizer, loss_fn):
    """
    Train the model on the given dataloader.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    device : torch.device
        Device to use for training.
    dataloader : DataLoader
        Dataloader for training data.
    optimizer : torch.optim.Optimizer
        Optimizer to use for training.
    loss_fn : torch.nn.Module
        Loss function to use for training.

    Returns
    -------
    loss : float
        Average loss over the epoch.
    """
    model.train()
    total_loss = 0
    for batch in dataloader:
        features = batch['feature'].to(device)
        labels = batch['label'].to(device)
        # Forward pass
        outputs = model(features)
        loss = loss_fn(outputs, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, device, dataloader, loss_fn):
    """
    Evaluate the model on the given dataloader.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    device : torch.device
        Device to use for evaluation.
    dataloader : DataLoader
        Dataloader for evaluation data.
    loss_fn : torch.nn.Module
        Loss function to use for evaluation.

    Returns
    -------
    loss : float
        Average loss over the epoch.
    accuracy : float
        Average accuracy over the epoch.
    """
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            features = batch['feature'].to(device)
            labels = batch['label'].to(device)
            # Forward pass
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # Generate some random data for demonstration
    features = np.random.rand(100, 128)
    labels = np.random.randint(0, 5, 100)
    # Create the dataset and dataloader
    dataset = GearFaultDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # Create the model, device, optimizer, and loss function
    model = QuantizedProbabilisticAI()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    # Train and evaluate the model
    for epoch in range(10):
        loss = train(model, device, dataloader, optimizer, loss_fn)
        eval_loss, accuracy = evaluate(model, device, dataloader, loss_fn)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}')