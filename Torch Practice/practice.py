import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Learning note: I want to create a synthetic dataset to test my neural network.
# Using numpy to generate some fake data for classification (2 classes).
np.random.seed(42)  # For reproducibility, learned this helps keep results consistent!
X = np.random.randn(1000, 2)  # 1000 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)  # Simple decision boundary: x1 + x2 > 0

# Converting numpy arrays to PyTorch tensors
# Note to self: PyTorch works with tensors, not numpy arrays directly.
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)  # LongTensor for classification labels

# Splitting data into train and test sets
# Found out I need to split data to evaluate the model later.
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Defining the neural network
# Learning note: nn.Module is the base class for all models in PyTorch.
# I need to define layers and the forward pass.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()  # Learned: Always call parent's init!
        # First layer: 2 input features -> 10 hidden units
        # Question to self: Why 10? Just experimenting, seems reasonable for a small dataset.
        self.layer1 = nn.Linear(2, 10)
        # Activation function: ReLU seems popular, prevents vanishing gradients (read this online).
        self.relu = nn.ReLU()
        # Second layer: 10 hidden units -> 2 output classes
        self.layer2 = nn.Linear(10, 2)  # 2 classes for binary classification

    def forward(self, x):
        # Learning note: Forward defines how data flows through the network.
        x = self.layer1(x)  # Linear transformation
        x = self.relu(x)    # Apply ReLU activation
        x = self.layer2(x)  # Output layer (no activation here, handled by loss function)
        return x

# Instantiate the model
model = SimpleNN()
print(model)  # Just checking what my model looks like!

# Loss function and optimizer
# Note: CrossEntropyLoss is good for classification, combines log softmax + NLL loss.
# Had to Google this, was confused why no softmax in the model!
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam seems like a good default, saw it in tutorials.
# Learning rate: 0.01 feels like a safe start, might tweak later.
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
# Learning note: This is where backpropagation happens!
num_epochs = 100  # Guessing 100 epochs, might overfit, will check test loss later.
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)  # Get predictions
    loss = criterion(outputs, y_train)  # Compute loss

    # Backward pass and optimization
    # Note: Zero gradients to avoid accumulating from previous iterations!
    # Forgot this initially and got weird results.
    optimizer.zero_grad()
    loss.backward()  # Compute gradients (backpropagation magic!)
    optimizer.step()  # Update weights

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
# Learning note: Set model to eval mode to disable dropout, batch norm, etc.
# Didn't use those here, but good practice!
model.eval()
with torch.no_grad():  # No need to compute gradients for evaluation
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)  # Get class with highest score
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy:.4f}')

# Reflection: Backpropagation was handled by loss.backward(), which computes gradients
# for all parameters. Optimizer.step() updates weights using those gradients.
# Took me a bit to understand that PyTorch tracks operations for autograd!