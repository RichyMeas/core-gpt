import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Example for MNIST input size (28x28)
        self.fc2 = nn.Linear(128, 10)   # 10 output classes for classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
model = SimpleNN()

# Create a random input tensor with size [batch_size, input_size]
# For example, a batch of 32 images with size 28x28 (flattened to 784)
input_data = torch.randn(32, 784)

# Compile the model using torch.compile (available from PyTorch 2.0)
compiled_model = torch.compile(model)

# Define a simple loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass using the compiled model
output = compiled_model(input_data)

# Example of how to use loss and backpropagate
labels = torch.randint(0, 10, (32,))  # Random labels for the batch
loss = criterion(output, labels)
loss.backward()

# Optimizer step
optimizer.step()

print(f"Output shape: {output.shape}")
print(f"Loss: {loss.item()}")
