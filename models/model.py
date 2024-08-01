import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = SimpleNN()
    test_input = torch.randn(1, 1, 28, 28)
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
    assert test_output.shape == (1, 10), "Output shape is incorrect!"
    print("Model forward pass test passed.")
