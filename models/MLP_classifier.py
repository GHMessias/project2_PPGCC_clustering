import torch.nn as nn
import torch

class MLP_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_classifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x