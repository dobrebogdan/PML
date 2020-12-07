import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.activation_function = nn.ReLU()


    def forward(self, x):
        current_output = self.input_layer(x)
        current_output = self.activation_function(current_output)
        current_output = self.hidden_layer(current_output)
        current_output = self.activation_function(current_output)
        current_output = self.output_layer(current_output)
        return current_output