import torch.nn as nn


class CarPriceModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(CarPriceModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
   
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x