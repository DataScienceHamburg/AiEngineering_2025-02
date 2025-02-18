#%% packages
from model_class import CarPriceModel
import torch

# %% load state_dict
state_dict = torch.load('model_001.pth')

#%% get number of input features
input_size = state_dict['linear.weight'].shape[1]
hidden_size = state_dict['linear.weight'].shape[0]
output_size = state_dict['linear.bias'].shape[0]

#%% create model instance
model = CarPriceModel(input_size=input_size, 
                      output_size=output_size, 
                      hidden_size=hidden_size)

#%% assign model weights
model.load_state_dict(state_dict)

# %%
model.state_dict()
# %%

