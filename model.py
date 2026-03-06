import torch.nn as nn # nn stands for Neural Networks

class CityGuesserCNN(nn.Module): # nn.Module tells PyTorch this is an AI model
    def __init__(self, num_cities):
        super().__init__()
        # 1. Define your layers here (the parts list)
        pass 

    def forward(self, x):
        # 2. Tell the image tensor 'x' exactly how to flow through the layers here
        return x