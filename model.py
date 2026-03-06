import torch
import torch.nn as nn
from torchvision import models

class CityGuesserTransfer(nn.Module):
    def __init__(self, num_cities):
        super().__init__()
        
        # 1. Download the pre-trained "Master Carpenter"
        # 'DEFAULT' tells PyTorch to grab the smartest, most up-to-date mathematical weights available
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Freeze the image processor (the convolutional layers)
        # By setting requires_grad to False, we tell the optimizer to completely ignore these layers. 
        # This saves massive amounts of computer memory and stops us from breaking what the AI already knows!
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 3. Replace the final decision-making layer (The Brain)
        # ResNet-18's default final layer is built to output 1000 different categories (like dogs, cars, planes).
        # We find out exactly how many inputs that final layer receives (which is 512 for ResNet-18)...
        num_features = self.base_model.fc.in_features
        
        # ...and we replace it with a brand new, untrained layer that only outputs your 23 cities!
        self.base_model.fc = nn.Linear(in_features=num_features, out_features=num_cities)

    def forward(self, x):
        # Because ResNet-18 is already perfectly assembled by Microsoft, we don't have to write out the steps.
        # We just hand the image directly to the base_model and it handles the entire assembly line automatically.
        return self.base_model(x)

# --- Optional Testing Block ---
if __name__ == "__main__":
    print("Testing the new Transfer Learning model...")
    test_model = CityGuesserTransfer(num_cities=23)
    
    # Generate a fake batch of images (32 images, 3 color channels, 224 width, 224 height)
    dummy_batch = torch.randn(32, 3, 224, 224)
    final_guesses = test_model(dummy_batch)
    
    print(f"Input batch shape: {dummy_batch.shape}")
    print(f"Output guesses shape: {final_guesses.shape}")
    print(final_guesses[0])