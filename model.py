import torch
import torch.nn as nn

class CityGuesserCNN(nn.Module):
    def __init__(self, num_cities):
        super().__init__()
        

        # Takes in 3 color channels, outputs 16 pattern channels using a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)# why 16 out chnanels
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image size in half takes only the most important parts
        

        self.flatten = nn.Flatten()# converts to 1D
        

        # The input number here requires a bit of math based on how much the pooling layers shrank your 224x224 image
        self.fc1 = nn.Linear(in_features=16 * 112 * 112, out_features=512) 
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) # Turns off 50% of connections to prevent memorization
        self.fc2 = nn.Linear(in_features=512, out_features=num_cities) # The final city guesses

    def forward(self, x):
        # --- USING THE TOOLS (Runs every time an image comes in) ---
        
        # Step 1: Pass the image 'x' through the convolutional layer
        x = self.conv1(x)
        
        # Step 2: Apply the activation gate
        x = self.relu1(x)
        
        # Step 3: Shrink the image down to save memory
        x = self.pool1(x)
        
        # Step 4: Squash the 3D block of patterns into a flat 1D list
        x = self.flatten(x)
        
        # Step 5: Feed the flat list into the first brain layer
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        
        # Step 6: Feed it to the final layer to output the city guesses
        x = self.fc2(x)
        
        # Step 7: Hand the final guesses back to the main training script
        return x
    
# --- Optional Testing Block ---
if __name__ == "__main__":
    print("Testing the CNN assembly line...")
    
    # 1. Create the model and tell it to expect 23 cities (based on your list from earlier)
    test_model = CityGuesserCNN(num_cities=23)
    
    # 2. Generate a fake batch of images (32 images, 3 color channels, 224 width, 224 height)
    # torch.randn() creates a tensor (a multi-dimensional grid of numbers) filled with random decimals
    dummy_batch = torch.randn(32, 3, 224, 224)
    
    # 3. Push the fake images through the model
    # Notice we just call test_model(dummy_batch) and it automatically triggers the 'forward' function!
    final_guesses = test_model(dummy_batch)
    
    # 4. Print the mathematical shape (the dimensions of the data block) to prove it worked
    print(f"Input batch shape: {dummy_batch.shape}")
    print(f"Output guesses shape: {final_guesses.shape}")
    print(f"Raw estimates for Image 1:\n{final_guesses[0]}")