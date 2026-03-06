import torch
import torch.nn as nn
import torch.optim as optim

# Import the tools you built in your other files!
from dataPipe import get_data_loaders
from model import CityGuesserTransfer

import constants as c
import math
import time

def train_model():
    # 1. Setup the Hardware
    # This checks if your Mac has Apple Silicon (M1/M2/M3 chips) to speed up the math using MPS (Metal Performance Shaders, Apple's built-in system for accelerating graphics and AI math). If not, it uses the standard CPU (Central Processing Unit, the main general-purpose brain of your computer).
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training using device: {device}")

    # 2. Get the Data
    print("Loading data trucks...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    # 3. Instantiate (create a usable, physical copy of a programming blueprint) the Model
    print("Building the AI brain...")
    # We move the model directly to the 'device' so the fast hardware can process it
    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)

    # 4. Set the Rules for Learning
    # The Loss Function (a mathematical formula that calculates exactly how wrong the AI's guesses are compared to the actual correct answers)
    criterion = nn.CrossEntropyLoss()
    
    # The Optimizer (an algorithm that goes inside the AI's brain and slightly adjusts its internal mathematical dials to make it guess better next time)
    # lr is the learning rate (a tiny decimal number that tells the optimizer how big of a step to take when adjusting the AI's dials)
    optimizer = optim.Adam(model.parameters(), lr=c.LEARNING_RATE)

    # 5. The Main Training Loop
    epochs = c.EPOCHS #5# An epoch is one complete pass through your entire collection of training pictures
    
    best_val_accuracy = 0
    ### TRAINING PHASE
    print("Starting training!")
    for epoch in range(epochs):
        model.train() # Tells the model it is in learning mode (turns on Dropout)
        running_loss = 0.0 # A temporary counter to track our error score
        
        # Loop through every single batch (a small group of data processed together at the exact same time) of 32 images
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move the images and answers to the Mac's fast hardware
            images, labels = images.to(device), labels.to(device)

            # Step A: Clear the old math
            # zero_grad is a PyTorch command that clears out the old calculus math from the previous batch so it doesn't accidentally mix with the new batch
            optimizer.zero_grad()

            # Step B: Make a guess (Forward Pass)
            predictions = model(images)

            # Step C: Grade the guess
            loss = criterion(predictions, labels)

            # Step D: Calculate the required fixes (Backpropagation)
            # This complex calculus process tells PyTorch automatically exactly which dials need to turn and in which direction to fix the errors
            loss.backward()

            # Step E: Apply the fixes
            optimizer.step()

            #optional line to rest my CPU i have a macbook air so no fan :(
            time.sleep(c.COOLING_TIME)

            # Add up the error score so we can print it later
            running_loss += loss.item()

            # Print an update every 100 batches so you don't stare at a blank screen!
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f} | Accuracy: {(math.e**-(loss.item()))*100:.1f}%")

        # Print the average error score at the end of the entire Epoch
        avg_loss = running_loss / len(train_loader)

        ### VALIDATION PHASE    
        model.eval() # Turns OFF learning mode (locks the dials so it can't cheat on the test)
        val_loss = 0.0
        correct_guesses = 0
        total_images = 0

        # torch.no_grad() tells PyTorch to completely turn off the calculus engine to save memory
        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Make guesses on the pop quiz
                predictions = model(images)
                
                # Grade the pop quiz
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                # Calculate the actual percentage of correct answers
                # torch.max finds the highest score out of the 23 city guesses
                _, predicted_class = torch.max(predictions, 1) 
                total_images += labels.size(0)
                correct_guesses += (predicted_class == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_guesses / total_images) * 100

        print(f"--- End of Epoch {epoch+1} ---")
        print(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        ### SAVING MODEL if its the best found so far
        if val_accuracy > best_val_accuracy:
            print(f"higher accuracy found saving model... ({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%)")
            best_val_accuracy = val_accuracy
            
            # This line permanently saves the mathematical weights to a file in your project folder
            torch.save(model.state_dict(), "best_city_guesser.pth")




if __name__ == "__main__":
    train_model()