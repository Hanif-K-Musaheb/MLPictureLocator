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
    print("torch version:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))

    # this will check for my device(hanif) if it has MPS which i found online will speed up training time for me.
    #TLDR dont need to change this since i pushed the NN to the repo bu tif you changing on non apple device change to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    print(f"Training using device: {device}")

    #this will get the data that we all split in make_splits.py and put in ot batchs of 32 images which is industry standard
    print("Loading data batchs...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    #this will instantiate a model of the brain so that it can then moved to the device outlined above which make it faster to model
    print("Building the AI brain...")
    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)

    #applies softmax to the output of the neural net so that all the answers are between 0 and 1 
    #it will also calculate the loss: loss = -log(probability its correct) essentially measuring how confident it was
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
            torch.save(model.state_dict(), "best_city_guesser_15_epochs.pth")




if __name__ == "__main__":
    train_model()