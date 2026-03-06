import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image # PIL stands for Python Imaging Library, a standard tool used to open image files

# Import your model blueprint and constants
from model import CityGuesserTransfer
import constants as c

def predict_city(image_path):
    # 1. Setup the Hardware
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading AI on device: {device}...")

    # 2. Build the Empty Brain and Inject the Memories
    # We build the blueprint, then immediately load the weights [the internal mathematical dials the AI perfectly tuned overnight]
    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)
    model.load_state_dict(torch.load("best_city_guesser.pth", map_location=device))
    
    # Lock the dials! This prevents the AI from trying to learn from the test image.
    model.eval()

    # 3. Prepare the New Image
    # We MUST use the exact same transformations we used during training, including the exact ResNet normalization [a mathematical process that shifts and scales data to match a standardized baseline].
    image_prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Converts the picture into a tensor [a multi-dimensional grid of numbers]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open the image file from your computer and apply the math transformations
    img = Image.open(image_path).convert("RGB")
    img_tensor = image_prep(img)
    
    # PyTorch expects a batch [a small group of data processed together], so we add a fake "batch" dimension to our single image
    img_batch = img_tensor.unsqueeze(0).to(device)

    # 4. Make the Guess
    # torch.no_grad() turns off the calculus engine to save computer memory, since we aren't learning anymore
    with torch.no_grad():
        raw_scores = model(img_batch)
        
        # Pass the raw scores through a Softmax [a mathematical filter that squashes a list of raw scores into a clean list of percentages that perfectly add up to 100%]
        percentages = F.softmax(raw_scores, dim=1)[0] * 100
        
        # Find the single highest percentage and its index 
        winning_percentage, winning_index = torch.max(percentages, 0)

    # 5. Print the Results
    # You will need to replace this list with your actual 23 cities in alphabetical order!
    city_names = [
    "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", 
    "Chicago", "Lisbon", "London", "LosAngeles", "Madrid", 
    "Medellin", "Melbourne", "MexicoCity", "Miami", "Minneapolis", 
    "Osaka", "OSL", "Phoenix", "PRG", "PRS", 
    "Rome", "TRT", "WashingtonDC"       
    ]
    
    guessed_city = city_names[winning_index.item()]
    
    print("-" * 30)
    print(f"Prediction: {guessed_city}")
    print(f"Confidence: {winning_percentage.item():.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # Put the path to any random image you download from the internet here!
    test_image = "test_photo.jpg" 
    predict_city(test_image)
