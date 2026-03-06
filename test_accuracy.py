import torch
from dataPipe import get_data_loaders
from model import CityGuesserTransfer
import constants as c

def run_final_exam():
    # 1. Setup the Hardware
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading grading system on: {device}")

    # 2. Get the Test Data
    # We use the underscore '_' to ignore the train and val loaders, we only want the test_loader!
    _, _, test_loader = get_data_loaders(batch_size=32)

    # 3. Load the Saved AI Brain
    print("Loading your trained AI...")
    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)
    model.load_state_dict(torch.load("best_city_guesser.pth", map_location=device))
    
    # model.eval() turns OFF learning mode so it just takes the test without cheating
    model.eval()

    # 4. Set up our Scoreboards
    city_names = [
        "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", 
        "Chicago", "Lisbon", "London", "LosAngeles", "Madrid", 
        "Medellin", "Melbourne", "MexicoCity", "Miami", "Minneapolis", 
        "Osaka", "OSL", "Phoenix", "PRG", "PRS", 
        "Rome", "TRT", "WashingtonDC"
    ]
    
    overall_correct = 0
    overall_total = 0
    
    # These dictionaries [a built-in coding tool that stores data in matching pairs] will track the score for each individual city
    class_correct = {city: 0 for city in city_names}
    class_total = {city: 0 for city in city_names}

    print("Starting the Final Exam... (This might take a minute)")
    
    # torch.no_grad() turns off the calculus engine to save memory
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Make guesses for the batch
            predictions = model(images)
            _, predicted_classes = torch.max(predictions, 1)
            
            # Tally the overall score
            overall_total += labels.size(0)
            overall_correct += (predicted_classes == labels).sum().item()
            
            # Tally the specific city scores
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                guessed_label = predicted_classes[i].item()
                
                # Look up the actual city name using our list
                city_name = city_names[true_label]
                
                # Add 1 to the total number of times we saw this city
                class_total[city_name] += 1
                
                # If the AI guessed it right, add 1 to its correct score!
                if true_label == guessed_label:
                    class_correct[city_name] += 1

    # 5. Calculate and Print the Final Report Card
    print("\n" + "="*40)
    print("🎓 FINAL EXAM REPORT CARD 🎓")
    print("="*40)
    
    overall_accuracy = (overall_correct / overall_total) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total} images)\n")
    
    print("--- Accuracy Breakdown by City ---")
    for city in city_names:
        if class_total[city] > 0:
            city_acc = (class_correct[city] / class_total[city]) * 100
            print(f"{city.ljust(15)}: {city_acc:>6.2f}%  ({class_correct[city]}/{class_total[city]})")
        else:
            print(f"{city.ljust(15)}: No test images found!")
    print("="*40)

if __name__ == "__main__":
    run_final_exam()