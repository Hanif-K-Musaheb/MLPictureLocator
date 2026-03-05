import os
from PIL import Image

# 1. Set your folders
# The folder where your original city folders are currently located
input_folder = "archive/Images" 
# The new folder where the script will save the smaller versions
output_folder = "Resized_Images" 

# The target size required by most neural networks
target_size = (224, 224) 

# 2. Create the main output folder if it doesn't exist yet
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 3. Loop [a programming command that tells the computer to repeat the same action multiple times] through every city folder
for city_name in os.listdir(input_folder):
    city_path = os.path.join(input_folder, city_name)
    
    # Check if the current item is actually a directory [another word for a folder on your computer]
    if os.path.isdir(city_path):
        print(f"Processing city: {city_name}...")
        
        # Create a matching city folder inside our new Resized_Images folder
        out_city_path = os.path.join(output_folder, city_name)
        if not os.path.exists(out_city_path):
            os.makedirs(out_city_path)
            
        # 4. Loop through every single image inside that specific city folder
        for file_name in os.listdir(city_path):
            # Only process files that look like images
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(city_path, file_name)
                out_img_path = os.path.join(out_city_path, file_name)
                
                # 5. Open, resize, and save the image using a try-except block [a way to write code that anticipates and manages errors so the program doesn't crash]
                try:
                    with Image.open(img_path) as img:
                        # Convert to RGB just in case some images are grayscale or have transparent backgrounds
                        img = img.convert('RGB') 
                        img_resized = img.resize(target_size)
                        img_resized.save(out_img_path)
                except Exception as e:
                    print(f"Failed to process {file_name}: {e}")

print("\nAll images have been successfully resized to 224x224!")
