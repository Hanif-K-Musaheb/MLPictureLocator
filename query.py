import pandas as pd
import os

# 1. Create a list of all available cities
cities = [
    "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", 
    "Chicago", "Lisbon", "London", "LosAngeles", "Madrid", 
    "Medellin", "Melbourne", "MexicoCity", "Miami", "Minneapolis", 
    "OSL", "Osaka", "PRG", "PRS", "Phoenix", "Rome", "TRT", "WashingtonDC"
]

# 2. Display the cities as a numbered menu
print("Available Cities:")
for index, city in enumerate(cities):
    # We add 1 to the index so the menu starts at 1 instead of 0
    print(f"{index + 1}. {city}")

# 3. Ask the user which city they want
try:
    # Get the input, convert it to an integer, and subtract 1 to match the list's hidden index
    city_choice = int(input("\nEnter the number of the city you want to view: ")) - 1
    selected_city = cities[city_choice]
except (ValueError, IndexError):
    print("Invalid selection. Please run the script again and choose a valid number.")
    exit() 
# 4. Ask the user how many rows they want to see
try:
    row_count = int(input(f"How many rows of data for {selected_city} would you like to see? "))
except ValueError:
    print("Invalid number entered. Defaulting to 5 rows.")
    row_count = 5

# 5. Construct the file path and read the data
file_path = f"Dataframes/{selected_city}.csv"

# 6. Check if the file actually exists before trying to open it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"\n--- Displaying the first {row_count} rows for {selected_city} ---")
    print(df.head(row_count))
else:
    print(f"\nError: Could not find the file at {file_path}.")
    print("Make sure you downloaded the CSV for this city and placed it inside a folder named 'Dataframes'!")