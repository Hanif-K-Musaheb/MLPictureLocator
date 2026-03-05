from dataPrep import split_all_cities_to_folders

if __name__ == "__main__":
    summary = split_all_cities_to_folders(
        raw_root="archive/images",
        output_root="data",
        seed=42
    )

    print("Done! Split summary:")
    for city, counts in summary.items():
        print(city, counts)