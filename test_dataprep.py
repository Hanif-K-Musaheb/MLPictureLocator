import os
from dataPrep import split_data

# CHANGE THIS to one real city folder path that contains images
CITY_FOLDER = "archive/images/Boston"

def check_disjoint(a, b, c):
    sa, sb, sc = set(a), set(b), set(c)
    assert len(sa & sb) == 0, "Train/Val overlap!"
    assert len(sa & sc) == 0, "Train/Test overlap!"
    assert len(sb & sc) == 0, "Val/Test overlap!"

def main():
    assert os.path.isdir(CITY_FOLDER), f"Folder not found: {CITY_FOLDER}"

    train, val, test = split_data(CITY_FOLDER, seed=42)

    total = len(train) + len(val) + len(test)
    print("Folder:", CITY_FOLDER)
    print("Counts:", {"train": len(train), "val": len(val), "test": len(test), "total": total})

    # 1) sanity: non-empty (unless folder is tiny)
    assert total > 0, "No images found. Check extensions or folder path."

    # 2) no overlaps
    check_disjoint(train, val, test)

    # 3) split reproducibility
    train2, val2, test2 = split_data(CITY_FOLDER, seed=42)
    assert train == train2 and val == val2 and test == test2, "Split not reproducible with same seed!"

    # 4) different seed -> likely different split
    train3, val3, test3 = split_data(CITY_FOLDER, seed=99)
    if train == train3 and val == val3 and test == test3:
        print("Warning: different seed produced same split (possible if folder is very small).")
    else:
        print("Different seed produced a different split (good).")

    # 5) show a few examples
    print("\nSample train files:")
    for p in train[:5]:
        print(" ", p)

    print("\nAll checks passed ✅")

if __name__ == "__main__":
    main()