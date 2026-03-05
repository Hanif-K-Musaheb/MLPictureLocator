import os
import math
import random
import constants

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

def split_number(total, percentages):
    exact_splits = [total * p for p in percentages]
    integer_splits = [math.floor(x) for x in exact_splits]
    remainder = total - sum(integer_splits)

    fractional_parts = [(exact_splits[i] - integer_splits[i], i) for i in range(len(exact_splits))]
    fractional_parts.sort(reverse=True, key=lambda x: x[0])

    for i in range(remainder):
        index_to_add = fractional_parts[i][1]
        integer_splits[index_to_add] += 1

    return integer_splits

def list_images(folder_path):
    files = []
    for f in os.listdir(folder_path):
        full = os.path.join(folder_path, f)
        if os.path.isfile(full) and f.lower().endswith(IMAGE_EXTS):
            files.append(full)
    return files

def split_data(folder_path, seed=42):
    files = list_images(folder_path)
    rng = random.Random(seed)
    rng.shuffle(files)

    total = len(files)

    # IMPORTANT: keep ordering consistent everywhere
    # We'll do: train, val, test
    n_train, n_val, n_test = split_number(
        total,
        [constants.TRAIN_SPLIT, constants.VALIDATION_SPLIT, constants.TEST_SPLIT]
    )

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:n_train + n_val + n_test]

    return train_files, val_files, test_files

