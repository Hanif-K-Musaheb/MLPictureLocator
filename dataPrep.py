import os
import constants
import math

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

def split_data(folder_path):
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    test_split_data,train_split_data,validation_split_data = split_number(file_count,
                                                                          [constants.TEST_SPLIT,
                                                                           constants.TRAIN_SPLIT,
                                                                           constants.VALIDATION_SPLIT])
    #shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    

    




    

