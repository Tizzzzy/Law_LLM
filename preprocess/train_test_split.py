import os
import shutil
from glob import glob
from random import shuffle
import json

# Define paths
# Define the path to the source folder containing the JSON files
source_folder = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\SD'

# Define paths to the train and test folders (these will be created if they don't exist)
train_folder = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\train_folder'
test_folder = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\test_folder'

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# List all files in the source folder that end with .json
json_files = [file for file in glob(os.path.join(source_folder, '*.json'))]

# Shuffle the list for randomness
shuffle(json_files)

# Calculate split point for 80% training data
split_point = int(len(json_files) * 0.8)

# Split files into training and testing
train_files = json_files[:split_point]
test_files = json_files[split_point:]

# Function to demonstrate parsing a JSON file with error handling
def parse_json_example(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
    except Exception as e:
        print(f"Unexpected error with file {file_path}: {e}")

# This function is an example; integrate error handling as needed in your actual file operations

# Move or copy files, ensuring they have a .json extension
def move_files(file_paths, destination_folder):
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.endswith('.json'):
            dest_path = os.path.join(destination_folder, filename)
            shutil.move(file_path, dest_path)  # or shutil.copy for copying
        else:
            print(f"Skipping non-JSON file: {filename}")

# Move training and testing files
move_files(train_files, train_folder)
move_files(test_files, test_folder)

print(f'Moved {len(train_files)} files to {train_folder}')
print(f'Moved {len(test_files)} files to {test_folder}')
