import json
import pandas as pd


# Path to the JSON file
LJP_file = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\test_processed2.csv'
SCR_file = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\similar_train.json'
PCR_file = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\precedent_train2.json'

train_file = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\train_file.json'


# Read the JSON file
with open(SCR_file, 'r', encoding='utf-8') as file:
    SCR_data = json.load(file)

with open(PCR_file, 'r', encoding='utf-8') as file:
    PCR_data = json.load(file)

df = pd.read_csv(LJP_file, header=None, usecols=[1])

csv_data = df[1].tolist()

combined_data = SCR_data + PCR_data + csv_data


# Write to a new JSON file
with open(train_file, 'w', encoding='utf-8') as file:
    json.dump(combined_data, file, indent=4, ensure_ascii=False)