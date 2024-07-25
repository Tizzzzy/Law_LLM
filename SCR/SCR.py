import sys
import os

# Assuming your PCR_temp folder is in "C:\Users\super\OneDrive\Desktop\research\LawLLM"
directory_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\SCR'
if directory_path not in sys.path:
    sys.path.append(directory_path)


from chroma import Chroma
from embeddings import OpenAIEmbeddings
import os
import chromadb
import json
from tqdm import tqdm
import csv
os.environ['OPENAI_API_KEY'] = 'sk-'
import pickle
embd = OpenAIEmbeddings()
client = chromadb.PersistentClient(path="chroma1000")
db = Chroma(client=client, embedding_function=embd, persist_directory='chroma1000')


import re

def clean_text(text):
    if not isinstance(text, str):
        return text
    # Remove all non-printable characters and excessively problematic symbols
    text = re.sub(r'[^\x20-\x7E]', ' ', text)  # ASCII printable characters
    text = re.sub(r'[\r\n]+', ' ', text)  # Replace newlines and carriage returns with space
    text = re.sub(r'[;"]', ' ', text)  # Remove semicolons and quotes that can break CSV format
    return text.strip()

def format_data(training_cases):
    instruction_key = "### Instruction:"
    instruction_predict = """You are a legal expert who specializes in comparing user-supplied legal cases to a list of candidate legal cases, which includes titles and content. Your main function is to identify and output the title of the most similar case from the list based on the description provided.
You should only output the case title and not any other information.
Consider the following choices:"""
    input_key = "### Input:"
    response_key = "### Response:"
    end_key = "### End"

    choice_list = "\n".join([f"Choice {i}:\n{case}" for i, case in enumerate(training_cases[1:])])
    

    formatted_prompt = (
        f"{instruction_key}\n{instruction_predict}{choice_list}\n"
        f"{input_key}\n"
        f"Case_deatil: {training_cases[0]}\n"
        f"{response_key}\n"
        f"{training_cases[1]}\n"
        f"{end_key}\n"
    )

    return formatted_prompt


import pandas as pd

data_to_write = []

csv_file_path = r"C:\Users\super\OneDrive\Desktop\research\LawLLM\test_processed2.csv"
json_train_path = r"C:\Users\super\OneDrive\Desktop\research\LawLLM\similar_train.json"

df = pd.read_csv(csv_file_path, header=None)

for index, row in df.iterrows():
    data = row[1]  # Use the actual column name where the data resides

    sections = data.split("###")
    if len(sections) > 2:  # Check if the required section exists
        case_detail = str(sections[2].replace("Input:", '').strip())  # Assuming the case details are in the third section
        training_cases = [str(case_detail)]
        # print(case_detail.strip())

        candidates = db.similarity_search(case_detail, k=10)
        for can in candidates:
            # temp = str(can.page_content).replace('\n', ' ').replace('\r', ' ')
            temp = clean_text(str(can.page_content))
            training_cases.append(str(temp))
            # print(temp)

        train_format = format_data(training_cases)
        print(train_format)

        data_to_write.append(train_format)



with open(json_train_path, 'w', encoding='utf-8') as file:
    json.dump(data_to_write, file, ensure_ascii=False, indent=4)