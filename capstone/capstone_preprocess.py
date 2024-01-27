import xml.etree.ElementTree as ET
import json
import pandas as pd
import os
import openai
import time
from pprint import pprint
import re
import csv
import requests
import sys
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def get_party(case_title):
    def split_title(title):
        for separator in ["versus", "v.", "vs."]:
            if separator in title:
                parts = title.split(separator)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()
        return None

    plaintiff = {'plaintiff_name': []}
    defendant = {'defendant_name': []}
    titles = case_title.split(';') if ';' in case_title else [case_title]
    
    for title in titles:
        parties = split_title(title)
        if parties:
            plaintiff['plaintiff_name'].append(parties[0])
            defendant['defendant_name'].append(parties[1])
    
    return plaintiff, defendant


def gpt_answer(question):
    max_retries = 3
    model="gpt-3.5-turbo-1106"
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(api_key='APIKEY')
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                # model="gpt-3.5-turbo-1106",
                # model="gpt-4-preview-1106",
                model=model
            )
            # Extract the answer from the GPT response
            answer = chat_completion.choices[0].message.content
            return answer
        except Exception as e:
            print(e)
            if attempt < max_retries - 1:
                model="gpt-4-1106-preview"
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                return None



def separate_answers(text):
    # Adjusting the pattern to be more flexible with spaces and line breaks
    pattern = r"Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)"

    # Using re.DOTALL flag to match across multiple lines
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return text, text
    
def preprocess_capstone(data):
    instruction_key = "### Instruction:"
    instruction_predict = "Below are the details of the legal case. Based on this information, can you anticipate the verdict? If the judgment favors the plaintiff or the defendant, what might the penalty be? In the event of a settlement, what could be the agreed-upon compensation amount?"
    input_key = "### Input:"
    response_key = "### Response:"
    end_key = "### End"

    try:
        case_text = data["casebody"]["data"]["opinions"][0]["text"]
    except IndexError:
        print(f"Error: 'opinions' list is out of range in the data")
        return None


    question = f"""
    I have a legal case description and require two distinct pieces of information:
    
    1. Summary: Please provide a detailed summary of the case, focusing on the facts and events. Exclude any information about the verdict.
    2. Verdict: State the outcome of the case. Specify whether the plaintiff or defendant won, the penalty imposed, or in the case of a settlement, the compensation agreed upon. If the verdict is not provided, respond "unsure" only.
    
    Format your responses as follows:
    - For the summary, begin with 'Answer 1:' 
    - For the verdict, start with 'Answer 2:'
    
    Please ensure that your total response does not exceed 3000 tokens.
    
    Here is the description of the case:
    {case_text}

    """

    gpt_response = gpt_answer(question)
    if gpt_response is None:
        print("GPT return None")
        return None
    # print(gpt_response)
    answer1, answer2 = separate_answers(gpt_response)
    if 'unsure' in answer2.lower() or 'verdict is not provided' in answer2.lower() or 'sorry' in answer2.lower():
        return None
    plaintiff, defendant = get_party(data["name"])


    formatted_prompt = (
        f"{instruction_key}\n{instruction_predict}\n\n"
        f"{input_key}\n"
        f"Court: {data['court']['name']}\n"
        f"Plaintiff_and_attorneys: {plaintiff}\n"
        f"Defendant_and_attorneys: {defendant}\n"
        f"Case_summary: {answer1}\n"
        f"{response_key}\n"
        f"{answer2} \n\n"
        f"{end_key}\n"
    )

    tokens = tokenizer.tokenize(formatted_prompt)

    # Check if the number of tokens exceeds the max_tokens
    if len(tokens) > 4096:
        print(f"exceed maximum token, has token size: {tokens}")
        return None
    else:
        return formatted_prompt

def main():
    folder_path = r'C:\Users\ds1657\Desktop\research\LawLLM\SD'
    csv_file_path = r'C:\Users\ds1657\Desktop\research\LawLLM\capstone_prompts_SD.csv'

    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Write a header row if needed
        writer.writerow(['File_name', 'Prompt'])

        # Iterate through each file in the folder
        for filename in os.listdir(folder_path):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)

            # Check if it's a file
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                # with open(file_path, 'r', encoding='latin-1') as f:
                    data = json.load(f)

                print(filename)
                prompt = preprocess_capstone(data)
                if prompt is not None:
                    writer.writerow([filename, prompt])
                    

                print(prompt)

if __name__ == "__main__":
    main()
