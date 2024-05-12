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
    try: 
        # client = openai.OpenAI(api_key='sk-')
        
        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": question,
        #         }
        #     ],
        #     model="gpt-3.5-turbo-1106",
        # )
        # # Extract the answer from the GPT response
        # answer = chat_completion.choices[0].message.content
        # return answer

        openai.api_key = "sk-"
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106", 
            messages=[
                {
                    "role": "user", 
                    "content": question
                 }
            ]
        )
        answer = chat_completion.choices[0].message.content
        return answer
    
    except Exception as e:
        print("error with gpt")
        return None

def separate_answers(text):
    # Adjusting the pattern to be more flexible with spaces and line breaks
    pattern = r"Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)"

    # Using re.DOTALL flag to match across multiple lines
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return text, "unsure"

def preprocess_capstone(data, csv_file_path, filename):

    try:
        opinion_text = data["casebody"]["data"]["opinions"][0]["text"]
        if not opinion_text.strip():  # Checks if the text is only whitespace
            return
    except Exception as e:
        print(f"Skipping data point due to error: {e}")
        return  # Skip this data point

    question = f"""I have a legal case description and require two distinct pieces of information:
1. Summary: Please provide a detailed summary of the case, focusing on the facts and
events. Exclude any information about the verdict.
2. Verdict: State the verdict of the case, you can only choose the following categories:
- Plaintiff win
- Defendant win
- Settlement
- Case dismissal
- Unsure
If the verdict is mentioned, respond exclusively with the chosen categories ONLY. If the
outcome is not explicitly mentioned or cannot be inferred from the information given, please
respond with 'unsure' only.
Format your responses as follows:
# - For the summary, begin with 'Answer 1:'
# - For the verdict, start with 'Answer 2:'
Here is the description of the case:
{opinion_text}"""

    gpt_response = gpt_answer(question)
    # print(gpt_response)
    if gpt_response is not None:
        answer1, answer2 = separate_answers(gpt_response)

        if "unsure" not in answer2.lower() and "sorry" not in answer2.lower() and "unclear" not in answer2.lower():
            plaintiff, defendant = get_party(data["name"])
            case_title = data["name_abbreviation"]

            if "plaintiff" in answer2.lower():
                answer2 = "Plaintiff win"
            elif "defendant" in answer2.lower():
                answer2 = "Defendant win"
            elif "settle" in answer2.lower():
                answer2 = "Settlement"
            else:
                answer2 = "Case dismissal"

            row_data = [filename, case_title, plaintiff, defendant, answer1, answer2]

            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)




    
def preprocess_capstone_LJP(data, csv_file_path, filename):
    instruction_key = "### Instruction:"
    instruction_predict = """You are a legal expert who specializes in predicting outcomes for legal cases. Utilize your internal knowledge base to predict verdict. Your main function is to anticipate the likely verdict of the legal case presented by the user.
You should only output the verdict and not any other information.
Consider the following choices:
1. Defendant Wins
2. Plaintiff Wins
3. Settlement
4. Case Dismissal
"""
    input_key = "### Input:"
    response_key = "### Response:"
    end_key = "### End"
    question = f"""I have a legal case description and require two distinct pieces of information:
1. Summary: Please provide a detailed summary of the case, focusing on the facts and events. Exclude any information about the verdict.
2. Verdict: State the verdict of the case, consider the following categories:
- Plaintiff win
- Defendant win
- Settlement
- Case dismissal
- Unsure
If the verdict is mentioned, respond exclusively with the chosen categories ONLY. If the outcome is not explicitly mentioned or cannot be inferred from the information given, please respond with 'unsure' only.
Format your responses as follows:
# - For the summary, begin with 'Answer 1:'
# - For the verdict, start with 'Answer 2:'
Here is the description of the case:
{data["casebody"]["data"]["opinions"][0]["text"]}"""

    gpt_response = gpt_answer(question)
    # print(gpt_response)
    if gpt_response is not None:
        answer1, answer2 = separate_answers(gpt_response)

        if "unsure" not in answer2.lower() and "sorry" not in answer2.lower() and "unclear" not in answer2.lower():
            plaintiff, defendant = get_party(data["name"])
            case_title = data["name_abbreviation"]

            if "plaintiff" in answer2.lower():
                answer2 = "Plaintiff win"
            elif "defendant" in answer2.lower():
                answer2 = "Defendant win"
            elif "settle" in answer2.lower():
                answer2 = "Settlement"
            else:
                answer2 = "Case dismissal"

            formatted_prompt = (
                f"{instruction_key}\n{instruction_predict}\n\n"
                f"{input_key}\n"
                f"Case Title: {case_title}\n"
                f"Court: {data['court']['name']}\n"
                f"Plaintiffs: {plaintiff}\n"
                f"Defendants: {defendant}\n"
                f"Case_summarize: {answer1}\n"
                f"{response_key}\n"
                f"{answer2} \n\n"
                f"{end_key}\n"
            )
            
            row_data = [filename, formatted_prompt]
                        
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)

        else:
            print("GPT response is empty")

            # return formatted_prompt

def main():
    folder_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\test_folder'
    csv_file_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\test_processed2.csv'

    processed_titles = set()
    count = 0

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            # Skip the header if there is one
            next(reader, None)
            for row in reader:
                if row:  # Ensure the row is not empty
                    processed_titles.add(row[0])
    except FileNotFoundError:
        print("CSV file not found. A new file will be created.")

    print(processed_titles)

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data["name_abbreviation"] not in processed_titles:
                # preprocess_capstone(data, csv_file_path, filename)
                preprocess_capstone_LJP(data, csv_file_path, filename)
                print(data["name"])
                processed_titles.add(data["name_abbreviation"])

        if count > 10:
            break
        count += 1

if __name__ == "__main__":
    main()


# def main():
#     folder_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\train_folder'
#     csv_file_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\train_processed.csv'
#     new_csv_file_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\train_processed_new.csv'


#     with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
#         reader = csv.reader(csv_file)
#         rows = list(reader)

#     filename_dict = {}

#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         if os.path.isfile(file_path):
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             temp = data["name_abbreviation"]
#             filename_dict[temp] = filename

#     for row in rows:
#         if row[0] in filename_dict:  # row[0] is where title_abbr would be
#             row.append(filename_dict[row[0]])

#     with open(new_csv_file_path, mode='w', newline='', encoding='utf-8') as new_csv_file:
#         writer = csv.writer(new_csv_file)
#         writer.writerows(rows)

# if __name__ == "__main__":
#     main()