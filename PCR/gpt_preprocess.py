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
        print("error with gpt", e)
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

def preprocess_capstone(data):

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

            return case_title, answer1, answer2

    return None, None, None


