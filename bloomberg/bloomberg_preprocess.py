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
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def extract_line_list_text(element):
    text = ''
    for docket_header in element.findall('.//docket_header'):
        for line_list in docket_header.findall('line_list'):
            for line in line_list.findall('line'):
                if line.text:
                    text += line.text + ' '
    return text.strip()

def extract_text_from_info(tag_name, info_section):
    tag = info_section.find(tag_name)
    return tag.text if tag is not None else ''

def extract_party_information_corrected(party_type, party_groups_section):
    parties_info = []
    for party_group in party_groups_section.findall('.//party_group'):
        party_info = {}
        for parties in party_group.findall('parties'):
            for party in parties.findall('party'):
                if party.find('type').text == party_type:
                    name = party.find('name')
                    if name is not None:
                        party_info_key = 'plaintiff_name' if party_type == 'Plaintiff' else 'defendant_name'
                        party_info[party_info_key] = name.text
        
                    attorneys_info = []
                    for attorneys in party_group.findall('attorneys'):
                        for attorney in attorneys.findall('attorney'):
                            attorney_name = attorney.find('name')
                            if attorney_name is not None:
                                # Append the text of the attorney name
                                attorneys_info.append(attorney_name.text)

                    if attorneys_info:  # Only add if there are attorneys
                        party_info['attorneys'] = attorneys_info
        if party_info:
            parties_info.append(party_info)
    return parties_info



def gpt_answer(question):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(api_key='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')
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                # model="gpt-3.5-turbo-1106",
                model="gpt-4-1106-preview",
            )
            # Extract the answer from the GPT response
            answer = chat_completion.choices[0].message.content
            return answer
        except Exception as e:
            print(e)
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                return None

def extract_update_info(xml_file_path):

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    pleadings = root.find('.//pleadings')
    pleading_data = []

    for pleading in pleadings.findall('pleading'):
        date = pleading.find('date').text if pleading.find('date') is not None else None
        text = pleading.find('text').text if pleading.find('text') is not None else None

        # Check if either date or text is not None
        if date is not None or text is not None:
            pleading_data.append((date, text))

    # Create a DataFrame from the list
    # df = pd.DataFrame(pleading_data, columns=['Date', 'Text'])

    question = f"""
    I have a legal case description and require two distinct pieces of information:
    
    1. Summary: Please provide a concise summary of the case, focusing on the facts and events. Exclude any information about the verdict.
    2. Verdict: State the outcome of the case. Specify whether the plaintiff or defendant won, the penalty imposed, or in the case of a settlement, the compensation agreed upon. If the verdict is not provided, respond "unsure" only.
    
    Format your responses as follows:
    - For the summary, begin with 'Answer 1:' 
    - For the verdict, start with 'Answer 2:'
    
    Please ensure that your total response does not exceed 2500 tokens.
    
    Here is the description of the case:
    {pleading_data}

    """

    return gpt_answer(question)

def separate_answers(text):
    # Adjusting the pattern to be more flexible with spaces and line breaks
    pattern = r"Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)"

    # Using re.DOTALL flag to match across multiple lines
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return text, text


def extract_basic_info(xml_file_path):
    
    instruction_key = "### Instruction:"
    instruction_predict = "Below are the details of the legal case. Based on this information, can you anticipate the verdict? If the judgment favors the plaintiff or the defendant, what might the penalty be? In the event of a settlement, what could be the agreed-upon compensation amount?"
    input_key = "### Input:"
    response_key = "### Response:"
    end_key = "### End"
    # Load and parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # print(tree)

    # Extract the docket_header
    docket_header = extract_line_list_text(root)

    # Extracting the required information for each column from the 'info' section
    info_section = root.find('.//info')
    assigned_to = extract_text_from_info('assigned_to', info_section)
    case_in_other_court = extract_text_from_info('case_in_other_court', info_section)
    cause = extract_text_from_info('cause', info_section)
    date_filed = extract_text_from_info('date_filed', info_section)
    date_terminated = extract_text_from_info('date_terminated', info_section)
    demand = extract_text_from_info('demand', info_section)
    jurisdiction = extract_text_from_info('jurisdiction', info_section)
    jury_demand = extract_text_from_info('jury_demand', info_section)
    nature_of_suit = extract_text_from_info('nature_of_suit', info_section)
    status = extract_text_from_info('status', info_section)
    title = extract_text_from_info('title', info_section)

    # Extracting the information for Plaintiff and Defendant
    party_groups_section = root.find('.//party_groups')
    plaintiff_information_corrected = extract_party_information_corrected('Plaintiff', party_groups_section)
    defendant_information_corrected = extract_party_information_corrected('Defendant', party_groups_section)

    gpt_response = extract_update_info(xml_file_path)
    if gpt_response is None:
        print("GPT return None")
        return None
    # print(gpt_response)
    answer1, answer2 = separate_answers(gpt_response)
    if 'unsure' in answer2.lower() or 'verdict is not provided' in answer2.lower():
        return None


    # data = [docket_header] + [assigned_to, case_in_other_court, cause, date_filed, 
    #         date_terminated, demand, jurisdiction, jury_demand, nature_of_suit, 
    #         status, title] + [plaintiff_information_corrected, defendant_information_corrected]


    formatted_prompt = (
        f"{instruction_key}\n{instruction_predict}\n\n"
        f"{input_key}\n"
        f"Title: {title}\n"
        f"Cause: {cause}\n"
        f"Nature_of_suit: {nature_of_suit}\n"
        f"Judge: {assigned_to}\n"
        f"Demand: {demand}\n"
        f"Plaintiff_and_attorneys: {str(plaintiff_information_corrected)}\n"
        f"Defendant_and_attorneys: {str(defendant_information_corrected)}\n"
        f"Case_summarize: {answer1}\n\n"
        f"{response_key}\n"
        f"{answer2}\n\n"
        f"{end_key}\n"
    )

    tokens = tokenizer.tokenize(formatted_prompt)

    if len(tokens) > 4096:
        print(f"exceed maximum token, has token size: {tokens}")
        return None
    else:
        return formatted_prompt



def main():
    folder_path = 'C:\\Users\\ds1657\\Downloads\\Law_LLM\\Law_LLM\\tutorial\\BloombergAPI_test\\{}'
    court_ids = [511444, 402234, 69086, 69084, 69082, 408914, 100860, 100863, 511446, 68853, 511912, 68836, 100739]
    # court_ids = [511444]
    csv_file_path = r'C:\Users\ds1657\Downloads\Law_LLM\Law_LLM\tutorial\bloombreg_IP_prompts.csv'
    count = 0

    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Write a header row if needed
        writer.writerow(['File_name', 'Prompt'])

        for court_id in court_ids:
            folder = folder_path.format(court_id)

            # Iterate through each file in the folder
            for filename in os.listdir(folder):
                # if count == 10:
                #     break
                # count += 1
                # Construct full file path
                file_path = os.path.join(folder, filename)

                # Check if it's a file
                if os.path.isfile(file_path):

                    print(filename)
                    prompt = extract_basic_info(file_path)
                    if prompt is not None:
                        writer.writerow([filename, prompt])
                        

                    print(prompt)

if __name__ == "__main__":
    main()
