import sys
import os

directory_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\SCR'
if directory_path not in sys.path:
    sys.path.append(directory_path)

import json
import pandas as pd
import pickle
import re
import csv
from tqdm import tqdm
import requests
import gpt_preprocess
from chroma import Chroma
from embeddings import OpenAIEmbeddings
import chromadb
import json
from tqdm import tqdm
os.environ['OPENAI_API_KEY'] = 'sk-'
import pickle

embd = OpenAIEmbeddings()
client = chromadb.PersistentClient(path="chroma1000")
db = Chroma(client=client, embedding_function=embd, persist_directory='chroma1000')

def get_cite_dict(folder_path):
    cite_to = {}
    for filename in os.listdir(folder_path):
        if filename == '.DS_Store':
            continue
        # Construct full file path
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if it's a file
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cite_to[data["id"]] = []
                    if data["cites_to"]:
                        for cite in data["cites_to"]:
                            if "case_ids" in cite.keys():
                                cite_to[data["id"]].extend(cite["case_ids"])    
        except Exception as e:
            print(f"An error occurred in case {filename}: {e}")
            break
    return cite_to

# def get_case_text(id):
#     response = requests.get(
#         f'https://api.case.law/v1/cases/{id}/?full_case=true',
#         headers={'Authorization': 'Token 788802dd98bf73d69d60aaca29b6bbbe32730401'}
#     )

#     # 检查响应状态码
#     if response.status_code == 200:
#         # 解析 JSON 数据
#         data = response.json()

#         return data
#     else:
#         print(f'Failed to get case {id}')
#         return None

def get_case_file(id):
    folder_path = "C:\\Users\\super\\OneDrive\\Desktop\\research\\LawLLM\\PCR_temp\\precedent_folder"
    file_path = f"{folder_path}\\{id}.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding="UTF-8") as f:
        case_info = json.load(f)
    return case_info

def precedent_format(case_title, case_context, case_verdict):
    precedent_datat = f"""Case Title: {case_title}
Case Context: {case_context}
Case Verdict: {case_verdict}"""
    return precedent_datat

def format_data(case_json):
    instruction_key = "### Instruction:"
    instruction_predict = """You are a legal expert who specializes in comparing user-supplied legal cases to a list of candidate legal cases, which includes titles and content. Your main function is to identify and output the precedent case from the list based on the description provided.
You should only output the reasoning process and case title.
Consider the following choices:
"""
    input_key = "### Input:"
    response_key = "### Response:"
    end_key = "### End"

    choice_list = "\n".join([f"Choice {i}:\n{case}" for i, case in enumerate(case_json[1:])])
    

    formatted_prompt = (
        f"{instruction_key}\n{instruction_predict}{choice_list}\n"
        f"{input_key}\n"
        f"Case_deatil: {case_json[0]}\n"
        f"{response_key}\n"
        f"{case_json[1]}\n"
        f"{end_key}\n"
    )

    return formatted_prompt

    # row_data = [
    #     instruction_key,
    #     instruction_predict,
    #     choice_list,
    #     input_key,
    #     f"Case_detail: {case_json[0]}",
    #     response_key,
    #     case_json[1],
    #     end_key
    # ]
            
                        
    # with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(row_data)



def main():

    results = []

    folder_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\train_folder'
    json_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\PCR_temp\precedent_train2.json'

    cite_dict = get_cite_dict(folder_path)
    with open(r'C:\Users\super\OneDrive\Desktop\research\LawLLM\PCR_temp\cite_train.json', 'w', encoding='utf-8') as file:
        json.dump(cite_dict, file, ensure_ascii=False, indent=4)

    count = 0

    for file_name, cites in tqdm(cite_dict.items()):

        if len(cites) == 0:
            continue

        case_json = []

        # 处理current case json
        with open(f"{folder_path}/{file_name}.json", 'r', encoding='utf-8') as file:
            # 加载JSON文件内容到data变量中
            case = json.load(file)

        if case:
            title, context, verdict = gpt_preprocess.preprocess_capstone(case)
            if title and context and verdict:
                current_case = precedent_format(title, context, verdict)
                case_json.append(current_case)

        # 查看current case是不是valid的，如果是的话 那么长度会等于1
        # 或没有precedent的话 就跳过
        if len(case_json) == 0:
            continue


        # 处理precedent data
        for cite in cites:
            if len(case_json)>5:
                break
            json_data = get_case_file(cite)
            if json_data:
                case_title, case_context, case_verdict = gpt_preprocess.preprocess_capstone(json_data)
                if case_title and case_context and case_verdict:
                    precedent_data = precedent_format(case_title, case_context, case_verdict)
                    case_json.append(precedent_data)
            else:
                print(f'Failed to get case {cite}')

        while len(case_json) < 6:
            case_json.append(case_json[-1])

        candidates = db.similarity_search(current_case, k = 5)
        print(len(candidates))
        for candidate in candidates:
            case_json.append(str(candidate.page_content).replace('\n', ' '))

        formated_data = format_data(case_json)

        results.append(formated_data)

        # case_json.extend([instruction_key, instruction_predict, input_key, response_key, end_key])

        # with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(case_json)


        if count > 1:
            break
        count += 1

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)





        
# def main_temp():

#     csv_file_path = r"C:\Users\super\OneDrive\Desktop\research\LawLLM\test_processed_new.csv"
#     precedent_json = r"C:\Users\super\OneDrive\Desktop\research\LawLLM\PCR_temp\cite_test.json"
#     csv_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\PCR_temp\precedent_test.csv'

#     df = pd.read_csv(csv_file_path, header=None)

#     with open(precedent_json, 'r', encoding='utf-8') as file:
#         precedent_jsons = json.load(file)

#     with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)

#         for index, row in df.iterrows():
#             print(index)
#             case_json = []

#             # Adjusted for tuple nature of iterrows() output
#             case_title, case_context, case_verdict, case_id = row[0], row[3], row[4], row[5]

#             current_case = precedent_format(case_title, case_context, case_verdict)
#             case_json.append(current_case)

#             try:
#                 precedent_cases = precedent_jsons[case_id.replace(".json", '')]
#                 for precedent_case in precedent_cases:
#                     if len(case_json) > 5:
#                         break
#                     json_data = get_case_file(precedent_case)
#                     if json_data:
#                         title, context, verdict = gpt_preprocess.preprocess_capstone(json_data)
#                         if title and context and verdict:
#                             precedent_data = precedent_format(title, context, verdict)
#                             case_json.append(precedent_data)
#                     else:
#                         print(f'Failed to get case {precedent_case}')
#             except Exception as e:
#                 print(f"Key {case_id.replace('.json', '')} not found in precedent JSONs")
            
#             while len(case_json) < 6:
#                 case_json.append(case_json[-1])
        

#             writer.writerow(case_json)

#             # if index == 3:
#             #     break

# def main_temp2():

#     csv_path = r'C:\Users\super\OneDrive\Desktop\research\LawLLM\PCR_temp\precedent_train.csv'


#     df_precedent = pd.read_csv(csv_path, header=None)

#     current_cases = []

#     for index, row in df_precedent.iterrows():
#         print(index)

#         # Adjusted for tuple nature of iterrows() output
#         current_case = row[0]

#         # tmp = list(map(lambda x : x.page_content, db.similarity_search(text, k=110)))[1:]

#     if len(current_cases) == len(df_precedent):
#         print(current_cases)
#         df_precedent[0] = current_cases

#         df_precedent.to_csv(r'C:\Users\super\OneDrive\Desktop\research\LawLLM\PCR_temp\precedent_train_new.csv', index=False, header=False)
        
#     else:
#         print("Error:")

if __name__ == "__main__":
    # main_temp2()
    main()