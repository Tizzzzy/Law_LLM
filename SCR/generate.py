import pickle
import random
from copy import deepcopy
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key='sk-',
)

random.seed(0)

prompt = f"""### Instruction:
You are a legal expert who specializes in comparing user-supplied legal cases to a list of candidate legal cases, which includes titles and content. Your main function is to identify and output the title of the most similar case from the list based on the description provided.
You should only output the case title and not any other information.
"""
input_question = "Please choose the most similar legal case to the case below. You should only output the case title and not any other information.\n"

def generate(data, num_choices=10):
    for i in data:
        # text = prompt + "\n### Input:\nBelow is the detail of a legal case:\n" + i['case_text'] + "\n\n"
        text = f"{prompt}\nConsider the following choices:\n\n"
        choices = deepcopy(i['choice'])[:num_choices]
        random.shuffle(choices)
        for index, j in enumerate(choices):
            text += f"Choice {index+1}:\n" + f"Case title: {j['data']['name']}" + "\n" + f"Case content: {j['content']}" + "\n\n"

        # text += "Please choose the most similar legal case to the case below. You should only output the case title and not any other information.\n\n"
        text += f"### Input:\n{input_question}Here is the detail of a legal case:\n{i['case_text']}\n\n#Response:"
        yield text, i['choice']



# def query_gpt3_5(text):
#     response = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": text,
#             }
#         ],
#         model="gpt-",
#     )

#     return response.choices[0].message.content

if __name__ == '__main__':
    data = pickle.load(open('case1000_test.pkl', 'rb'))
    print(len(data))
    # exit(0)
    # anss = []
    # with open("test.txt", 'r') as f:
    #     anss = f.readlines()
    LOG = []
    for i, (text, choices) in tqdm(enumerate(generate(data))):
        LOG.append(dict(text=text, choices=choices))
        # print(text)
        # print(">>>" * 20 + str(i))
        ans = query_gpt3_5(text).strip()
        LOG[-1]['ans'] = ans
        LOG[-1]['GT'] = [i['data']['name'] for i in choices]
        c = [i['data']['name'] for i in choices]
        try:
            LOG[-1]['index'] = c.index(ans)
        except:
            LOG[-1]['index'] = -1
        print(LOG[-1]['ans'], LOG[-1]['index'])
        pickle.dump(LOG, open('Log_4.pkl', 'wb'))
            # print(c)
            
        # print("<<<" * 20)
        # print(f"ground truth: {choices[0]['data']['name']}")
        # break
        # if i == 10:
        #     break