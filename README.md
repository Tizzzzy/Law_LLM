# Law_LLM

![](fig.png?raw=true)

# About
In the rapidly evolving field of legal analytics, finding relevant cases and accurately predicting judicial outcomes are challenging because of the complexity of legal language, which often includes specialized terminology, complex syntax, and historical context. Moreover, the subtle distinctions between similar and precedent cases require a deep understanding of legal knowledge. Researchers often conflate these concepts, making it difficult to develop specialized techniques to effectively address these nuanced tasks. In this paper, we introduce the Law Large Language Model (`LawLLM`), a multi-task model specifically designed for the US legal domain to address these challenges. `LawLLM` excels at Similar Case Retrieval (SCR), Precedent Case Recommendation (PCR), and Legal Judgment Prediction (LJP). By clearly distinguishing between precedent and similar cases, we provide essential clarity, guiding future research in developing specialized strategies for these tasks. We propose customized data preprocessing techniques for each task that transform raw legal data into a trainable format. Furthermore, we also use techniques such as in-context learning (ICL) and advanced information retrieval methods in `LawLLM`. The evaluation results demonstrate that `LawLLM` consistently outperforms existing baselines in both zero-shot and few-shot scenarios, offering unparalleled multi-task capabilities and filling critical gaps in the legal domain.

# Requesting model access from META
## 1. Requesting model access from Google
visit this [link](https://ai.google.dev/gemma) and request the access to the Gemma-7B model. 

## 2. Requesting model access from Hugging Face
Once request is approved, use the same email adrress to get the access of the model from HF [here](https://huggingface.co/google/gemma-7b).

Once both requests are approved, follow the below directions.

# Setup

## 1. Environment preparation
```python
git clone https://github.com/Tizzzzy/Law_LLM.git

cd Law_LLM

pip install git+https://github.com/huggingface/transformers

# python 3.10 or higher recommended
pip install -r requirements.txt
pip install pyarrow~=12.0

huggingface-cli login
```

## 2. Authorising HF token
Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

Run below code and enter your token. It will authenticate your HF account
```python
>>> huggingface-cli login

or

>>> from huggingface_hub import login
>>> login(YOUR_HF_TOKEN)
```


# Dataset

## 1. Preparing data for the model

# How to Run

After get all of the legal data from `case.law` into a folder. In `preprocess` folder, you should run `train_test_split.py`. This will seperate your data folder into a train folder and a text folder

## For Legal Judgment Prediction

1. In `LJP` folder, run `gpt_preprocess.py`. This will preprocess all of the legal document in the folder and write it in a csv file. The csv file will have the format of:

| File Name | Train Data |
|-----------|------------|

2. Notice that the code takes two file paths:
   - `folder_path`:  your train folder or test folder from the `Preprocess` step.
   - `csv_file_path`: where do you wish to store the processed data.

## For Precedent Case Recommondation

1. In `PCR` folder, run `precedent_get.py`. This will return a json file that contains all the processed data for PCR task.

2. Notice that the code takes three file paths:
   - `folder_path`: your train folder or test folder from the `Preprocess` step.
   - `json_path`: where do you wish to store the processed data.
   - `cite_path`: where do you wish to store the precedent data (Optional).

## For Similar Case Retrival

1. In `SCR` folder, run `SCR.py`. This will return a json file that contains all the processed data for SCR task.

2. Notice that the code takes two file paths:
   - `csv_file_path`: your processed data path from the `LJP` task.
   - `json_train_path`: where do you wish to store the processed data.
  
## Convert Three Tasks

1. In `preprocess` folder, run `merge_data.py`. This will return a json file that conatins all the processed data from all tasks.

2. Notice that the code takes four file paths:
   - `LJP_file`: your data path for `LJP` task.
   - `SCR_file`: your data path for `SCR` task.
   - `PCR_file`: your data path for `PCR` task.
   - `train_file`: where do you wish to store the final training data.

# Finetune the Model

1. In `train` folder, run `train.py`.
   ```bash
   python train.py --file_path train_file.json --output_dir final_checkpoint
   ```
   - `file_path` is your combined training file.
   - `output_dir` is where you wish to store your model checkpoint.

2. If you want to modify any hyperparameters, feel free to do so.

# Merge and Test the Model

1. In `train` folder, run `LawLLM_merge_4bit.ipynb`. This will merge your previously trained checkpoint with `Gemma` model.

2. Notice that the code takes two file paths:
   - `checkpoint folder`: your trained checkpoint folder.
   - `output_merged_dir`: where do you wish to store your final merged model.
  
3. After you model is ready, simply change the `text` to test the model.

If you like our project, please leave us a star :star:
