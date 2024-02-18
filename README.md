# Law_LLM

# Requesting model access from META
## 1. Requesting model access from META
visit this [link](https://ai.meta.com/llama/) and request the access to the Llama-2 models.  

## 2. Requesting model access from Hugging Face
Once request is approved, use the same email adrress to get the access of the model from HF [here](https://huggingface.co/meta-llama/Llama-2-7b).  

Once both requests are approved, follow the below directions.

# Dataset

## 1. Preparing data for the model

# Finetune the model

## 1. Environment preparation
```
git clone https://github.com/facebookresearch/llama-recipes.git

cd llama-recipes

pip install git+https://github.com/huggingface/transformers

# python 3.10 or higher recommended
pip install -r requirements.txt
pip install pyarrow~=12.0

huggingface-cli login
```

## 2. Authorising HF token
Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

Run below code and enter your token. It will authenticate your HF account
```
>>> huggingface-cli login

or

>>> from huggingface_hub import login
>>> login(YOUR_HF_TOKEN)
```

## 3. Run Code

Make sure you changed all the file paths and hyperparameter in the `finetune.py`, and simply run with:
```
python finetune.py
```

# Prediction

Once model is trained, copy your best checkpoint path and run the [predict]() to make the predictions.

# Evaluation

Once you save the `predictions.csv` file, run [evaluate]() to get the scores on the dataset. 

