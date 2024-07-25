from chroma import Chroma
from embeddings import OpenAIEmbeddings
import os
os.environ['OPENAI_API_KEY'] = ''

if __name__ == '__main__':
    embd = OpenAIEmbeddings()
    db = Chroma(embedding_function=embd)
    print(db)