from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text_chunks = [
    "The sky is blue.",
    "The grass is green.",
    "The sun is shining.",
    "I love chocolate.",
    "Pizza is delicious.",
    "Coding is fun.",
    "Roses are red.",
    "Violets are blue.",
    "Water is essential for life.",
    "The moon orbits the Earth.",
]
columnas=[]
for i in range(0,384):
    columnas.append(str(i))
columnas.append("text_chunk")
embeddings = model.encode(text_chunks)
text = np.array(text_chunks).reshape((10, 1))
print(embeddings.shape)
print(text)
print(text.shape)
emb=np.concatenate((embeddings, text), axis=1)

df = pd.DataFrame(emb, columns = columnas)
df.to_csv('./embeddings_df.csv', index=False)
