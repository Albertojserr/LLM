from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import time
import pandas as pd
import os
import requests
def _get_embeddings(text_chunk):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunk)
    return embeddings


def calculate_cosine_similarity(text_chunk, embeddings_df):
    # use the _get_embeddings function the retrieve the embeddings for the text chunk
    sentence_embedding = _get_embeddings(text_chunk)

    # combine all dimensions of the vector embeddings to one array
    embeddings_df['embeddings_array'] = embeddings_df.apply(lambda row: row.values[:-1], axis=1)

    # start the timer
    start_time = time.time()
    print(start_time)

    # create a list to store the calculated cosine similarity
    cos_sim = []

    for index, row in embeddings_df.iterrows():
        A = row.embeddings_array
        B = sentence_embedding

        # calculate the cosine similarity
        cosine = np.dot(A,B)/(norm(A)*norm(B))

        cos_sim.append(cosine)

    embeddings_cosine_df = embeddings_df
    embeddings_cosine_df["cos_sim"] = cos_sim
    embeddings_cosine_df.sort_values(by=["cos_sim"], ascending=False)

    # stop the timer
    end_time = time.time()

    # calculate the time needed to calculate the similarities
    elapsed_time = (end_time - start_time)
    print("Execution Time: ", elapsed_time, "seconds")

    return embeddings_cosine_df


# Load embeddings_df.csv into data frame
embeddings_df = pd.read_csv('./embeddings_df1.csv')
# test query sentence
text_chunk = "Lilies are white."

embeddings_df['embeddings_array'] = embeddings_df.apply(lambda row: row.values[:-1], axis=1)
sentence_embedding = _get_embeddings(text_chunk)
for index, row in embeddings_df.iterrows():
        print(row.embeddings_array)
        print(sentence_embedding)

# calculate cosine similarity
embeddings_cosine_df = calculate_cosine_similarity(text_chunk, embeddings_df)

# save data frame with text chunks and embeddings to csv
embeddings_cosine_df.to_csv('embeddings_cosine_df.csv', index=False)

# rank based on similarity
similarity_ranked_df = embeddings_cosine_df[["text_chunk", "cos_sim"]].sort_values(by=["cos_sim"], ascending=False)