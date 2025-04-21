import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np

class BertEmbedding:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()

    def encode(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:,0,:].numpy()  # [CLS] token embeddings
        return embeddings


import faiss

def faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def faiss_search(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return indices, distances


import chromadb

def chroma_init(collection_name="issue_jira"):
    client = chromadb.Client()
    collection = client.create_collection(collection_name)
    return collection

def chroma_add(collection, embeddings, metadata, ids):
    collection.add(
        embeddings=embeddings.tolist(),
        metadatas=metadata,
        ids=ids
    )

def chroma_query(collection, query_embedding, k=5):
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )
    return results

import pandas as pd
from pathlib import Path

# Paths
bert_weights_path = "bert_weights"
issues_excel_path = "issues.xlsx"
jira_excel_path = "jira.xlsx"

# Load Data
issues_df = pd.read_excel(issues_excel_path)
jira_df = pd.read_excel(jira_excel_path)

# Combine relevant text fields
issues_df["text"] = issues_df["title"] + " " + issues_df["summary"] + " " + issues_df["description"]
jira_df["text"] = jira_df["summary"] + " " + jira_df["issue"]

# Initialize embedding
embedder = BertEmbedding(bert_weights_path)

# Generate embeddings
issues_embeddings = embedder.encode(issues_df["text"].tolist())
jira_embeddings = embedder.encode(jira_df["text"].tolist())

# Combine data
all_embeddings = np.vstack([issues_embeddings, jira_embeddings])
metadata = (
    [{"type": "issue", **row} for _, row in issues_df.iterrows()] + 
    [{"type": "jira", **row} for _, row in jira_df.iterrows()]
)
ids = [f"issue_{i}" for i in range(len(issues_df))] + [f"jira_{i}" for i in range(len(jira_df))]

# ---------------------------
# FAISS Implementation
# ---------------------------
faiss_indexer = faiss_index(all_embeddings)

# FAISS Search Example:
query_text = "Example search issue description"
query_embedding = embedder.encode([query_text])

faiss_indices, faiss_distances = faiss_search(faiss_indexer, query_embedding)

print("FAISS Search Results:")
for idx, dist in zip(faiss_indices[0], faiss_distances[0]):
    print(f"Distance: {dist:.4f}, Metadata: {metadata[idx]}")

# ---------------------------
# ChromaDB Implementation
# ---------------------------
collection = chroma_init()

# Add data
chroma_add(collection, all_embeddings, metadata, ids)

# ChromaDB Search Example:
chroma_results = chroma_query(collection, query_embedding)

print("\nChromaDB Search Results:")
for md, dist, id in zip(chroma_results['metadatas'][0], chroma_results['distances'][0], chroma_results['ids'][0]):
    print(f"ID: {id}, Distance: {dist:.4f}, Metadata: {md}")




