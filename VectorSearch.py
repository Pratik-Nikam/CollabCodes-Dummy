# Installation commands
# pip install oracledb jira nltk transformers torch chromadb pandas

# Import necessary libraries
import oracledb
from jira import JIRA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from transformers import BertModel, BertTokenizer
import torch
import chromadb
import pandas as pd

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Database connection details - fill in your credentials
oracle_username = "your_username"
oracle_password = "your_password"
oracle_dsn = "your_host:port/service_name"

# Jira connection details - fill in your credentials
jira_server = "[invalid url, do not cite]
jira_username = "your_email"
jira_api_token = "your_api_token"

# Path to BERT weights folder
bert_weights_path = "/path/to/your/bert/weights"

# Step 1: Extract tickets from Oracle
connection = oracledb.connect(
    user=oracle_username,
    password=oracle_password,
    dsn=oracle_dsn
)

query = "SELECT ticket_id, description FROM tickets WHERE status = 'OPEN'"
cursor = connection.cursor()
cursor.execute(query)
tickets = [{"ticket_id": row[0], "description": row[1]} for row in cursor.fetchall()]
cursor.close()
connection.close()

# Step 2: Extract issues from Jira
jira = JIRA(
    server=jira_server,
    basic_auth=(jira_username, jira_api_token)
)

issues = jira.search_issues("project=YOUR_PROJECT_KEY", maxResults=100)
jira_issues = [{"issue_key": issue.key, "summary": issue.fields.summary, 
                "description": issue.fields.description or ""} for issue in issues]

# Step 3: Text preprocessing
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

for ticket in tickets:
    ticket["processed_text"] = preprocess_text(ticket["description"])

for issue in jira_issues:
    issue["processed_text"] = preprocess_text(issue["summary"] + " " + issue["description"])

# Step 4: Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_weights_path)
model = BertModel.from_pretrained(bert_weights_path)
model.eval()

# Step 5: Generate embeddings
def generate_bert_embeddings(texts, tokenizer, model, max_length=128, batch_size=16):
    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings_batch = outputs[0].mean(dim=1).cpu().numpy().tolist()
        
        embeddings.extend(embeddings_batch)
    
    return embeddings

ticket_texts = [ticket["processed_text"] for ticket in tickets]
ticket_embeddings = generate_bert_embeddings(ticket_texts, tokenizer, model)

issue_texts = [issue["processed_text"] for issue in jira_issues]
issue_embeddings = generate_bert_embeddings(issue_texts, tokenizer, model)

# Step 6: Store embeddings in ChromaDB
client = chromadb.Client()
collection = client.create_collection("ticket_issue_similarities")

# Add ticket embeddings
collection.add(
    embeddings=ticket_embeddings,
    metadatas=[{"type": "ticket", "id": t["ticket_id"]} for t in tickets],
    ids=[f"ticket_{t['ticket_id']}" for t in tickets]
)

# Add Jira issue embeddings
collection.add(
    embeddings=issue_embeddings,
    metadatas=[{"type": "issue", "id": i["issue_key"]} for i in jira_issues],
    ids=[f"issue_{i['issue_key']}" for i in jira_issues]
)

# Step 7: Find similar items
def find_similar_items(query_embedding, collection, n_results=5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["ids"][0], results["distances"][0]

# Find relationships
relationships = []
for i, ticket in enumerate(tickets):
    similar_ids, distances = find_similar_items(ticket_embeddings[i], collection)
    for sim_id, distance in zip(similar_ids, distances):
        if sim_id.startswith("issue_") and distance < 0.7:  # Adjust threshold as needed
            relationships.append({
                "ticket_id": ticket["ticket_id"],
                "issue_key": sim_id.replace("issue_", ""),
                "similarity_score": 1 - distance
            })

# Step 8: Save results to CSV
df = pd.DataFrame(relationships)
df.to_csv("ticket_issue_relationships.csv", index=False)

print("Relationships saved to ticket_issue_relationships.csv")
