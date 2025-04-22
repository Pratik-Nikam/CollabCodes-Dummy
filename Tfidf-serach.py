import pandas as pd

# Load Excel files
issues_df = pd.read_excel('issues.xlsx')
jira_df = pd.read_excel('jira.xlsx')

# Combine relevant text fields into a single string for each entry
issues_df['combined_text'] = issues_df[['title', 'summary', 'description']].fillna('').agg(' '.join, axis=1)
jira_df['combined_text'] = jira_df[['summary', 'issue']].fillna('').agg(' '.join, axis=1)


from sklearn.feature_extraction.text import TfidfVectorizer

# Combine all texts for fitting the vectorizer
all_texts = pd.concat([issues_df['combined_text'], jira_df['combined_text']])

# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(all_texts)

# Transform texts into TF-IDF vectors
issues_tfidf = vectorizer.transform(issues_df['combined_text'])
jira_tfidf = vectorizer.transform(jira_df['combined_text'])



from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_similar_entries(query, tfidf_matrix, data_df, top_n=5):
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between the query and the TF-IDF matrix
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get indices of top matching entries
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    
    # Retrieve and return the top matching entries
    return data_df.iloc[top_indices], similarity_scores[top_indices]


query_issue = "Application crashes when submitting the form"
similar_jira_entries, scores = find_similar_entries(query_issue, jira_tfidf, jira_df)

# Display the results
for idx, (entry, score) in enumerate(zip(similar_jira_entries.itertuples(), scores), 1):
    print(f"{idx}. Score: {score:.4f} | Summary: {entry.summary}")

query_jira = "Form submission leads to unexpected error"
similar_issue_entries, scores = find_similar_entries(query_jira, issues_tfidf, issues_df)

# Display the results
for idx, (entry, score) in enumerate(zip(similar_issue_entries.itertuples(), scores), 1):
    print(f"{idx}. Score: {score:.4f} | Title: {entry.title}")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "The cat in the hat disabled the alarm.",
    "A quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog quickly.",
    "The dog barks at the mailman."
]

# Step 1: Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 2: Apply SVD
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

# Step 3: Compute similarity
similarity_matrix = cosine_similarity(X_lsa)

# Display similarity matrix
print(pd.DataFrame(similarity_matrix))



from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The cat in the hat disabled the alarm.",
    "A quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog quickly.",
    "The dog barks at the mailman."
]

# Preprocessing: Tokenization
def tokenize(text):
    return set(text.lower().split())

# Create MinHash signatures
minhashes = []
for doc in documents:
    m = MinHash(num_perm=128)
    for d in tokenize(doc):
        m.update(d.encode('utf8'))
    minhashes.append(m)

# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for i, m in enumerate(minhashes):
    lsh.insert(f"doc_{i}", m)

# Query for similar documents
query = minhashes[0]
result = lsh.query(query)
print(f"Documents similar to doc_0: {result}")









