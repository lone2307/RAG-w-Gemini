from urllib import response
import google.generativeai as genai
import chromadb
from settings import *
from sentence_transformers import SentenceTransformer


# Get API key
API_KEY = input("API: ")

# Init model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel()

# Get user query
query = input("Question: ")

# Setting up Chroma
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

# Embed and normalize the query
query_embedding = model.encode(
    ["best strength training exercises"], 
    normalize_embeddings=True
)

# Get query
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

# Setting up prompt
retrieved_chunks = results["documents"][0]

context = "\n".join(retrieved_chunks)
prompt =f"""
Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer:

"""

# Response
response = model.generate_content(prompt)
print(prompt)
print(response.text)
