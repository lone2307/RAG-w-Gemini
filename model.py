import chromadb
from settings import *
from dataset_cleaning import cleaned
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# Generate clean dataset
dataset = cleaned()

# Splitting dataset
splitter = RecursiveCharacterTextSplitter(chunk_size = 300, overlap = 50)
dataset = splitter.split_text(dataset)

# Chunking data
data_chunk = []
for idx, dset in enumerate(dataset):
    chunk = {
        "id" : f"{DATASET_NAME}_{idx}",
        "document": dset,
    }
    data_chunk.append(chunk)

# Embedding chunks
embed_model = SentenceTransformer(EMBEDDING_MODEL)
embedding = embed_model.encode(data_chunk)

# ChromaDB for fetching chunks of text
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# Store into Chroma DB
for i, chunk in enumerate(data_chunk):
    collection.add(
        documents=[chunk['text']],
        ids=[chunk['id']],
        embeddings=[embedding[i]],
        metadata={"source": chunk['source']}
    )