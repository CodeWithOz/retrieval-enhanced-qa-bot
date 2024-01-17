import os
import pinecone
from dotenv import load_dotenv
from pinecone_datasets import load_dataset

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

dataset = load_dataset("youtube-transcripts-text-embedding-ada-002")

# drop the sparse values as they are not needed for this example
dataset.documents.drop(["metadata", "sparse_values"], axis=1, inplace=True)
dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)

# initialize pinecone
pinecone.init(api_key=pinecone_api_key)
