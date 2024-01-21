import os
import datetime
from time import time
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
from pinecone_datasets import load_dataset

def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")


timed_print(f"loading youtube transcripts dataset")
dataset = load_dataset("youtube-transcripts-text-embedding-ada-002")
timed_print(f"loaded youtube transcripts dataset")


# drop the sparse values as they are not needed for this example
timed_print(f"dropping metadata and sparse_values columns")
dataset.documents.drop(["metadata", "sparse_values"], axis=1, inplace=True)
timed_print(f"dropped metadata and sparse_values columns")
timed_print(f"renaming blob column")
dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)
timed_print(f"renamed blob column")


# store embeddings for vector search
pc = Pinecone(api_key=pinecone_api_key)
spec = PodSpec(environment=pinecone_environment)
index_name = "gen-qa-openai-fast-pod-index"
# create index if it doesn't already exist
current_indexes = pc.list_indexes()
if index_name not in current_indexes.names():
    timed_print(f"index {index_name!r} does not exist, creating it")
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric="cosine",
        spec=spec,
    )
    timed_print(f"created index {index_name!r}")

# connect to the index
index = pc.Index(index_name)

# # view the index
index_stats = index.describe_index_stats()
timed_print(index_stats)
