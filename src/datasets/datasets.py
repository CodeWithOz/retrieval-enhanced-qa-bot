import openai
import time
import os
import datetime
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
from pinecone_datasets import Dataset


def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")


timed_print(f"loading youtube transcripts dataset")
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube-transcripts")
dataset = Dataset.from_path(file_path)
timed_print(f"loaded youtube transcripts dataset")


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

# view the index
index_stats = index.describe_index_stats()
timed_print(index_stats)

index_total_vector_count =  index_stats.get("total_vector_count", 0)
if index_total_vector_count <= 0:
    batch_size = 100
    timed_print(f"index has no vectors, upserting {dataset.documents.shape[0]} documents in batches of {batch_size}")
    start_timestamp = get_current_timestamp()
    for i, batch in enumerate(dataset.iter_documents(batch_size=batch_size)):
        timed_print(f"upserting batch {i + 1}")
        index.upsert(batch)
        timed_print(f"upserted batch {i + 1}\n")

    end_timestamp = get_current_timestamp()
    timed_print(f"started upserting documents into index at {start_timestamp}")
    timed_print(f"finished upserting documents into index at {end_timestamp}")
else:
    timed_print(f"index {index_name!r} contains {index_total_vector_count}")

# retrieval
openai.api_key = openai_api_key
embed_model = "text-embedding-ada-002"

query = (
    "Which training method should I use for sentence transformers when " +
    "I only have pairs of related sentences?"
)

timed_print("creating openai embedding")
res = openai.Embedding.create(
    input=[query],
    engine=embed_model,
)
timed_print(f"created openai embedding: {res}")

# retrieve from pinecone
xq = res["data"][0]["embedding"]

# get the relevant contexts (including the questions)
res = index.query(vector=xq, top_k=2, include_metadata=True)

timed_print(f"contexts relevant to query: {res}")