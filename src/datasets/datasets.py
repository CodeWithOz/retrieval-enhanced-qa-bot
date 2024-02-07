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


timed_print(f"loading wikipedia dataset")
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikipedia-dataset")
dataset = Dataset.from_path(file_path)
timed_print(f"loaded wikipedia dataset")


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

index_total_vector_count = index_stats.get("total_vector_count", 0)
if index_total_vector_count <= dataset.documents.shape[0]:
    batch_size = 100
    timed_print(f"index has no vectors, upserting {dataset.documents.shape[0]} documents in batches of {batch_size}")
    start_timestamp = get_current_timestamp()
    last_batch = int((index_stats.get("index_fullness", 0.0) * batch_size) + 1)
    for i, batch in enumerate(dataset.iter_documents(batch_size=batch_size)):
        if i+1 < last_batch:
            continue
        timed_print(f"upserting batch {i + 1}")
        try:
            index.upsert(batch)
        except Exception as e:
            timed_print(f"Error during upsert:\n{str(e)}")
            if i+1 == last_batch:
                # the same batch failed, stop now
                raise e
            # wait a bit then try again
            timed_print(f"waiting for 10 seconds before re-attempting upsert of batch {i + 1}")
            time.sleep(10)
            timed_print(f"re-attempting upsert of batch {i + 1}")
            try:
                index.upsert(batch)
            except Exception as e:
                timed_print(f"Error during first re-attempted upsert:\n{str(e)}")
                timed_print(f"waiting for another 10 seconds before re-attempting upsert of batch {i + 1} one more time")
                time.sleep(10)
                timed_print(f"re-attempting upsert of batch {i + 1} one more time")
                index.upsert(batch)
        last_batch = i + 1
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

token_limit = 3750

def retrieve(query: str):
    timed_print("creating openai embedding")
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model,
    )
    timed_print(f"created openai embedding")

    # retrieve from pinecone
    xq = res["data"][0]["embedding"]

    # get the relevant contexts
    contexts = []
    time_waited = 0
    time_to_wait = 1 * 60
    min_num_contexts = 3
    while (len(contexts) < min_num_contexts and time_waited < time_to_wait):
        timed_print("querying vector in pinecone index")
        res = index.query(vector=xq, top_k=min_num_contexts, include_metadata=True)
        timed_print("queried vector in pinecone index")

        contexts.extend([
            x["metadata"]["text"] for x in res["matches"]
        ])
        timed_print(f"scores of retrieved contexts: {[x['score'] for x in res['matches']]}")
        timed_print(f"Retrieved {len(contexts)} contexts from index, sleeping for 15 seconds...")
        time.sleep(15)
        time_waited += 15

    if time_waited >= time_to_wait and len(contexts) == 0:
        timed_print("Timed out when retrieving contexts")
        contexts.append("No contexts retrieved. Try to answer the question yourself!")

    # include the retrieved contexts in the prompt
    prompt_start = (
        "Answer the question based on the context below.\n\n"
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    # append contexts until we hit the character limit
    for i in range(len(contexts)):
        joined_contexts = "\n\n---\n\n".join(contexts[:i+1])
        if len(joined_contexts) >= token_limit:
            # only join up to the previous context
            joined_contexts = "\n\n---\n\n".join(contexts[:i])
            prompt = (
                prompt_start +
                f"\n\n{joined_contexts}\n\n" +
                prompt_end
            )
            break
        elif i == len(contexts) - 1:
            # this is the last context, so we create the prompt at this point
            joined_contexts = "\n\n---\n\n".join(contexts)
            prompt = (
                prompt_start +
                f"\n\n{joined_contexts}\n\n" +
                prompt_end
            )

    return prompt


def complete(prompt: str):
    # instructions
    sys_prompt = "You are a helpful assistant that always answers questions."
    timed_print(f"\n\nprompt:\n{prompt}\n\n")
    # query text-davinci-003
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return res["choices"][0]["message"]["content"].strip()

timed_print(f"\n\n{complete(retrieve(query=query))}")
