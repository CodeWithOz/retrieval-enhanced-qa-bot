import os
import datetime
from dotenv import load_dotenv
from pinecone_datasets import load_dataset

def get_current_timestamp():
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")


def get_current_timestamp_prefix():
    return f"{get_current_timestamp()} - "


def timed_print(msg: str):
    print(f"{get_current_timestamp_prefix()}{msg}")


load_dotenv()


timed_print(f"loading wikipedia dataset")
dataset = load_dataset("wikipedia-simple-text-embedding-ada-002-100K")
timed_print(f"loaded wikipedia dataset")

timed_print(f"dropping metadata column")
dataset.documents.drop(["metadata"], axis=1, inplace=True)
timed_print(f"dropped metadata column")
timed_print(f"renaming blob column")
dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)
timed_print(f"renamed blob column")

# save the dataset locally
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wikipedia-dataset")
timed_print(f"saving dataset locally")
dataset.to_path(file_path)
timed_print(f"saved dataset locally")
