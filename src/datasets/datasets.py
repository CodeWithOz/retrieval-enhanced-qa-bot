from pinecone_datasets import load_dataset, list_datasets

dataset = load_dataset("youtube-transcripts-text-embedding-ada-002")

# drop the sparse values as they are not needed for this example
dataset.documents.drop(["metadata"], axis=1, inplace=True)
dataset.documents.rename(columns={"blob": "metadata"}, inplace=True)
