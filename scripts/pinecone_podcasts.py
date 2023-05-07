"Downloads a zip file containing podcast transcript data and indexes it using Pinecone."

import json
import os
import zipfile

from dotenv import load_dotenv
import pinecone
from tqdm.auto import tqdm
import wget


print("Init Pinecone")
load_dotenv()
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(os.getenv("PINECONE_API_KEY"), environment=pinecone_env)
print(f"Pinecone: {pinecone.whoami()}")

INDEX_NAME = "podcasts"
if INDEX_NAME in pinecone.list_indexes():
    print(f"Removing existing index `{INDEX_NAME}`")
    pinecone.delete_index(INDEX_NAME)

print(f"Creating new Pinecone index `{INDEX_NAME}`")
pinecone.create_index(name=INDEX_NAME, dimension=1536)
index = pinecone.Index(index_name=INDEX_NAME)

FILE_NAME = "sysk_podcast_transcripts_embedded.json"
file_name_zip = f"{FILE_NAME}.zip"
content_url = f"https://cdn.openai.com/API/examples/data/{file_name_zip}"

print(f"Starting download {file_name_zip}")
wget.download(content_url)

print(f"\nExtracting {file_name_zip}")
with zipfile.ZipFile(file_name_zip, "r") as zip_ref:
    zip_ref.extractall(".")

print(f"Loading {FILE_NAME}")
with open(FILE_NAME, encoding="utf-8") as f:
    processed_podcasts = json.load(f)

print("Adding the text embeddings to Pinecone")
BATCH_SIZE = 100
meta_keys = ["filename", "title", "text_chunk", "url"]
for i in tqdm(range(0, len(processed_podcasts), BATCH_SIZE)):
    i_end = min(len(processed_podcasts), i + BATCH_SIZE)
    meta_batch = processed_podcasts[i:i_end]
    ids_batch = [x["cleaned_id"] for x in meta_batch]
    embeds = [x["embedding"] for x in meta_batch]
    meta_batch = [{key: x[key] for key in meta_keys} for x in meta_batch]
    index.upsert(vectors=list(zip(ids_batch, embeds, meta_batch)))

print("Clean up")
os.remove(FILE_NAME)
os.remove(file_name_zip)

print("Finished")
