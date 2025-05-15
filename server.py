""" server class, implementing an API web server """

import os
import glob
import yaml
import numpy as np
import faiss
import llama_cpp
import pickle
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from config import Config

app = FastAPI()

@app.on_event("startup")
async def startup_event():
  load_faiss_index()

# Mount the assets folder to make /assets/* available
app.mount("/assets", StaticFiles(directory = "assets"), name = "assets")

# Global State
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
index = None
texts = []
metadata = []

# Article Model
class Article(BaseModel):
  filename: str
  url: str
  title: str
  author: str
  source_url: str = ""
  publishing_date: str
  translator: str = ""
  categories: str
  contents: str

class QueryRequest(BaseModel):
  query: str

# --- Utilities ---

def load_articles():
  articles = []
  for file_path in glob.glob(os.path.join(Config.ARTICLES_LOCAL_FOLDER, f"*{Config.ARTICLES_LOCAL_EXTENSION}")):
    with open(file_path, 'r', encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
      print(f"loading article {file_path}")
      try:
        data = yaml.safe_load(f)
        articles.append(data)
      except yaml.YAMLError as e:
        print(f"Error loading YAML file {file_path}: {e}")
        #raise e
  return articles

def chunk_text(text, size = Config.CHUNK_SIZE, overlap = Config.CHUNK_OVERLAP):
  words = text.split()
  chunks = []
  for i in range(0, len(words), size - overlap):
    chunk = ' '.join(words[i:i+size])
    if chunk:
      chunks.append(chunk)
  return chunks

def sync_faiss_index():
  global index, texts, metadata

  last_updated_path = Config.FAISS_INDEX_PATH + "_last_updated.txt"
  os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)

  # Read last update time or default to epoch
  if os.path.exists(last_updated_path):
    with open(last_updated_path, "r", encoding="utf-8") as f:
      last_updated_str = f.read().strip()
      last_updated = datetime.datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
  else:
    last_updated = datetime.datetime.min

  articles = []
  new_files = []

  for file_path in glob.glob(os.path.join(Config.ARTICLES_LOCAL_FOLDER, f"*{Config.ARTICLES_LOCAL_EXTENSION}")):
    try:
      mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
      if mtime > last_updated:
        with open(file_path, 'r', encoding=Config.ARTICLES_LOCAL_ENCODING) as f:
          article = yaml.safe_load(f)
          if article:
            article['__filepath'] = file_path
            article['__mtime'] = mtime
            articles.append(article)
            new_files.append(file_path)
    except Exception as e:
      print(f"Error reading {file_path}: {e}")

  if not articles:
    print("No new articles to index.")
    return 0

  # Load or initialize index
  if os.path.exists(Config.FAISS_INDEX_PATH) and \
     os.path.exists(Config.METADATA_PATH) and \
     os.path.exists(Config.TEXTS_PATH):
    
    print("Loading existing FAISS index...")
    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    with open(Config.METADATA_PATH, 'rb') as f:
      metadata = pickle.load(f)
    with open(Config.TEXTS_PATH, 'rb') as f:
      texts = pickle.load(f)
  else:
    print("Creating new FAISS index...")
    index = None
    metadata = []
    texts = []

  embeddings = []

  for article in articles:
    try:
      chunks = chunk_text(article['contents'])
      filename = os.path.basename(article.get('filename') or article.get('url') or 'unknown')
      print(f"Indexing: {filename} ({len(chunks)} chunks)")

      for chunk in chunks:
        texts.append(chunk)
        metadata.append({
          'url': article.get('url', ''),
          'title': article.get('title', ''),
          'author': article.get('author', ''),
          'date': article.get('publishing_date', ''),
          'filename': filename
        })
        embeddings.append(embedding_model.encode(chunk))
    except Exception as e:
      print(f"Error chunking {article.get('__filepath')}: {e}")

  if not embeddings:
    print("No new articles to add.")
    return 0

  # Create or append to index
  embeddings_array = np.array(embeddings).astype('float32')
  if index is None:
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dim)
  index.add(embeddings_array)

  # Save everything
  faiss.write_index(index, Config.FAISS_INDEX_PATH)
  with open(Config.METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)
  with open(Config.TEXTS_PATH, 'wb') as f:
    pickle.dump(texts, f)

  # Save timestamp
  now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  with open(last_updated_path, "w", encoding="utf-8") as f:
    f.write(now_str)

  print(f"Indexed {len(embeddings)} new chunks from {len(new_files)} articles.")
  return len(embeddings)

def load_faiss_index():
  global index, texts, metadata
  
  if \
    os.path.exists(Config.FAISS_INDEX_PATH) and \
    os.path.exists(Config.METADATA_PATH) and \
    os.path.exists(Config.TEXTS_PATH) \
  :
    print("Loading existing index...")
    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    with open(Config.METADATA_PATH, 'rb') as f:
      metadata = pickle.load(f)
    with open(Config.TEXTS_PATH, 'rb') as f:
      texts = pickle.load(f)
  else:
    print("No existing index, building new index...")
    build_faiss_index()



# --- API Routes ---

@app.post("/download_articles")
def download_articles(article: Article):
  import download_articles

  download_articles.main()
  return {"status": "downloaded", "filename": article.filename}

@app.post("/sync_index")
def sync_index():
  count = sync_faiss_index()
  return {"status": "index updated", "embeddings_added": count}

@app.post("/query")
def query_articles(req: QueryRequest):
  if index is None:
    return {"error": "Index not built yet"}

  # Retrieval from index
  query_vec = embedding_model.encode(req.query)
  distances, indices = index.search(np.array([query_vec]), Config.FAISS_K_PROTOTYPE)

  # Filter and format context with error handling
  context_chunks = []
  for j, dist in zip(indices[0], distances[0]):
    if dist < Config.DISTANCE_THRESHOLD:
      try:
        # Safely access metadata
        meta = metadata[j] if j < len(metadata) else {}
        text = texts[j] if j < len(texts) else ""
        
        # Format chunk
        context_chunks.append(
          f"Fonte: {meta.get('filename', 'sconosciuto')}\n"
          f"Data: {meta.get('publishing_date', 'sconosciuta')}\n"
          f"Estratto: {' '.join(text.split()[:150])}\n"
        )
      except Exception as e:
        print(f"Error processing chunk {j}: {e}")
        continue

  # Construct prompt
  prompt = f"""
Analizza questi documenti e rispondi in italiano completo.
    
Documenti:
{"\n---\n".join(context_chunks)}

Domanda: {req.query}

Rispondi con 3 frasi complete, basandoti SOLO sui documenti forniti sopra.
"""

  # Load LLM model - TODO: use Config
  try:
    """
      Parameter            Purpose                                                         Typical Values
      model_path           Path to the .gguf model file                                    "./models/mistral-7b.gguf"
      n_ctx                Max context length (input + output tokens)                      2048, 4096, 8192
      n_threads            Number of CPU threads to use                                    4, 8, 10 (== your CPU cores)
      n_batch              Number of tokens to process at once                             32, 64, 128 (higher = faster, more RAM)
      temperature          Controls randomness of generation (lower = more deterministic)  0.1 - 1.0, usually 0.3 - 0.7
      top_k                Limits token sampling to top-K probable tokens                  40, 100
      top_p                Nucleus sampling: % of probability mass to include              0.8 - 0.95
      repeat_penalty       Penalizes token repetition                                      1.0 1.3
      last_n_tokens_size   How many tokens to remember for repetition penalty              64, 128
      verbose              Logs internal behavior (set False to reduce console clutter)    True / False
    """
    llm = llama_cpp.Llama(
      model_path = Config.MISTRAL_MODEL_PATH,
      n_ctx = 32768, #4096,
      chat_format = "llama-2",
      n_threads = 8,
      # n_batch = 128,
      # top_k = 40,
      # top_p = 0.95,
      repeat_penalty = 1.3,
      # last_n_tokens_size = 128,
      stop = [], # empty list means no stop tokens
      temperature = 0.5,
      verbose = False,
    )
  except Exception as e:
    return {"error loading LLM:": str(e)}

  # Generate response - TODO: use Config
  try:
    start_time = time.time()
    response = llm(
      prompt,
      max_tokens = 768,
      echo = False,
    )
    print("--- query llm: {%.1f} seconds ---" % (time.time() - start_time))
    print(f"--- finish reason: {response["choices"][0]["finish_reason"]} ---")
    print(repr(response))

    return {"response": response['choices'][0]['text'].strip()}

    # response = llm(prompt)
    # return {"response": response['choices'][0]['text'].strip()}
  except Exception as e:
    return {"error queryng LLM:": str(e)}

@app.get("/favicon.ico")
def favicon():
  return FileResponse("assets/images/favicon.png")

@app.get("/", response_class = HTMLResponse)
def web_ui():
  with open("index.html", encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
    return HTMLResponse(content = f.read())
