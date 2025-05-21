""" server class, implementing an API web server """

import os
import sys
import glob
import yaml
import numpy as np
import faiss
import llama_cpp
import pickle
import time
import logging
import traceback
from typing import AsyncIterator
from datetime import datetime, date
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from utils import data_dump, data_load
from config import Config

# Global State
status = {}
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
index = None
texts = []
metadata = []

# Setup logging
ENV = os.getenv("ENV")
LOG_LEVEL = logging.DEBUG if not ENV or ENV == "DEVEL" else logging.WARNING
logging.basicConfig(
  level = LOG_LEVEL,
  format = "🔵 %(asctime)s - %(levelname)s - %(message)s"
  #format = "🔵 %(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Env is {ENV}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
  # Startup code
  logger.info("Application startup - initializing resources")
  
  # Initializations code
  status = data_load()
  index_load()

  yield  # The application runs here
  
  # Shutdown code
  logger.info("Application shutdown - cleaning up resources")
  # Here you could close database connections, etc.

app = FastAPI(lifespan = lifespan)

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
  return JSONResponse(
    status_code = 500,
    content = {
      "message": "Internal server error",
      "detail": str(exc),
      "type": exc.__class__.__name__,
    }
  )

# Mount the assets folder to make /assets/* available
app.mount("/assets", StaticFiles(directory = "assets"), name = "assets")

# Article Model - TODO: in separate file in models/ folder
class Article(BaseModel):
  filename: str = ""
  url: str = ""
  title: str = ""
  author: str = ""
  source_url: str = ""
  publishing_date: None # TODO: ok?
  translator: str = ""
  categories: list = []
  number: int = 0
  contents: str = ""


# QueryRequest Model - TODO: in separate file in models/ folder
class QueryRequest(BaseModel):
  query: str
  #top_k: int = Config.FAISS_K_PROTOTYPE
  # ... TODO...


# --- Utilities ---

def index_load():
  global index, texts, metadata
  
  if all(os.path.exists(path) for path in [
    Config.FAISS_INDEX_PATH,
    Config.METADATA_PATH,
    Config.TEXTS_PATH
  ]):
    logger.info("Loading existing index...")
    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    with open(Config.METADATA_PATH, 'rb') as f:
      metadata = pickle.load(f)
    with open(Config.TEXTS_PATH, 'rb') as f:
      texts = pickle.load(f)
  else:
    logger.error("No existing index!")
    raise



# --- API Routes ---

@app.post("/download_articles")
def download_articles():
  import download_articles
  count = download_articles.main()
  return {"status": "downloaded", "articles_count": count}

# @app.post("/sync_index")
# def sync_index():
#   count = sync_faiss_index()
#   return {"status": "index updated", "embeddings_added": count}

@app.post("/query")
def query_articles(req: QueryRequest):
  if index is None:
    return {"error": "Index not built yet"}

  logger.info(f"query: {req.query}")

  # Retrieval from index
  query_vec = embedding_model.encode(req.query, show_progress_bar = False)
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
    Analizza questi documenti e rispondi in italiano CORRETTO.
    Documenti: {"\n---\n".join(context_chunks)}
    Domanda: {req.query}
    Rispondi con frasi complete, basandoti SOLO sui documenti forniti sopra.
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
      #n_threads = 8,
      # n_batch = 128,
      # top_k = 40,
      # top_p = 0.95,
      repeat_penalty = 1.5,
      # last_n_tokens_size = 128,
      stop = [], # empty list means no stop tokens
      temperature = 0.2,
      verbose = False,
    )
  except Exception as e:
    return {"error loading LLM:": str(e)}

  # Generate response - TODO: use Config
  try:
    start_time = time.time()
    response = llm(
      prompt,
      #max_tokens = 768,
      max_tokens = 256,
      echo = False,
    )
    logger.debug(f"--- prompt: {prompt}")
    logger.debug(r"--- query llm: {%.1f} seconds ---" % (time.time() - start_time))
    logger.debug(f"--- finish reason: {response["choices"][0]["finish_reason"]} ---")
    answer = response['choices'][0]['text'].strip()
    logger.info(f"response to query: {answer}")
    return {"response": response['choices'][0]['text'].strip()}
  except Exception as e:
    logger.error(f"error queryng LLM: {str(e)}")
    return {"error": f"error queryng LLM: {str(e)}"}

@app.get("/favicon.ico")
def favicon():
  return FileResponse("assets/images/favicon.png")

@app.get("/", response_class = HTMLResponse)
def web_ui():
  with open("index.html", encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
    return HTMLResponse(content = f.read())
