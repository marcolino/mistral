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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
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
LOG_LEVEL = logging.DEBUG if not ENV or ENV == "DEV" else logging.WARNING # TODO: use a switch...
logging.basicConfig(
  level = LOG_LEVEL,
  format = "🔵 %(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Env is {ENV}")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
  # Startup code
  logger.info("Application startup - initializing resources")
  
  # Initializations code
  status = data_load()
  data_sync()

  yield  # The application runs here
  
  # Shutdown code
  logger.info("Application shutdown - cleaning up resources")
  # Here you could close database connections, etc.

app = FastAPI(lifespan = lifespan)

@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, e: Exception):
  return JSONResponse(
    status_code = 500,
    content = {
      "message": "Internal server error",
      "detail": str(e),
      "type": e.__class__.__name__,
    }
  )

# Mount the assets folder to make /assets/* available
app.mount("/assets", StaticFiles(directory = "assets"), name = "assets")

# # Article Model - TODO: in separate file in models/ folder
# class Article(BaseModel):
#   filename: str = ""
#   url: str = ""
#   title: str = ""
#   author: str = ""
#   source_url: str = ""
#   publishing_date: None # TODO: ok?
#   translator: str = ""
#   categories: list = []
#   number: int = 0
#   contents: str = ""


# QueryRequest Model - TODO: in separate file in models/ folder
class QueryRequest(BaseModel):
  query: str
  #top_k: int = Config.FAISS_K_PROTOTYPE
  # ... TODO...


# --- Utilities ---

def data_sync():
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
    return JSONResponse(
      status_code = 400,
      content = {
        "message": "No existing index!",
      }
    )

# --- API Routes ---

@app.post("/download_articles")
def download_articles():
  import download_articles
  count = download_articles.main()
  return JSONResponse(
    status_code = 200,
    content = {
      "message": "Downloaded",
      "articles_count": count,
    }
  )

# @app.post("/sync_index")
# def sync_index():
#   count = sync_faiss_index()
#   return {"status": "index updated", "embeddings_added": count}

@app.post("/query")
def query_articles(req: QueryRequest):
  if index is None:
    return JSONResponse(
      status_code = 400,
      content = {
        "message": "Index not built yet",
      }
    )

  logger.info(f"query: {req.query}")

  if not req.query:
    return JSONResponse(
      status_code = 400,
      content = {
        "message": "Query is empty",
      }
    )

  # Retrieval from index
  query_vec = embedding_model.encode(req.query, show_progress_bar = False)
  try:
    distances, indices = index.search(
      np.array([query_vec])
      , Config.FAISS_K_PROTOTYPE
    )
  except Exception as e:
    return JSONResponse(
      status_code = 500,
      content = {
        "message": "Error searching index\
 (probably the embedding model changed from the indexed one, it should be rebuilt)",
        "detail": str(e),
        "type": e.__class__.__name__,
      }
    )

  # Print all distances - TODO: DEBUG ONLY
  logger.debug(f"<DBG> distances retrieved from index: {distances[0]}")
  
  # Filter and format context with error handling
  context_chunks = []
  total_tokens = 0
  max_tokens = Config.MAX_TOKENS_IN_CONTEXT

  # Take top results regardless of absolute distance,
  # but apply a relative threshold (only if the best result is reasonable)
  valid_indices = []
  if len(distances[0]) > 0:
      best_distance = distances[0][0]
      # If best result is reasonable (< 4.0), use relative threshold
      if best_distance < 4.0:
        logger.debug(f"<DBG> best distance is {best_distance}, good")
        # Take results within 2x the best distance, or absolute threshold
        relative_threshold = min(best_distance * 2.0, Config.DISTANCE_THRESHOLD)
        valid_indices = [(j, dist) for j, dist in zip(indices[0], distances[0]) if dist <= relative_threshold]
      else:
        logger.debug(f"<DBG> best distance is {best_distance}, poor")
        # If even best result is poor, take top 2 anyway
        valid_indices = list(zip(indices[0][:2], distances[0][:2]))
  
  logger.debug(f"<DBG> Valid indices after filtering: {len(valid_indices)}")

  for j, dist in valid_indices:
    if total_tokens >= max_tokens:
      break
          
    try:
      # Safely access metadata
      meta = metadata[j] if j < len(metadata) else {}
      text = texts[j] if j < len(texts) else ""
      
      # Estimate tokens (rough estimate: words × 1.3)
      chunk_tokens = len(text.split()) * 1.3
      if total_tokens + chunk_tokens > max_tokens:
        continue
      total_tokens += chunk_tokens

      # Format chunk
      context_chunks.append(
        #f"Fonte: {meta.get('url', 'sconosciuto')}\n"
        f"Titolo: {meta.get('title', 'sconosciuto')}\n" # Fixed: use 'title' not 'filename'
        #f"Autore: {meta.get('author', 'sconosciuto')}\n"
        f"Data: {meta.get('date', 'sconosciuta')}\n"
        f"Distanza: {dist:.3f}\n" # Debug: show distance
        f"Estratto: {' '.join(text.split()[:150])}\n"
      )
      logger.debug(f"<DBG> Added chunk with distance {dist:.3f}")
    except Exception as e:
      logger.error(f"Error processing chunk {j}: {e}")
      continue
  
  logger.debug(f"<DBG> final context chunks: {len(context_chunks)}")
  logger.debug(f"<DBG> total tokens: {total_tokens}")
  
  # for j, dist in zip(indices[0], distances[0]):
  #   print(dist)
  #   if dist < Config.DISTANCE_THRESHOLD and total_tokens < max_tokens:
  #     try:
  #       # Safely access metadata
  #       meta = metadata[j] if j < len(metadata) else {}
  #       text = texts[j] if j < len(texts) else ""
        
  #       # Estimate tokens (rough estimate: words × 1.3)
  #       chunk_tokens = len(text.split()) * 1.3
  #       if total_tokens + chunk_tokens > max_tokens:
  #         continue
  #       total_tokens += chunk_tokens

  #       # Format chunk
  #       context_chunks.append(
  #         f"Fonte: {meta.get('filename', 'sconosciuto')}\n"
  #         f"Data: {meta.get('publishing_date', 'sconosciuta')}\n"
  #         f"Estratto: {' '.join(text.split()[:150])}\n"
  #       )
  #     except Exception as e:
  #       logger.error(f"Error processing chunk {j}: {e}")
  #       continue
  # logger.debug(f"<DBG> total tokens in context: {total_tokens}")

  # Construct prompt
  # prompt = f"""
  #   Analizza questi documenti e rispondi in italiano CORRETTO.
  #   Documenti: {"\n---\n".join(context_chunks)}
  #   Domanda: {req.query}
  #   Rispondi con frasi complete, basandoti SOLO sui documenti forniti sopra.
  # """
  prompt = f"""
<context>
{"\n\n".join(context_chunks)}
</context>

Sei un assistente per una rivista politica. Ti è stata posta questa domanda: "{req.query}"

Risponde alla domanda basandoti SOLO sui documenti nel contesto sopra (context).
Cita SEMPRE il titolo (e non la data) del documento a cui ti riferisci.
Se i documenti non contengono informazioni sufficienti per rispondere, dillo chiaramente.
Se i documenti contengono informazioni sufficienti per rispondere, fornisci una risposta di 4 frasi al massimo.
Per rispondere utilizza SOLO la lingua Italiana.
Sii preciso.
"""
#Sii conciso e preciso.

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
      n_ctx = 8192,
      n_batch = 512,
      n_threads = 8,
      n_gpu_layers = 0, # Explicitly set to 0 for CPU-only
      temperature = 0.1, # [0-1] 0.1 is more deterministic for factual answers
      repeat_penalty = 1.1, # Repeat penalty
      verbose = False,
    )
  except Exception as e:
    return JSONResponse(
      status_code = 500,
      content = {
        "message": "Error loading LLM",
        "detail": str(e),
        "type": e.__class__.__name__,
      }
    )

  # Generate response - TODO: use Config
  try:
    start_time = time.time()
    response = llm(
      prompt,
      max_tokens = Config.MAX_TOKENS_IN_ANSWER,
      echo = False, # Do not repet prompt in answer
      stream = False, # Ensure we're not streaming unnecessarily
    )
    end_time = time.time()
    total_time = end_time - start_time
    finish_reason = response["choices"][0]["finish_reason"]
    answer = response['choices'][0]['text'].strip()
    logger.debug(f"<DBG> prompt: {prompt}")
    logger.debug(r"<DBG> query llm: {%.1f} seconds" % total_time)
    logger.debug(f"<DBG> finish reason: {finish_reason}")
    logger.info(f"response to query: {answer}")
    return JSONResponse(
      status_code = 200,
      content = {
        "response": answer,
        "total_time": round(total_time, 1),
        "finish_reason": finish_reason,
      }
    )
  except Exception as e:
    logger.error(f"error queryng LLM: {str(e)}") # TODO: always logger.error before return 500 ?
    return JSONResponse(
      status_code = 500,
      content = {
        "message": "error queryng LLM",
        "details": str(e),
      }
    )

@app.get("/favicon.ico")
def favicon():
  return FileResponse("assets/images/favicon.png")

@app.get("/", response_class = HTMLResponse)
def web_ui():
  with open("index.html", encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
    return HTMLResponse(content = f.read())
