""" configuration class """

from dataclasses import dataclass

@dataclass
class Config:
  """ configuration class values """
  ARTICLES_SOURCE_URL_DOMAIN = "www.resistenze.org"
  ARTICLES_SOURCE_URL_BASE = f"https://{ARTICLES_SOURCE_URL_DOMAIN}"
  ARTICLES_SOURCE_URL_PATH = "sito"
  ARTICLES_SOURCE_URL_PATTERN = (
    f"{ARTICLES_SOURCE_URL_BASE}/{ARTICLES_SOURCE_URL_PATH}/re00dx%03d.htm"
  )
  ARTICLES_SOURCE_ENCODING = "windows-1252"
  ARTICLES_LOCAL_FOLDER = "./articles"
  ARTICLES_LOCAL_ENCODING = "utf-8"
  ARTICLES_FORCE_OVERWRITE = True
  REQUEST_TIMEOUT_SECONDS = 10
  EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
  MISTRAL_MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
  CHUNK_SIZE = 200  # in words
  CHUNK_OVERLAP = 40  # in words
  FAISS_K_PROTOTYPE = 5  # number of nearest neighbors to retrieve
