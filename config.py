""" configuration class """

class Config:
  """ configuration class values """

  ARTICLES_SOURCE_FILE = False # DEV ONLY - request articles from disk instead of the web

  ARTICLES_SOURCE_URL_DOMAIN = "www.resistenze.org"
  ARTICLES_SOURCE_URL_BASE = f"https://{ARTICLES_SOURCE_URL_DOMAIN}"
  ARTICLES_SOURCE_URL_PATH = "sito"
  ARTICLES_SOURCE_URL_PATTERN = (
    f"{ARTICLES_SOURCE_URL_BASE}/{ARTICLES_SOURCE_URL_PATH}/re00dx%03d.htm"
  )
  ARTICLES_SOURCE_NEWS_PATTERN = r"sito/re00.*\.htm$"
  ARTICLES_SOURCE_ENCODING = "windows-1252"
  ARTICLES_LOCAL_FOLDER = "./articles"
  ARTICLES_LOCAL_ENCODING = "utf-8"
  ARTICLES_LOCAL_EXTENSION = ".txt"
  ARTICLES_FORCE_OVERWRITE = True

  DATA_PATH = "./data"
  STATUS_PATH = f"{DATA_PATH}/status.yaml"
  FAISS_INDEX_LAST_SYNC_DATETIME_PATH = f"{DATA_PATH}/last_sync.datetime"
  FAISS_INDEX_PATH = f"{DATA_PATH}/index.faiss"
  METADATA_PATH = f"{DATA_PATH}/metadata.pkl"
  TEXTS_PATH = f"{DATA_PATH}/texts.pkl"

  REQUEST_TIMEOUT_SECONDS = 10
  EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
  MISTRAL_MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
  CHUNK_SIZE = 200 # Number of words per chunk
  CHUNK_OVERLAP = 40 # Number of overlapping words between chunks
  FAISS_K_PROTOTYPE = 3 # Number of nearest neighbors to retrieve
  DISTANCE_THRESHOLD = 0.85 # For L2 distance (default in FAISS IndexFlatL2) for similarity search
