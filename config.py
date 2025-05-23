""" configuration class """

class Config:
  """ configuration class values """

  ARTICLES_SOURCE_FILE = False # DEV ONLY - Request articles from disk instead of the web
  ARTICLES_MAX = 1024 # DEV ONLY - Limit articles to be indexed (0: no limit)
  ARTICLES_FORCE_OVERWRITE = True

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

  DATA_PATH = "./data"
  STATUS_PATH = f"{DATA_PATH}/status.yaml"
  ARTICLES_PATH = f"{DATA_PATH}/articles.pkl"
  FAISS_INDEX_PATH = f"{DATA_PATH}/index.faiss"
  METADATA_PATH = f"{DATA_PATH}/metadata.pkl"
  TEXTS_PATH = f"{DATA_PATH}/texts.pkl"
  REQUEST_TIMEOUT_SECONDS = 10
  #EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
  EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2" # Better quality
  #MISTRAL_MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf" # Older version
  #MISTRAL_MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf" # Newer version, best quality
  MISTRAL_MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q3_K_S.gguf" # More compressed version, medium quality
  #MISTRAL_MODEL_PATH = "./models/phi-2.Q5_K_S.gguf" # Simpler version, low quality, maximum speed
  MAX_TOKENS_IN_CONTEXT = 4000 # Maximum number of tokens in the context
  CHUNK_SIZE = 200 # Number of tokens per chunk
  CHUNK_OVERLAP = 40 # Number of overlapping tokens between chunks
  FAISS_K_PROTOTYPE = 3 # Number of nearest neighbors to retrieve
  DISTANCE_THRESHOLD = 10.0 #0.85 # For L2 distance (default in FAISS IndexFlatL2) for similarity search
  MAX_TOKENS_IN_ANSWER = 250 # Maximum number of tokens in the answer (150 forces brevity)