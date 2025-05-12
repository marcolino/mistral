from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import os
import yaml
import shutil
import uuid
from sentence_transformers import SentenceTransformer
import faiss
import llama_cpp
import glob
from config import Config

app = FastAPI()

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

def save_article(article: Article):
    path = os.path.join(Config.DATA_DIR, article.filename)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(article.dict(), f, allow_unicode=True)

def load_articles():
    articles = []
    for file_path in glob.glob(os.path.join(Config.DATA_DIR, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            articles.append(data)
    return articles

def chunk_text(text, size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_faiss_index():
    global index, texts, metadata
    articles = load_articles()
    texts = []
    metadata = []
    embeddings = []

    for article in articles:
        chunks = chunk_text(article['contents'])
        for chunk in chunks:
            texts.append(chunk)
            metadata.append(article)
            emb = embedding_model.encode(chunk)
            embeddings.append(emb)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

# --- API Routes ---

@app.post("/import_articles")
def import_articles(article: Article):
    save_article(article)
    return {"status": "imported", "filename": article.filename}

@app.post("/build_index")
def build_index():
    build_faiss_index()
    return {"status": "index built", "items": len(texts)}

@app.post("/query")
def query_articles(req: QueryRequest):
    if index is None:
        return {"error": "Index not built yet"}

    query_vec = embedding_model.encode(req.query)
    D, I = index.search(np.array([query_vec]), Config.FAISS_K_PROTOTYPE)

    context_chunks = [texts[i] for i in I[0]]
    context = "\n---\n".join(context_chunks)
    prompt = f"Contesto:\n{context}\n\nDomanda:\n{req.query}\n\nRispondi in modo informato basandoti sul contesto."

    llm = llama_cpp.Llama(model_path=Config.MISTRAL_MODEL_PATH, n_ctx=4096, n_threads=6)
    response = llm(prompt)

    return {"response": response['choices'][0]['text'].strip()}

@app.get("/favicon.ico")
def favicon():
    return FileResponse("assets/images/favicon.png")

@app.get("/", response_class=HTMLResponse)
def web_ui():
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())