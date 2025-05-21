#!/usr/bin/env python

"""This module downloads all articles from resistenze.org site."""

import os
import re
import sys
import glob
import html
import html5lib
import yaml
import numpy as np
import pickle
import faiss
import requests
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin #, urlparse
from bs4 import BeautifulSoup
from requests_file import FileAdapter # DEV ONLY
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import data_dump, data_load
from config import Config


# Setup logging
ENV = os.getenv("ENV")
LOG_LEVEL = logging.DEBUG if not ENV or ENV == "DEVEL" else logging.WARNING
logging.basicConfig(
  level = LOG_LEVEL,
  format = "🔵 %(asctime)s - %(levelname)s - %(message)s"
  #format = "🔵 %(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d %(message)s"
)
logger = logging.getLogger(__name__) # __name__ gives module's dotted path (e.g. "myapp.utils")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR) # Suppress INFO messages from SentenceTransformers
logger.info(f"Env is {ENV}")

# Global State
status = {}
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

# Force yaml dump a literal multiline string representation in "|" style
def literal_str_representer(dumper, data):
  if '\n' in data: # check if multiline
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style = "|")
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.add_representer(str, literal_str_representer)

def download_all_articles(n):
  """Download all articles in a number."""

  while True:
    n += 1
    target_url = Config.ARTICLES_SOURCE_URL_PATTERN % n
    html, base_url = download_articles_index(target_url) # Download the index page for the current article number
    if html:
      links = parse_html(html, base_url)
      count = len(links)
      logger.info(f"Number: {n}, links: {count}")
      # if n >= 400: # TODO: DEBUG ONLY!
      #   break
      if count > 0:
        download_links(links, n)
      else:
        n = n - 1 # we ignore the last article, "in preparazione" ...
        break
    else:
      break
  #status["last_article_number"] = n
  #data_dump(status)
  logger.info(f"Finished downloading {n} articles from {Config.ARTICLES_SOURCE_URL_BASE}")
  return n

def download_articles_index(url):
  """Download the HTML content of all articles in a number."""
  try:
    os.makedirs(Config.ARTICLES_LOCAL_FOLDER, exist_ok = True)
  except OSError as e:
    logger.error(e)
    return None, url

  try:
    response = requests.get(url, timeout = Config.REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    html_content = response.text
    return html_content, url
  except requests.RequestException as e:
    if response.status_code != 404: # 404 is expected for the last page
      logger.error(e)
    return None, url

def parse_html(html, base_url):
  """Parse the HTML content and extract links to articles."""
  soup = BeautifulSoup(html, 'html.parser')
  links = set()
  for tag in soup.find_all('a', href = True):
    href = tag['href']
    full_url = urljoin(base_url, href)
    if full_url.startswith('http'):
      # Only download links from the same domain
      if full_url.startswith(Config.ARTICLES_SOURCE_URL_BASE):
        # Only download links that do match the expected frame pattern
        #if not re.compile(r'sito/re00.*\.htm$').search(full_url):
        if not re.compile(Config.ARTICLES_SOURCE_NEWS_PATTERN).search(full_url):
          links.add(full_url)
  return links

def download_links(links, n):
  """Download the articles from the extracted links."""
  for link in links:
    try:
      # Get path and filename from the link
      categories_and_filename = link.replace(f"{Config.ARTICLES_SOURCE_URL_BASE}/{Config.ARTICLES_SOURCE_URL_PATH}/", '')
      categories = categories_and_filename.split('/') # Get the categories from the link
      filename = categories.pop() # Build the filename from the categories

      filename = f"{"_".join(categories)}_{filename}"
      basename, ext = os.path.splitext(filename)
      filename = basename + Config.ARTICLES_LOCAL_EXTENSION # Change the extension
      filepath = os.path.join(Config.ARTICLES_LOCAL_FOLDER, filename)

      # Write file locally

      # Check if article exists, or if we have to force overwrite
      exists = os.path.exists(filepath)
      if not exists or Config.ARTICLES_FORCE_OVERWRITE:
        try:
          # Download article
          # DEV ONLY: READ FILE FROM FILE SYSTEM OR FROM THE INTERNET ###########################################
          url_or_filename = None
          if (Config.ARTICLES_SOURCE_FILE):
            session = requests.Session()
            session.mount('file://', FileAdapter())
            url_or_filename = f"file://{os.getcwd()}/articles_html/{os.path.basename(link)}"
            response = session.get(url_or_filename, timeout = Config.REQUEST_TIMEOUT_SECONDS)
          else:
            url_or_filename = link
            response = requests.get(url_or_filename, timeout = Config.REQUEST_TIMEOUT_SECONDS)
          #######################################################################################################
          response.raise_for_status() # Check if the response is valid, raise exception if not
        except requests.RequestException as e:
          logger.error(e)
          continue

        article = convert_to_text(response, n)
        try: # Dump article to disk
          with open(filepath, 'w', encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
            yaml.dump(article, f, allow_unicode = True, sort_keys = False)
        except Exception as e:
          logger.error(e)
          continue

        if not exists:
          logger.info(f"Downloaded: {link}")
        else:
          logger.info(f"Downloaded and overwritten {link}")
      else:
        logger.info(f"Skipped: {link}")
    except Exception as e:
      logger.error(e)

def convert_to_text(response, n):
  # Browsers treat charset=iso-8859-1 encoding as Windows-1252.
  # This behavior is a de facto standard in browsers to maintain compatibility with legacy content.
  if response.encoding == 'ISO-8859-1':
    response.encoding = 'windows-1252'

  html = response.text

  # Preprocess html to solve quirks in the source
  html = html.replace("’", "'") # replace ’ with ' (in this site the typographic close quote is wrongly used for the regular 
  html = html.replace("–", "'") # replace – with -
  html = html.replace(r'\302\240', r'') # Remove non-breaking spaces

  # Try to match header fields
  # TODO: article = Article()
  title = ""
  category = ""
  topics = ""
  date_publishing_resistenze = ""
  date_publishing_source = ""
  number = ""
  author = ""
  source = ""
  translator = ""
  contents = ""
  
  # Match title
  match_title = re.search(f'<title>(.*?)</title>', html, flags = re.IGNORECASE | re.DOTALL) # match title
  if match_title:
    title = match_title.group(1)

  # Match category, number and date_publishing_resistenze
  match_header = re.search(
    f'<a href="https?://{Config.ARTICLES_SOURCE_URL_DOMAIN}/">{Config.ARTICLES_SOURCE_URL_DOMAIN}</a>[\\s\\-]*(.*?)<br\\s*/?>',
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_header:
    header = match_header.group(1)
    html = html.replace(match_header.group(0), "") # Remove header from html
    parts_separator = " - "
    header = header.strip().replace("<i>", "").replace("</i>", "")
    parts = [p.strip() for p in header.split(parts_separator)]
      
    if len(parts) < 3: 
      # Input string must have at least 3 groups separated by '{parts_separator}'
      # Ignore error and keep the original category, number, date
      pass
    else:
      category = parts[0].strip()
      number = parts[-1].strip()
      number = re.sub(r"n\.\s?", "", number) # Remove "n." from number
      date = parts[-2].strip()
      from datetime import datetime
      try:
        date = datetime.strptime(date, "%d-%m-%y").strftime("%Y-%m-%d")
      except ValueError:
        logger.debug(f"Unforeseen date format: {date}")
        # Ignore error and keep the original date
        pass
      date_publishing_resistenze = date

    topics = [s.strip() for s in parts[1:-2]] # All groups between first and last two
    
  # Remove unwanted parts of the html
  html = html.replace(r"&nbsp;", " ") # Remove non-breaking spaces

  # Remove repeated title in body
  html = re.sub(r"<div><font size=\"4\">.*?<\/font><\/div>", "", html, flags = re.IGNORECASE | re.DOTALL)

  # Match author
  author_pattern = re.compile(
    r"""<div><font\s+size=["']3["'].*?>\s*<br\s*/?>\s*     # anchor start
        ([^|<]+?)\s*\|\s*                               # capture author (text before pipe)
        (.+?)\s*                                        # capture source (including tags)
        <br\s*/?>                                       # ending <br />
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL
  )
  match_author_and_source = author_pattern.search(html)
  if match_author_and_source:
    author = match_author_and_source.group(1)
    source = match_author_and_source.group(2) if match_author_and_source.group(2) else ""
    # print("Author:", author)
    # print("Source:", source)
    html = html.replace(match_author_and_source.group(0), "") # Remove author from html

  # Match date_publishing_source in format dd/mm/yyyy
  match_date = re.search(
    r"\s*(\d\d)/(\d\d)/(\d\d\d\d)\s*<br\s*/?>",
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_date:
    date1 = match_date.group(1)
    date2 = match_date.group(2)
    date3 = match_date.group(3)
    date_publishing_source = f"{date3}-{date2}-{date1}"
    html = html.replace(match_date.group(0), "") # Remove date from html

  # Match date_publishing_source in format yyyy
  match_date = re.search(
    r"\s*(\d\d\d\d)\s*<br\s*/?>",
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_date:
    date1 = match_date.group(1)
    date_publishing_source = f"{date1}"
    html = html.replace(match_date.group(0), "") # Remove date from html

  # Match date_publishing_source in format "italian month name"
  match_date = re.search(
    r"\s*(Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre)\s*<br\s*/?>",
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_date:
    date1 = match_date.group(1)
    date_publishing_source = f"{date1}"
    html = html.replace(match_date.group(0), "") # Remove date from html

  # Match translator
  match_translator = re.search(
    r"(?:Traduzione|Trascrizione)\s*(?:di\s+(?:.*?)?)?\s*(?:per\s*<a .*?>.*?</a>)?\s*(?:a\s*cura\s*(?:di|del|della|dello|degli|dei)\s*(.*?))<br\s*/?>",
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_translator:
    translator = match_translator.group(1)
    html = html.replace(match_translator.group(0), "") # Remove translator from html

  html = re.sub(r"<table.*(sostieni resistenze|fai una donazione).*</table>", "", html, flags = re.IGNORECASE | re.DOTALL) # Remove footer parts

  # Parse html to extract only text
  title = html_to_text(title)
  category = html_to_text(category)
  topics = topics
  date_publishing_resistenze = html_to_text(date_publishing_resistenze)
  number = html_to_text(number)
  author = html_to_text(author)
  source = html_to_text(source)
  date_publishing_source = html_to_text(date_publishing_source)
  translator = html_to_text(translator)
  text = html_to_text(html, tag_name = "body")

  if not int(number) == int(n):
    logger.warning(f"Real number is {n}, extracted number is {number}")
  article = {
    "url": response.url,
    "title": title,
    "category": category,
    "topics": topics,
    "date_publishing_resistenze": date_publishing_resistenze,
    "number": n, #number,
    "author": author,
    "source": source,
    "date_publishing_source": date_publishing_source,
    "translator": translator,
    "contents": text,
  }

  # DEBUG ONLY #################################################
  if ENV == "DEVEL":
    missing = 0
    if not number: missing += 1
    if not title: missing += 1
    if not author: missing += 1
    if not date_publishing_resistenze: missing += 1
    logger.info(f"💡 - n° {number or '?'}, title: {title}, author: {author or '?'}, date: {date_publishing_resistenze or '?'}, missing fields: {missing}")
  ##############################################################

  return article

def html_to_text(html, tag_name = None):
  soup = BeautifulSoup(html, "html5lib") # Use html5lib parser for better compatibility with HTML5
  tag = soup.find(tag_name) if tag_name else soup

  if tag is None:
    return ""

  for br in tag.find_all("br"): # Replace <br> tags with \n
    br.replace_with("\n")

  text = tag.get_text(separator = "\n", strip = True)
  return text

def data_sync():
  """Sync data: faiss index, metadata and texts."""
  global index, texts, metadata

  #last_updated = get_article_last_updated_timestamp()
  articles = []
  new_files = []

  logger.info(f"Syncing FAISS index, metadata and texts")

  #logger.info(f"Loading articles...")
  pattern = os.path.join(Config.ARTICLES_LOCAL_FOLDER, f"*{Config.ARTICLES_LOCAL_EXTENSION}")
  files = glob.glob(pattern)
  with tqdm(total = len(files)) as pbar:
    #for file_path in glob.glob(os.path.join(Config.ARTICLES_LOCAL_FOLDER, f"*{Config.ARTICLES_LOCAL_EXTENSION}")):
    for file_path in files:
      try:
        #if mtime > last_updated:
        with open(file_path, 'r', encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
          article = yaml.safe_load(f)
          if article:
            article['__filepath'] = file_path
            article['__mtime'] = datetime.fromtimestamp(os.path.getmtime(file_path))
            articles.append(article)
            new_files.append(file_path)
          else:
            logger.error(f"Void article found from {file_path}")
            raise ValueError(f"Void article found from {file_path}")
          pbar.set_description(f"Loaded article from file {Path(file_path).name.ljust(36)}")
          pbar.update(1)
      except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        raise

  if not articles:
    logger.warning("No new articles to index")
    return 0

  # Initialize or load existing data
  if all(os.path.exists(path) for path in [
    Config.FAISS_INDEX_PATH,
    Config.METADATA_PATH,
    Config.TEXTS_PATH
  ]):
    logger.info("Loading existing data...")
    index = faiss.read_index(Config.FAISS_INDEX_PATH)
    with open(Config.METADATA_PATH, 'rb') as f:
      metadata = pickle.load(f)
    with open(Config.TEXTS_PATH, 'rb') as f:
      texts = pickle.load(f)
  else:
    logger.info("Creating new data...")
    metadata = []
    texts = []
    dim = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)

  # Track new data
  new_articles = 0
  new_embeddings = []
  new_texts = []
  new_metadata = []

  with tqdm(total = len(articles), desc = "Indexing articles") as pbar:
    for n, article in enumerate(articles):
      try:
        chunks = chunk_text(article['contents'])
        #filename = os.path.basename(article.get('filename') or article.get('url') or 'unknown')
        filename = os.path.basename(article.get('__filepath'))
        #logger.debug(f"Indexing [{n}/{len(articles)}]: {filename} ({len(chunks)} chunks)")
        new_article = False

        for chunk in chunks:
          # Only process chunks we haven't seen before in texts chunks
          if chunk not in texts:
            new_article = True
            new_texts.append(chunk)
            new_metadata.append({
              'url': article.get('url', ''),
              'title': article.get('title', ''),
              'author': article.get('author', ''),
              'date': article.get('date_publishing_resistenze', ''),
              'source_path': article['__filepath'],
              'last_updated': article['__mtime'].isoformat()
            })
            new_embeddings.append(embedding_model.encode(chunk))
        if new_article:
          new_articles += 1
          pbar.set_description(f"Added chunks from article {n}")
        else:
          pbar.set_description(f"Skipped chunks from article {n} because already in texts")
        pbar.update(1)

      except Exception as e:
        logger.error(f"Error chunking article number {article["number"]} in file {article.get('__filepath')}: {e}")
        raise

  if new_embeddings:
    # Add new data to existing structures
    texts.extend(new_texts)
    metadata.extend(new_metadata)
    embeddings_array = np.array(new_embeddings).astype('float32')
    index.add(embeddings_array)

    
    faiss.write_index(index, Config.FAISS_INDEX_PATH) # Save faiss index
    with open(Config.METADATA_PATH, 'wb') as f: # Save metadata
      pickle.dump(metadata, f)
    with open(Config.TEXTS_PATH, 'wb') as f: # Save texts
      pickle.dump(texts, f)

    logger.info(f"Indexed {len(new_embeddings)} new chunks from {len(new_files)} files, in {new_articles} articles.")
  else:
    logger.info("No new articles to index")
  return len(new_embeddings)

def chunk_text(text, size = Config.CHUNK_SIZE, overlap = Config.CHUNK_OVERLAP):
  words = text.split()
  chunks = []
  for i in range(0, len(words), size - overlap):
    chunk = ' '.join(words[i:i+size])
    if chunk:
      chunks.append(chunk)
  return chunks

if __name__ == "__main__":
  try:
    status = data_load()
    try:
      n = status["last_article_number"]
    except KeyError:
      n = 0
    n = download_all_articles(n)
    status["last_article_number"] = n
    data_dump(status)
    data_sync()
    exit(0)

  except KeyboardInterrupt:
    logger.warning('Interrupted')
    os._exit(130)
