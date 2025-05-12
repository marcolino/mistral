#!/usr/bin/env python

"""This module downloads all articles from resistenze.org site."""

import os
import re
import sys
import html
from urllib.parse import urljoin #, urlparse
from bs4 import BeautifulSoup
import requests
from config import Config

first_article_number = 930 # first article number to download

def download_html(url):
  """Download the HTML content of a page."""
  try:
    os.makedirs(Config.ARTICLES_LOCAL_FOLDER, exist_ok = True)
  except OSError as e:
    print(e)
    return None, url

  try:
    response = requests.get(url, timeout = Config.REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    html_content = response.text
    return html_content, url
  except requests.RequestException as e:
    if response.status_code != 404: # 404 is expected for the last page
      print(e)
    return None, url

def parse_html(html, base_url):
  """Parse the HTML content and extract links to articles."""
  soup = BeautifulSoup(html, 'html.parser')
  links = set()
  for tag in soup.find_all('a', href = True):
    href = tag['href']
    full_url = urljoin(base_url, href)
    if full_url.startswith('http'):
      # only download links from the same domain
      if full_url.startswith(Config.ARTICLES_SOURCE_URL_BASE):
        # only download links that do not match the right frame pattern
        if not re.compile(r'sito/re00.*\.htm$').search(full_url):
          links.add(full_url)
  return links

def download_links(links):
  """Download the articles from the extracted links."""
  for link in links:
    try:
      response = requests.get(link, timeout = Config.REQUEST_TIMEOUT_SECONDS)
      response.raise_for_status() # Check if the response is valid, raise exception if not

      title, category, topics, date, number, text = convert_to_text(response)
      
      # Get path and filename from the link
      categories_and_filename = link.replace(f"{Config.ARTICLES_SOURCE_URL_BASE}/{Config.ARTICLES_SOURCE_URL_PATH}/", '')
      categories = categories_and_filename.split('/') # Get the categories from the link
      filename = categories.pop() # Extract the filename from the categories

      filename = f"{"_".join(categories)}_{filename}"
      filepath = os.path.join(Config.ARTICLES_LOCAL_FOLDER, filename)

      # Write file locally
      if not os.path.exists(filepath) or Config.ARTICLES_FORCE_OVERWRITE: # Avoid overwriting if the filename already exists
        with open(filepath, 'w', encoding = Config.ARTICLES_LOCAL_ENCODING) as f:
          f.write(f"""
title: {title}
category: {category}
topics: {", ".join(topics)}
date_publishing: {date}
number: {number}
contents: |
{text}
""")
        print(f"Downloaded: {link}")
      else:
        print(f"Skipped: {link}")
    except requests.RequestException as e:
      print(e)

def convert_to_text(response):
  # Browsers treat charset=iso-8859-1 encoding as Windows-1252.
  # This behavior is a de facto standard in browsers to maintain compatibility with legacy content.
  if response.encoding == 'ISO-8859-1':
    response.encoding = 'windows-1252'

  html = response.text

  # Preprocess html to solve quirks in the source
  html = re.sub(r"(l)’(\\w)", r"\1'\1", html, flags = re.IGNORECASE) # replace l’ (followed by alphanum, ignoring case) with l'
  html = html.replace(r'\302\240', r'') # Remove non-breaking spaces

  # Try to match header fields
  title = None
  category = None
  topic = None
  date_publishing = None
  number = None

  match_title = re.search(f'<title>(.*?)</title>', html, flags = re.IGNORECASE | re.DOTALL) # match title
  if match_title:
    title = match_title.group(1)

  match_header = re.search(
    f'<a href="https?://{Config.ARTICLES_SOURCE_URL_DOMAIN}/">{Config.ARTICLES_SOURCE_URL_DOMAIN}</a>[\\s\\-]*(.*?)<br>',
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_header:
    header = match_header.group(1)
    # print(f"Header: {header}")
    html = html.replace(header, "XXXXXXXXXXX") # Remove header from html
    parts_separator = " - "
    header = header.strip().replace("<i>", "").replace("</i>", "")
    parts = [p.strip() for p in header.split(parts_separator)]
      
    if len(parts) < 3:
      raise ValueError(f"Input string must have at least 3 groups separated by '{parts_separator}'")
    
    category = parts[0].strip()
    number = parts[-1].strip()
    date = parts[-2].strip()
    from datetime import datetime
    date = datetime.strptime(date, "%d-%m-%y").strftime("%Y-%m-%d")
    # topics = parts[1:-2] # All groups between first and last two
    # topics = [s.strip() for s in topics]
    topics = [s.strip() for s in parts[1:-2]] # All groups between first and last two
    
  # print(f"category: {category}")
  # print(f"topics: {topics}")
  # print(f"date: {date}")
  # print(f"number: {number}")


  # Remove unwanted parts of the html
  html = re.sub(r"<div><font size=\"4\">.*?</font></div>", "", html, flags = re.IGNORECASE | re.DOTALL) # Remove title in body

  match_author = re.search(
    r"<br />\s*(.*?)\s*(\| (.*?))?<br />", # "|" is optional !!!
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_author:
    author1 = match_author.group(1)
    author2 = match_author.group(3)
    print(f"Author: {author1} | {author2}")

  match_translator = re.search(
    r"Traduzione per (<a .*?>.*?</a>)? a cura (di|del|della|dello|degli|dei) (.*?)<br />",
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_translator:
    translator = match_translator.group(3)
    print(f"Translator: {translator}")

  match_date = re.search(
    r"\s*(\d\d)/(\d\d)/(\d\d\d\d)\s*<br />",
    html,
    flags = re.IGNORECASE | re.DOTALL
  )
  if match_date:
    date1 = match_date.group(1)
    date2 = match_date.group(2)
    date3 = match_date.group(3)
    print(f"Date: {date3}-{date2}-{date1}")

  """
  <div><font size="4"><strong>Cosa ci aspetta?</strong></font></div>
  <div><font size="3"><br />
  Greg Godels | <a href="https://zzs-blg.blogspot.com/2025/04/whats-next.html">zzs-blg.blogspot.com</a><br />
  Traduzione per <a href="http://www.resistenze.org/">Resistenze.org</a> a cura del Centro di Cultura e Documentazione Popolare<br />
  <br />
  23/04/2025<br />
  """

  html = re.sub(r"<table.*sostieni resistenze.*</table>", "", html, flags = re.IGNORECASE | re.DOTALL) # Remove footer parts

  # Parse html with BeautifulSoup to extract only text
  soup = BeautifulSoup(html, "html.parser")
  text = soup.get_text(separator = " ", strip = True)
  
  print("\n")
  return title, category, topics, date, number, text

def convert_to_text_OLD(text):
  """Clean up the HTML content and remove unwanted characters."""
  text_cleaned = html.unescape(text)
  return text_cleaned
  
def main():
  """Main function to download all articles."""
  n = first_article_number
  while True:
    target_url = Config.ARTICLES_SOURCE_URL_PATTERN % n
    html, base_url = download_html(target_url) # Download the index page for the current article number
    if html:
      links = parse_html(html, base_url)
      print(f"Number: {n}, links: {len(links)}")
      download_links(links)
      n += 1
    else:
      break
  print(f"Finished downloading all articles from {Config.ARTICLES_SOURCE_URL_BASE}")

if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print('Interrupted')
    try:
      sys.exit(130)
    except SystemExit:
      os._exit(130)
