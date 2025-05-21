#!/usr/bin/env python

from tqdm import trange
import time

articles = [None] * 100
tot = len(articles)
pbar = trange(tot)
for n in pbar:
  pbar.set_description(f"Added chunks from article {n:5}/{tot} to index")
  time.sleep(0.5)
