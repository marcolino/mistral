#!/usr/bin/env python

import os
import codecs

path = "articles"
encodings = ['utf-8'] #,'windows-1250', 'windows-1252']

n = 1
for name in os.listdir(path):
  pathname = os.path.join(path, name)
  if os.path.isfile(pathname):
    for e in encodings:
      try:
        fh = codecs.open(pathname, 'r', encoding = e)
        fh.readlines()
        fh.seek(0)
      except UnicodeDecodeError:
        print(f'got unicode error with {e}, trying different encoding')
      else:
        print(f'{n} - opening the file {pathname} with encoding: {e}')
        break  
  n = n + 1
