import time

import numpy as np
#import tensorflow as tf

import utils

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile

dataset_folder_path = 'data'
dataset_filename = 'text8.zip'
dataset_name = 'Text8 Dataset'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile(dataset_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
        urlretrieve(
            'http://mattmahoney.net/dc/text8.zip',
            dataset_filename,
            pbar.hook)

if not isdir(dataset_folder_path):
    with zipfile.ZipFile(dataset_filename) as zip_ref:
        zip_ref.extractall(dataset_folder_path)
        
with open('data/text8') as f:
    text = f.read()

words = utils.preprocess(text)
print(words[:30])

totalWords = len(words)
uniqueWords = len(set(words))
print("Total words: {}".format(totalWords))
print("Unique words: {}".format(uniqueWords))

#vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
#int_words = [vocab_to_int[word] for word in words]

from collections import Counter
word_counts = Counter(words)
t =  1e-5   
frequencies = {word: (count / totalWords) for word, count in word_counts.items()}
prob_drop = {word: 1 - np.sqrt(t/frequencies[word]) for word in word_counts}
print(prob_drop['the'])
pass