"""
Reads the .dict output file of the CMU pronunciation dictionary
and creates and saves a Python dictionary: dict[word] = phoneme.
"""

import argparse
import pickle
import os

parser = argparse.ArgumentParser(description='word2phoneme')
parser.add_argument('--dataset-name', type=str, default='dataset1')
args = parser.parse_args()

# read .txt-file with words and phonemes, make dict word2phonemes
words2phonemes_file = open('files/{}_word2phonemes.txt'.format(args.dataset_name))
lines = words2phonemes_file.readlines()

word2phonemes = {}

for line in lines:
    line = line.replace('\n', '').split('\t')

    word = line[0].lower().replace('’', "'")
    phonemes = line[1]
    word2phonemes[word] = phonemes

# save dict
pickle_out = open("files/{}_word2phonemes.pickle".format(args.dataset_name), "wb")
pickle.dump(word2phonemes, pickle_out)
pickle_out.close()