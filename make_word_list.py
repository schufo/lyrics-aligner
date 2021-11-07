"""
Generates a .txt-file with all unique words in a dataset. This .txt/file
can be used to translate words into phoneme sequences with the
CMU pronunciation dictionary (http://www.speech.cs.cmu.edu/tools/lextool.html)
"""

import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Word list generation')
parser.add_argument('lyrics', type=str, help='path to a directory with lyrics stored in .txt-files')
parser.add_argument('--dataset-name', type=str, default='dataset1')
args = parser.parse_args()

unique_words = set()

lyrics_files = glob.glob(os.path.join(args.lyrics, '*.txt'))
assert len(lyrics_files) > 0, 'No .txt-files found in {}'.format(args.lyrics)

# go through .txt-files and save unique words in the unique_words set
for file in lyrics_files:

    with open(file) as word_file:
        lines = word_file.readlines()
        for line in lines:
            line = line.lower().replace('\n', '').replace('’', "'")
            clean_line = ''.join(c for c in line if c.isalnum() or c in ["'", ' '])

            if clean_line == ' ' or clean_line == '': continue
            words = clean_line.split(' ')
            for word in words:
                unique_words.add(word)

unique_words.remove('')

# create .txt-file
word_file_path = 'files/{}_word_list.txt'.format(args.dataset_name)
assert not os.path.isfile(word_file_path), 'file {} exists already. Delete or choose different' \
                                           ' file to avoid appending to existing file'.format(word_file_path)

# write words in .txt-file
words_file = open(word_file_path, 'a')
for word in sorted(unique_words):
    words_file.write(word + '\n')
words_file.close()

# create empty .txt-file which will contain the output of the CMU pronuciation dictionary.
empty_file_path =  'files/{}_word2phonemes.txt'.format(args.dataset_name)
empty_file = open(empty_file_path, 'a')
empty_file.write('')
empty_file.close()
