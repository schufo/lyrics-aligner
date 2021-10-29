import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Word list generation')
parser.add_argument('-lyrics-path', type=str)
parser.add_argument('--dataset-name', type=str, default='dataset1')
args, _ = parser.parse_known_args()

unique_words = set()

lyrics_files = glob.glob(args.lyrics_path + '*raw.txt')

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

print("Unique words:", sorted(unique_words))
print("Number of unique words:", len(unique_words))


word_file_path = 'files/{}_word_list.txt'.format(args.dataset_name)
if os.path.isfile(word_file_path):
    print('file {} exists already. Delete or choose different file to avoid appending to existing file'.format(word_file_path))
    quit()

words_file = open(word_file_path, 'a')

for word in sorted(unique_words):
    words_file.write(word + '\n')