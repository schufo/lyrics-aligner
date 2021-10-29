import argparse
import pickle
import os

parser = argparse.ArgumentParser(description='word2phoneme')
parser.add_argument('--dataset-name', type=str, default='dataset1')
args, _ = parser.parse_known_args()



# read text file with words and phonemes, make dict musdb_word2cmu_phoneme
words2cmu_phonemes_file = open('files/jamendo_words_cmu_phonemes.txt')
lines = words2cmu_phonemes_file.readlines()

jamendo_word2cmu_phoneme = {}

for line in lines:
    line = line.replace('\n', '').split('\t')

    word = line[0].lower().replace('’', "'")
    phonemes = line[1]
    jamendo_word2cmu_phoneme[word] = phonemes

print(jamendo_word2cmu_phoneme)
# save hansen_word2cmu_phoneme
pickle_out = open(os.path.join('../Datasets/jamendolyrics', "jamendo_word2cmu_phoneme.pickle"), "wb")
pickle.dump(jamendo_word2cmu_phoneme, pickle_out)
pickle_out.close()