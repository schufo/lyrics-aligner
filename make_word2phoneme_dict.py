"""
Reads the .dict output file of the CMU pronunciation dictionary
and creates and saves a Python dictionary: dict[word] = phoneme.
"""

import argparse
import pickle
import os

import Levenshtein


def create_word2phoneme_file(input_file_path, output_path):
    # read .txt-file with words and phonemes, make dict word2phonemes
    words2phonemes_file = open(input_file_path)
    lines = words2phonemes_file.readlines()

    word2phonemes = {}

    for line in lines:
        line = line.replace('\n', '').split('\t')

        word = line[0].lower().replace('’', "'")
        phonemes = line[1]
        word2phonemes[word] = phonemes

    # save dict
    pickle_out = open(output_path, "wb")
    pickle.dump(word2phonemes, pickle_out)
    pickle_out.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='word2phoneme')
#     parser.add_argument('--dataset-name', type=str, default='dataset1')
#     args = parser.parse_args()
#     output_path = "files/{}_word2phonemes.pickle".format(args.dataset_name)
#     input_file_path = 'files/{}_word2phonemes.txt'.format(args.dataset_name)
#     create_word2phoneme_file(input_file_path, output_path)


if __name__ == "__main__":
    # Check for all words that are not present in the word2phoneme dict global
    # Load the global word->phoneme dict
    path = "dataset/word2phonemeglobal.pickle"
    with open(path, "rb") as handle:
        w2ph_dict = pickle.load(handle)
    # Get all the lyrics files
    lyrics_files = ["dataset/casta_diva/text/song.txt"]

    # Iterate over all lyrics file to check if all words are present in the dict or not.
    for text_file_path in lyrics_files:
        with open(text_file_path, 'r') as text_file:
            words = text_file.read().split()

        for word in words:
            if word in w2ph_dict:
                phoneme_list = w2ph_dict[word]
            else:
                print(f"Unable to find the match for word: {word} in Aria: {text_file_path.split('/')[-3]}")
                # Find the closest words in the word2ph dictionary.
                # Compute distances from target_key to each key in dictionary
                distances = [(key, Levenshtein.distance(word, key)) for key in w2ph_dict.keys()]
                # Sort by distance
                sorted_distances = sorted(distances, key=lambda x: x[1])
                # Print those words.
                print("These are the closest 5 matches")
                for i in range(5):
                    print(sorted_distances[i][0] + " -> " + " ".join(w2ph_dict[sorted_distances[i][0]]))
                # Ask for the phoneme string from the user
                phoneme_str = input(f'Enter the phoneme string for {word}: \n')

                # Add the word to the dictionary
                phoneme_list = phoneme_str.split(" ")
                w2ph_dict[word] = phoneme_list
    print("Added all phonemes into the dict.\n")
    save = input("Press 1 to save this into the global word2ph dict, press any other key to NOT save.")
    if save == '1':
        with open(path, 'wb') as file:
            pickle.dump(w2ph_dict, file)
