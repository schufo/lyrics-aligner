import os
from dataclasses import dataclass
from typing import List

import librosa
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle


def get_dataloader_obj(dataset_patt="dataset/", pickle_file="dataset/aria_dataset.pickle", batch_size=1, shuffle=True):
    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = AriaDataset(dataset_patt)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def words_to_phonemes(words, w2ph_dict):
    phonemes = []
    word_start_indexes = []
    word_end_indexes = []
    phonemes.append(">")
    start_idx = 1
    for word in words:
        if word in w2ph_dict:
            phoneme_list = w2ph_dict[word]
            phonemes.extend(phoneme_list)
            word_start_indexes.append(start_idx)
            word_end_indexes.append(start_idx+len(phoneme_list))
            start_idx = start_idx+len(phoneme_list) + 1
        else:
            raise KeyError(f"{word} is not present in the phonetic transcriptions")
        phonemes.append(">")
    return phonemes, word_start_indexes, word_end_indexes


@dataclass
class Aria:
    name: str
    audio: Tensor   # (timesteps, )
    phonemes: List
    start_time: List
    end_time: List
    word_start_indexes: List
    word_end_indexes: List


class AriaDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.X = []
        self.labels = []

        folders = os.listdir(self.path)
        for folder_name in folders:
            if folder_name == '.DS_Store':
                continue
            audio_file_path = os.path.join(self.path, folder_name, 'audio', 'song.mp3')
            text_file_path = os.path.join(self.path, folder_name, 'text', 'song.txt')
            labels_file_path = os.path.join(self.path, folder_name, 'labels.tsv')
            phoneme_dict_path = os.path.join(self.path, folder_name, 'word2phonemes.pickle')
            with open(phoneme_dict_path, 'rb') as f:
                w2ph_dict = pickle.load(f)
                w2ph_dict = {k: v.split() for k, v in w2ph_dict.items()}
            # Load audio file using librosa
            audio, _ = librosa.load(audio_file_path, sr=16000, mono=True)

            # Read text file as a list of words
            with open(text_file_path, 'r') as text_file:
                words = text_file.read().split()
                phonemes, word_start_indexes, word_end_indexes = words_to_phonemes(words, w2ph_dict)
            # TODO: Convert phonemes in to indexes using the phoneme2idx.pickle
            self.X.append({"audio": audio, "phonemes": phonemes})

            # Read timestamps from labels.tsv
            with open(labels_file_path, 'r') as labels_file:
                start_times, end_times = [], []
                lines = labels_file.readlines()[1:]  # Skip the first row (headers)
                for line in lines:
                    parts = line.strip().split('\t')
                    start_times.append(float(parts[1]))
                    end_times.append(float(parts[2]))

                self.labels.append({"start_time": start_times,
                                    "end_time": end_times,
                                    "start_indexes": word_start_indexes,
                                    "end_indexes": word_end_indexes,
                                    "name": folder_name})
            print(f"Processed {folder_name}...")

    def pickle_dataset(self):
        with open('/dataset/aria_dataset.pickle', 'wb') as file:
            pickle.dump(self, file)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.labels[index]
        audio, phonemes = X['audio'], X['phonemes']
        start_time, end_time = Y['start_time'], Y['end_time']
        word_start_indexes, word_end_indexes = Y['start_indexes'], Y['end_indexes']
        name = Y['name']
        return name, audio, phonemes, start_time, end_time, word_start_indexes, word_end_indexes


if __name__ == "__main__":
    dataset = AriaDataset(path="dataset/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for name, audio, phonemes, start_time, end_time, word_start_indexes, word_end_indexes in dataloader:
        print("Done")
