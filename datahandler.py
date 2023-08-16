import os
from dataclasses import dataclass
import numpy as np
import librosa
import torch
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import List, Tuple
import Levenshtein


def get_dataloader_obj(dataset_patt="dataset/", pickle_file="dataset/aria_dataset.pickle", batch_size=1, shuffle=True):
    if os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = AriaDataset(dataset_patt)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def words_to_phonemes(words, w2ph_dict, folder_name):
    phonemes = []
    word_start_indexes = []
    word_end_indexes = []
    phonemes.append(">")
    start_idx = 1
    for word in words:
        if word in w2ph_dict:
            phoneme_list = w2ph_dict[word]
        else:
            print(f"Unable to find the match for word: {word} in Aria: {folder_name}")

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
        phonemes.extend(phoneme_list)
        word_start_indexes.append(start_idx)
        word_end_indexes.append(start_idx+len(phoneme_list))
        start_idx = start_idx+len(phoneme_list) + 1

        phonemes.append(">")
    return phonemes, word_start_indexes, word_end_indexes


def softmax(x: ndarray) -> List[float]:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_phonemes_in_intervals(phoneme_list: List[Tuple[int, float, float]], intervals: List[Tuple[float, float]]) \
            -> Tuple[List[List[int]], List[List[float]]]:
    """
    Given a list of phonemes with start and end times, and a list of time intervals, this function returns a list
    of phonemes that fall within each interval. It also computes probabilities based on the overlap of
    phonemes with each interval.

    Args:
        phoneme_list (List[Tuple[int, float, float]]): A sorted list of tuples where each tuple represents a
            phoneme. Each tuple is of the form (phoneme_index, start_time, end_time).

        intervals (List[Tuple[float, float]]): A sorted list of tuples where each tuple represents a time
            interval. Each tuple is of the form (start_time, end_time).

    Returns:
        Tuple[List[List[int]], List[List[float]]]: A tuple of two lists. The first list contains lists of phoneme
        indices that fall within each interval. The second list contains lists of probabilities
        corresponding to the overlap of phonemes within each interval.
    """

    result_indices = []
    result_probabilities = []
    phoneme_pointer = 0

    for interval in intervals:
        interval_start, interval_end = interval
        phonemes_in_interval = []
        probabilities = []
        total_overlap = 0

        # Start from the current phoneme pointer and add phonemes that overlap with the current interval
        temp_pointer = phoneme_pointer
        while temp_pointer < len(phoneme_list) and phoneme_list[temp_pointer][1] < interval_end:
            _, phoneme_start, phoneme_end = phoneme_list[temp_pointer]
            # Calculate overlap only for phonemes that start before the end of the interval and end after the start of the interval
            if phoneme_end > interval_start:
                overlap = min(phoneme_end, interval_end) - max(phoneme_start, interval_start)
                total_overlap += overlap
                phonemes_in_interval.append(temp_pointer)
                probabilities.append(overlap)
            # Move the original pointer forward only if the phoneme ends before the start of the next interval
            if phoneme_end <= interval_end:
                phoneme_pointer = temp_pointer + 1
            temp_pointer += 1

        # Normalize to probabilities
        probabilities = [overlap / total_overlap for overlap in probabilities] if total_overlap > 0 else []

        result_indices.append(phonemes_in_interval)
        result_probabilities.append(probabilities)

    return result_indices, result_probabilities


def phoneme_to_stft_frame(phoneme_timestamps, sample_rate, window_size, hop_length, total_duration):
    """
    This function maps the phonemes given in 'phoneme_timestamps' to STFT frames for a given audio file with specified
    sample rate and total duration. It outputs a list of phoneme indices and their corresponding percentages of overlap
    for each STFT frame.

    The function creates a list of tuples representing the phonemes and their start and end times, with the end time of
    a phoneme being the start time of the next. It then generates a list of time intervals corresponding to the STFT
    frames. Finally, the function gets the phonemes that fall into each STFT frame along with their percentages of
    overlap.

    Args:
        phoneme_timestamps (list of tuples): A list of tuples where each tuple is of the form (phoneme, start_time).

        sample_rate (int): The sample rate of the audio file in Hz.

        window_size (int): The size of the window used for the STFT in samples.

        hop_length (int): The number of samples between successive frames in STFT.

        total_duration (float): The total duration of the audio file in seconds.

    Returns:
        phonemes_in_stft_frames (List[List[int]]): A list where each element is a list of indices of the phonemes
        that fall into the corresponding STFT frame.

        percentages (List[List[float]]): A list where each element is a list of percentages representing the
        proportion of the corresponding phoneme's duration that falls into the STFT frame.
    """

    def wave_samples_to_seconds(frames):
        start_frame_number, end_frame_number = frames
        start_time_in_sec = start_frame_number/sample_rate
        end_time_in_sec = end_frame_number/sample_rate
        return start_time_in_sec, end_time_in_sec

    # Calculate the total number of frames
    total_frames = int(round(total_duration * sample_rate - window_size + hop_length) / hop_length)

    # Convert phoneme_timestamps into format (phoneme, start_time, end_time) by assuming that the end time for a phoneme
    #  is the start time of the next phoneme. The end_time of the last silence (">") is the song end time.
    phoneme_start_end_time = []
    for i in range(len(phoneme_timestamps) - 1):
        ph, start_time = phoneme_timestamps[i]
        _, end_time = phoneme_timestamps[i+1]
        phoneme_start_end_time.append((ph, start_time, end_time))

    # Add the start and end time for the final phoneme (which is going to be ">"). The end time is the song duration
    ph, start_time = phoneme_timestamps[-1]
    end_time = total_duration
    phoneme_start_end_time.append((ph, start_time, end_time))

    # Generate the list of intervals based on the STFT in the format of a List[float, float] where each entry contains
    #  the start time and the end time in seconds.
    frames = []
    for i in range(total_frames):
        frames.append((i * hop_length, i * hop_length + window_size))
    frames = list(map(wave_samples_to_seconds, frames))

    # Get the phonemes in each frame and their overlap percentage.
    phonemes_in_stft_frames, percentages = get_phonemes_in_intervals(phoneme_start_end_time, frames)

    return phonemes_in_stft_frames, percentages


def create_sparse_alpha_tensor_from_labels(words, w2ph_dict, sample_rate, start_time_arr, end_time_arr,
                                           song_duration, window_size=512, hop_length=256):
    """
    Create a sparse matrix (alpha matrix) based on labels, where each frame's phoneme indices are represented along
    with their corresponding softmax probabilities. This sparse tensor provides an efficient way to store and process
    large tensors where most values are zero.

    Args:
        words (List[str]): List of words in the song.
        w2ph_dict (Dict[str, List[str]]): Dictionary mapping words to phonemes.
        sample_rate (int): Sample rate of the audio file.
        start_time_arr (List[float]): Array of start times for each word in the song.
        end_time_arr (List[float]): Array of end times for each word in the song.
        song_duration (float): Total duration of the song in seconds.
        window_size (int, optional): Size of the window for the STFT. Defaults to 512.
        hop_length (int, optional): Number of samples to hop for the STFT. Defaults to 256.

    Returns:
        sparse_alpha_tensor (torch.sparse.FloatTensor): A sparse tensor where each column represents a frame, and the
                                                        rows represent the phonemes present in that frame.
                                                        The value at each non-zero entry is the corresponding softmax
                                                        probability of that phoneme in that frame. This tensor is
                                                        returned as a Sparse Tensor.
    """

    # sample_rate, song_duration = sample_rate.item(), song_duration.item()
    phoneme_timestamps = []     # Tuple of (str, float) containing the phoneme and it's onset time.
    silence_start_time = 0
    phoneme_timestamps.append((">", silence_start_time))
    for word, s_time, e_time in zip(words, start_time_arr, end_time_arr):
        # s_time, e_time = s_time.item(), e_time.item()
        phonemes_in_word = w2ph_dict[word]
        # Removing the tuple by using the first element. [('EH',), [(S,)] -> ['EH', 'S']
        phonemes_in_word = list(map(lambda x: x[0], phonemes_in_word))
        num_phonemes = len(phonemes_in_word)
        duration = e_time - s_time
        silence_start_time = e_time     # Assumption: Silence Starts when the word ends.
        duration_per_phoneme = duration/num_phonemes
        for i, ph in enumerate(phonemes_in_word):
            phoneme_timestamps.append((ph, s_time + i*duration_per_phoneme))
        phoneme_timestamps.append((">", silence_start_time))
    phonemes_in_stft_frames, percentages = phoneme_to_stft_frame(phoneme_timestamps, sample_rate, window_size, hop_length, song_duration)

    # Create the alpha matrix from the index matrix and percentages in "phonemes_in_stft_frames, percentages"

    # Find total number of frames and total phonemes (1-based indexing)
    total_frames = len(phonemes_in_stft_frames)
    total_phonemes = max(max(phonemes_in_stft_frames, default=0), default=0) + 1

    # Collect indices and values
    indices = []
    values = []
    for frame_idx, frame in enumerate(phonemes_in_stft_frames):
        for phoneme_idx, phoneme in enumerate(frame):
            # # In your indexing system, the x-coordinate (frame index) is flipped
            # indices.append([total_phonemes - phoneme - 1, frame_idx])
            # values.append(percentages[frame_idx][phoneme_idx])

            # Now using normal x-coordinate indexing
            indices.append([phoneme, frame_idx])
            values.append(percentages[frame_idx][phoneme_idx])

    # Convert to tensors
    indices = torch.LongTensor(indices).t()  # The indices tensor should be 2D with each column representing an index
    values = torch.FloatTensor(values)

    # Create sparse tensor
    sparse_alpha_tensor = torch.sparse.FloatTensor(indices, values, size=(total_phonemes, total_frames))
    # alpha_tensor = sparse_alpha_tensor.to_dense()
    return sparse_alpha_tensor


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
    def __init__(self, path, word2phoneme_dict, pickle_file='./dataset/aria_dataset.pickle', ignore_list=[]):
        """
        Create the dataset by either loading it from the pickle file if it is provided else creates one using the path.

        Folder structure requirements:
        The path should contain multiple folders, where each folder represents a song dataset.
        Each song dataset folder should have the following structure:

            song_dataset_folder/
            ├── audio/
            │   └── song.mp3
            ├── text/
            │   └── song.txt
            ├── labels.tsv
            └── word2phonemes.pickle [Deprecated] Not used directly.

        Folder Names:
        The name of the song dataset folder is used as the identifier for the dataset.

        File Names and Their Contents:
        - song.mp3: An audio file of the song.
        - song.txt: A text file containing the lyrics of the song. Each word should be separated by a space.
        - labels.tsv: A tab-separated file containing the start and end times of each word in the song.
                        The file should contain header lines [Text	Start Time	End Time].
        - [Depricated] w̶o̶r̶d̶2̶p̶h̶o̶n̶e̶m̶e̶s̶.̶p̶i̶c̶k̶l̶e̶: A̶ ̶p̶i̶c̶k̶l̶e̶ ̶f̶i̶l̶e̶ ̶t̶h̶a̶t̶ ̶m̶a̶p̶s̶ ̶e̶a̶c̶h̶ ̶w̶o̶r̶d̶ ̶t̶o̶ ̶i̶t̶s̶ ̶c̶o̶r̶r̶e̶s̶p̶o̶n̶d̶i̶n̶g̶ ̶p̶h̶o̶n̶e̶m̶e̶s̶.̶
                                            Now there is a global word2phoneme file that has been combined from all.

        Args:
            pickle_file (object):
            path (str): The path to the directory containing the song datasets.
        """
        self.ignore_list = ignore_list
        self.path = path
        self.word2phoneme_dict = word2phoneme_dict
        print("Creating the dataset, this may take a few minutes.")
        self._create_dataset()

    def _create_dataset(self):
        self.X = []
        self.labels = []

        folders = [folder for folder in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, folder))]
        for folder_name in folders:
            if folder_name in self.ignore_list:
                print(f"Ignoring {folder_name}.")
                continue
            if folder_name in ['.DS_Store', "aria_dataset.pickle"]:
                continue
            audio_file_path = os.path.join(self.path, folder_name, 'audio', 'song.mp3')
            text_file_path = os.path.join(self.path, folder_name, 'text', 'song.txt')
            labels_file_path = os.path.join(self.path, folder_name, 'labels.tsv')
            # Load audio file using librosa
            audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
            audio_duration = librosa.get_duration(y=audio, sr=sr)

            # Read text file as a list of words
            with open(text_file_path, 'r') as text_file:
                words = text_file.read().split()
                phonemes, word_start_indexes, word_end_indexes = words_to_phonemes(words, self.word2phoneme_dict, folder_name)
            self.X.append({"audio": audio, "phonemes": phonemes, "sr": sr, "audio_duration": audio_duration, "words": words})
            # Read timestamps from labels.tsv
            with open(labels_file_path, 'r') as labels_file:
                start_times, end_times = [], []
                lines = labels_file.readlines()[1:]  # Skip the first row (headers)
                for line in lines:
                    parts = line.strip().split('\t')
                    start_times.append(float(parts[1]))
                    end_times.append(float(parts[2]))

                # generate the alpha tensor for labels as a sparse matrix.
                # Using default hop length and window size.
                alpha_tensor = create_sparse_alpha_tensor_from_labels(words, self.word2phoneme_dict, sr, start_times,
                                                                      end_times, audio_duration)

                self.labels.append({"start_time": start_times,
                                    "end_time": end_times,
                                    "start_indexes": word_start_indexes,
                                    "end_indexes": word_end_indexes,
                                    "name": folder_name,
                                    "alpha_labels": alpha_tensor})
            print(f"Processed {folder_name}...")

    def pickle_dataset(self):
        with open('./dataset/aria_dataset.pickle', 'wb') as file:
            pickle.dump(self, file)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.labels[index]
        audio, phonemes, audio_duration, words = X['audio'], X['phonemes'], X["audio_duration"], X["words"]
        start_time, end_time = Y['start_time'], Y['end_time']
        word_start_indexes, word_end_indexes = Y['start_indexes'], Y['end_indexes']
        alpha_tensor = Y["alpha_labels"]
        name = Y['name']
        sr = X["sr"]

        audio = audio[None, :]    # Adding extra channel for model

        return name, audio, phonemes, sr, audio_duration, words, start_time, end_time, word_start_indexes, word_end_indexes, alpha_tensor


if __name__ == "__main__":
    dataset = AriaDataset(path="dataset/")
    # dataset.pickle_dataset()
    print("Done")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)