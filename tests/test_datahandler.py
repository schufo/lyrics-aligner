from unittest import TestCase

import torch

from datahandler import phoneme_to_stft_frame, get_phonemes_in_intervals, create_sparse_alpha_tensor_from_labels


class Test(TestCase):
    def setUp(self):
        self.phoneme_list = [
            (0, 0.0, 1.0),
            (1, 1.0, 2.0),
            (2, 2.0, 3.0),
            (3, 3.0, 4.0),
            (4, 4.0, 5.0)
        ]

        self.intervals = [
            (0.0, 1.5),
            (1.5, 2.5),
            (2.5, 3.5),
            (3.5, 4.5),
            (4.5, 5.0)
        ]
        self.phoneme_timestamps = [('A', 0.0), ('B', 1.0), ('C', 2.0), ('D', 3.0), ('E', 4.0)]
        self.sample_rate = 22050  # Commonly used sample rate
        self.window_size = 1024
        self.hop_length = 512
        self.total_duration = 5.0  # In seconds

        self.words = ["word1", "word2", "word3"]
        self.w2ph_dict = {
            "word1": ["A"],
            "word2": ["B"],
            "word3": ["C"]
        }
        self.sample_rate_t = torch.Tensor([22050])
        self.start_time_arr = torch.Tensor([0.0, 1.0, 2.0])
        self.end_time_arr = torch.Tensor([1.0, 2.0, 3.0])
        self.song_duration = torch.Tensor([3.0])

    # def test_get_phonemes_in_intervals(self):
    #     result_indices, result_probabilities = get_phonemes_in_intervals(self.phoneme_list, self.intervals)
    #     self.assertEqual(result_indices, [[0, 1], [1, 2], [2, 3], [3, 4], [4]])
    #     self.assertEqual(result_probabilities, [[0.6666666666666666, 0.3333333333333333], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1.0]])
    #
    # def test_phoneme_to_stft_frame(self):
    #     phonemes_in_stft_frames, percentages = phoneme_to_stft_frame(
    #         self.phoneme_timestamps,
    #         self.sample_rate,
    #         self.window_size,
    #         self.hop_length,
    #         self.total_duration
    #     )
    #     self.assertEqual(phonemes_in_stft_frames, [[0], [0, 1], [1], [1, 2], [2], [2, 3], [3], [3, 4], [4], [4]])
    #     self.assertEqual(percentages, [[1.0], [0.5, 0.5], [1.0], [0.5, 0.5], [1.0], [0.5, 0.5], [1.0], [0.5, 0.5], [1.0], [1.0]])
    #
    # def test_create_sparse_alpha_matrix_from_labels(self):
    #     alpha_tensor = create_sparse_alpha_matrix_from_labels(
    #         self.words,
    #         self.w2ph_dict,
    #         self.sample_rate_t,
    #         self.start_time_arr,
    #         self.end_time_arr,
    #         self.song_duration,
    #         self.window_size,
    #         self.hop_length
    #     )
    #     self.assertEqual(alpha_tensor._indices().tolist(), [[1, 1, 0, 0], [0, 1, 1, 2]])
    #     self.assertEqual(alpha_tensor._values().tolist(), [0.5, 0.5, 0.5, 0.5])
    #
    #
