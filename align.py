"""
Generates .txt-files with phoneme and/or word onsets.
"""
import argparse
import os
import json
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

import librosa as lb
import torch
import numpy as np

import model

def compute_phoneme_onsets(optimal_path_matrix, hop_length, sampling_rate, return_skipped_idx=False):
    """

    Args:
        optimal_path_matrix: binary numpy array with shape (N, M)
        hop_length: int, hop length of the STFT
        sampling_rate: int, sampling frequency of the audio files

    Returns:
        phoneme_onsets: list
    """

    phoneme_indices = np.argmax(optimal_path_matrix, axis=1)

    # find positions that have been skiped:
    skipped_idx = [x+1 for i, (x, y) in
                   enumerate(zip(phoneme_indices[:-1], phoneme_indices[1:]))
                   if x == y - 2]

    # compute index of list elements whose right neighbor is different from itself
    last_idx_before_change = [i for i, (x, y) in
                              enumerate(zip(phoneme_indices[:-1], phoneme_indices[1:]))
                              if x != y]

    phoneme_onsets = [(n + 1) * hop_length / sampling_rate for n in last_idx_before_change]
    phoneme_onsets.insert(0, 0)  # the first space token's onset is 0

    if return_skipped_idx:
        return phoneme_onsets, skipped_idx
    else:
        for idx in skipped_idx:
            # set the onset of skipped tokens to the onset of the previous token
            phoneme_onsets.insert(idx, phoneme_onsets[idx])
        return phoneme_onsets

def compute_word_alignment(phonemes, phoneme_onsets):
    """

    Args:
        phonemes: list of phoneme symbols as strings. '>' as space character between words, at start, and end.
        phoneme_onsets: list of phoneme onsets. Must have same length as phonemes

    Returns:
        word_onsets: list of word onsets
        word_offsets: list of word offsets

    """
    word_onsets = []
    word_offsets = []

    for idx, phoneme in enumerate(phonemes):
        if idx == 0:
            word_onsets.append(phoneme_onsets[1])  # first word onset is first phoneme onset after space
            continue  # skip the first space token
        if phoneme == '>' and idx != len(phonemes) - 1:
            word_offsets.append(phoneme_onsets[idx])  # space onset is offset of previous word
            word_onsets.append(phoneme_onsets[idx+1]) # word onset is phoneme onset after space character
    word_offsets.append(phoneme_onsets[-1])  # last token (space token) onset is the last word's offset

    return word_onsets, word_offsets

def accumulated_cost_numpy(score_matrix, init=None):
    """
    Computes the accumulated score matrix by the "DTW forward operation"

    Args:
        score_matrix: torch.Tensor of shape (batch_size, length_sequence1, length_sequence2)
        init: int, value to initialize the point (0, 0) in accumpated cost matrix

    Returns:
        dtw_matrix: accumulated score matrix

    """
    B, N, M = score_matrix.size()
    score_matrix = score_matrix.numpy().astype('float64')

    dtw_matrix = np.ones((N + 1, M + 1)) * -100000
    dtw_matrix[0, 0] = init

    # Sweep diagonally through alphas (as done in https://github.com/lyprince/sdtw_pytorch/blob/master/sdtw.py)
    # See also https://towardsdatascience.com/gpu-optimized-dynamic-programming-8d5ba3d7064f
    for (m,n),(m_m1,n_m1) in zip(model.MatrixDiagonalIndexIterator(m = M + 1, n = N + 1, k_start=1),
                                 model.MatrixDiagonalIndexIterator(m = M, n= N, k_start=0)):
        d1 = dtw_matrix[n_m1, m] # shape(number_of_considered_values)
        d2 = dtw_matrix[n_m1, m_m1]
        max_values = np.maximum(d1, d2)
        dtw_matrix[n, m] = score_matrix[0, n_m1, m_m1] + max_values
    return dtw_matrix[1:N+1, 1:M+1]



def optimal_alignment_path(matrix, init=200):
    """
    Args:
        matrix: torch.Tensor with shape (1, sequence_length1, sequence_length2)
        init: int, value to initialize the point (0, 0) in accumpated cost matrix

    Returns:
        optimal_path_matrix:
    """

    # forward step DTW
    accumulated_scores = accumulated_cost_numpy(matrix, init=init)

    N, M = accumulated_scores.shape

    optimal_path_matrix = np.zeros((N, M))
    optimal_path_matrix[-1, -1] = 1  # last phoneme is active at last time frame
    # backtracking: go backwards through time steps n and put value of active m to 1 in optimal_path_matrix
    n = N - 2
    m = M - 1

    while m > 0:
        d1 = accumulated_scores[n, m]  # score at n of optimal phoneme at n-1
        d2 = accumulated_scores[n, m - 1]  # score at n of phoneme before optimal phoneme at n-1
        arg_max = np.argmax([d1, d2])  # = 0 if same phoneme active as before, = 1 if previous phoneme active
        optimal_path_matrix[n, m - arg_max] = 1
        n -= 1
        m -= arg_max
        if n == -2:
            print("DTW backward pass failed. n={} but m={}".format(n, m))
            break
    optimal_path_matrix[0:n+1, 0] = 1

    return optimal_path_matrix


def make_phoneme_and_word_list(text_file, word2phoneme_dict):
    word_list = []
    lyrics_phoneme_symbols = ['>']
    with open(text_file, encoding='utf-8') as lyrics:
        lines = lyrics.readlines()
        for line in lines:
            line = line.lower().replace('\n', '').replace('’', "'")
            clean_line = ''.join(c for c in line if c.isalnum() or c in ["'", ' '])
            if clean_line == ' ' or clean_line == '': continue
            words = clean_line.split(' ')
            for word in words:
                if word == '': continue
                word_list.append(word)
                phonemes = word2phoneme_dict[word].split(' ')
                for p in phonemes:
                    lyrics_phoneme_symbols.append(p)
                lyrics_phoneme_symbols.append('>')
    return lyrics_phoneme_symbols, word_list

def make_phoneme_list(text_file):
    lyrics_phoneme_symbols = []
    with open(text_file, encoding='utf-8') as lyrics:
        lines = lyrics.readlines()
        for line in lines:
            phoneme = line.replace('\n', '').upper()
            if phoneme in [' ', '']: continue
            lyrics_phoneme_symbols.append(phoneme)
    return lyrics_phoneme_symbols


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lyrics aligner')

    parser.add_argument('audio_path', type=str)
    parser.add_argument('lyrics_path', type=str)
    parser.add_argument('--lyrics-format', type=str, choices=['w', 'p'], default='w')
    parser.add_argument('--onsets', type=str, choices=['p', 'w', 'pw'], default='p')
    parser.add_argument('--dataset-name', type=str, default='dataset1')
    parser.add_argument('--vad-threshold', type=float, default=0)
    args = parser.parse_args()

    audio_files = sorted(glob.glob(os.path.join(args.audio_path, '*')))

    pickle_in = open('files/{}_word2phonemes.pickle'.format(args.dataset_name), 'rb')
    word2phonemes = pickle.load(pickle_in)
    pickle_in = open('files/phoneme2idx.pickle', 'rb')
    phoneme2idx = pickle.load(pickle_in)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device_printed = 'GPU' if torch.cuda.is_available() else 'CPU'
    print('Running model on {}.'.format(device_printed))

    # load model
    lyrics_aligner = model.InformedOpenUnmix3().to(device)
    state_dict = torch.load('model_parameters.pth', map_location=device)
    lyrics_aligner.load_state_dict(state_dict)

    if args.onsets in ['p', 'pw']:
        os.makedirs('outputs/{}/phoneme_onsets'.format(args.dataset_name), exist_ok=True)
    if args.onsets in ['w', 'pw']:
        os.makedirs('outputs/{}/word_onsets'.format(args.dataset_name), exist_ok=True)

    for audio_file_path in audio_files:

        audio_file = os.path.basename(audio_file_path)
        print('Processing {} ...'.format(audio_file))
        file_name, ext = os.path.splitext(audio_file)

        # get corresponding lyrics file
        lyrics_file_path = os.path.join(args.lyrics_path, file_name + '.txt')

        if args.lyrics_format == 'w':
            lyrics_phoneme_symbols, word_list = make_phoneme_and_word_list(lyrics_file_path, word2phonemes)
        elif args.lyrics_format == 'p':
            lyrics_phoneme_symbols = make_phoneme_list(lyrics_file_path)

        lyrics_phoneme_idx = [phoneme2idx[p] for p in lyrics_phoneme_symbols]
        phonemes_idx = torch.tensor(lyrics_phoneme_idx, dtype=torch.float32, device=device)[None, :]

        # audio processing: load, resample, to mono, to torch
        audio, sr = lb.load(audio_file_path, sr=16000, mono=True)
        audio_torch = torch.tensor(audio, dtype=torch.float32, device=device)[None, None, :]

        # compute alignment
        with torch.no_grad():
            voice_estimate, _, scores = lyrics_aligner((audio_torch, phonemes_idx))
            scores = scores.cpu()

        if args.vad_threshold > 0:
            # vocal activity detection
            voice_estimate = voice_estimate[:, 0, 0, :].cpu().numpy().T
            vocals_mag = np.sum(voice_estimate, axis=0)

            # frames with vocal magnitude below threshold are considered silence
            predicted_silence = np.nonzero(vocals_mag < args.vad_threshold)

            is_space_token = torch.nonzero(phonemes_idx == 3, as_tuple=True)

            # set score of space tokens to high value in silent frames
            for n in predicted_silence[0]:
                scores[:, n, is_space_token[1]] = scores.max()

        optimal_path = optimal_alignment_path(scores)
        phoneme_onsets = compute_phoneme_onsets(optimal_path, hop_length=256, sampling_rate=16000)

        if args.onsets in ['p', 'pw']:
            # save phoneme onsets
            p_file = open('outputs/{}/phoneme_onsets/{}.txt'.format(args.dataset_name, file_name), 'a')
            for m, symb in enumerate(lyrics_phoneme_symbols):
                p_file.write(symb + '\t' + str(phoneme_onsets[m]) + '\n')
            p_file.close()

        if args.onsets in ['w', 'pw']:
            word_onsets, word_offsets = compute_word_alignment(lyrics_phoneme_symbols, phoneme_onsets)

            # save word onsets
            w_file = open('outputs/{}/word_onsets/{}.txt'.format(args.dataset_name, file_name), 'a')
            for m, word in enumerate(word_list):
                w_file.write(word + '\t' + str(word_onsets[m]) + '\n')
            w_file.close()

        print('Done.')
