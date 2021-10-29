"""
This script saves the phoneme or word onsets estimated by a specified model (tag) on a specified dataset
"""
import argparse
#import testx
import torch
import numpy as np
import os
import soundfile as sf
import json
import matplotlib.pyplot as plt

#import data
import model
#import utils

def compute_phoneme_onsets(optimal_path_matrix, hop_length, sampling_rate, return_skipped_idx=False):
    """

    Args:
        optimal_path_matrix: binary numpy array with shape (N, M)
        hop_length: int, hop length of the STFT
        sampling_rate: int, sampling frequency of the audio files

    Returns:

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
    phoneme_onsets.insert(0, 0)  # the random token (%) onset is 0

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
        phonemes: list of phoneme symbols as strings. '>' as space character between words
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

def accumulated_cost_numpy(score_matrix, mode='max', init=None):
    """
    Computes the accumulated score matrix by the "DTW forward operation"
    Args:
        score_matrix: distance matrix, shape(batch_size, length_sequence1, length_sequence2)
        mode:

    Returns:
        dtw_matrix: accumulated score matrix

    """
    B, N, M = score_matrix.size()
    score_matrix = score_matrix.numpy().astype('float64')

    if mode == 'max':
        # there is an issue with pytorch backward computation when using 'faster' with pytorch 1.2.0:
        # https://github.com/pytorch/pytorch/issues/24853
        #dtw_matrix = np.zeros((N + 1, M + 1)) #, device=device)
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


def optimal_alignment_path(matrix, mode='max_numpy', init=200000):

    # matrix is torch.tensor with size (1, sequence_length1, sequence_length2)

    # it is not allowed to skip idx 0 and 1

    # forward step DTW
    if mode == 'max_numpy':
        accumulated_scores = accumulated_cost_numpy(matrix, mode='max', init=init)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lyrics aligner')

    parser.add_argument('-audio-path', type=str)
    parser.add_argument('-lyrics-path', type=str)


    # load model
    lyrics_aligner = model.InformedOpenUnmix3()
    state_dict = torch.load('../open_unmix/trained_models/inf_umx_dtw_3.2.3sil2/vocals.pth', map_location='cpu')
    lyrics_aligner.load_state_dict(state_dict)


    print(lyrics_aligner)

    quit()










    # decide model
    tag = 'JOINT3'
    parser.add_argument('--tag', type=str, default=tag)

    parser.add_argument('--vad-threshold', type=float, default=20)

    args, _ = parser.parse_known_args()
    tag = args.tag

    parser.add_argument(
        '--eval-tag',
        type=str,
        default= args.tag,
        help ='tag for evaluation folder etc.')

    parser.add_argument(
        '--testset',
        type=str,
        default='Hansen',
        help ='dataset on which to run the evaluation')

    args, _ = parser.parse_known_args()

    # decide test set
    test_set = args.testset

    if test_set == 'Hansen':
        dataset = data.Hansen()
    elif test_set == 'Jamendo':
        dataset = data.Jamendo()
    elif test_set == 'NUS_acapella':
        dataset = data.NUS(acapella=True)
    elif test_set == 'NUS':
        parser.add_argument('--snr', type=int, default=5)  # SNR for mixing vocals and music
        args, _ = parser.parse_known_args()
        dataset = data.NUS(acapella=False, snr=args.snr)

    model_path = 'trained_models/{}'.format(tag)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("Device:", device)
    target = 'vocals'

    # load model
    model_to_test = testx.load_model(target, model_path, device)
    model_to_test.return_alphas = True
    model_to_test.eval()

    # load model config
    with open(os.path.join(model_path, target + '.json'), 'r') as stream:
        config = json.load(stream)
        samplerate = config['args']['samplerate']
        text_units = config['args']['text_units']
        nfft = config['args']['nfft']
        nhop = config['args']['nhop']

    mean_onset_errors = []
    median_onset_errors = []

    if test_set[:3] == 'NUS':
        # phoneme level alignment

        if test_set == 'NUS':
            path_to_save_alignment = 'evaluation/{}/alignments/{}_snr{}'.format(args.eval_tag, test_set, args.snr)
        else:
            path_to_save_alignment = 'evaluation/{}/alignments/{}'.format(args.eval_tag, test_set)
        if not os.path.isdir(path_to_save_alignment):
            os.makedirs(path_to_save_alignment)

        for idx in range(len(dataset)):
            test_example = dataset[idx]
            name = test_example['name']
            audio = test_example['audio'].unsqueeze(dim=0).unsqueeze(dim=1)
            phoneme_idx = test_example['text_phoneme_idx'].unsqueeze(dim=0)
            true_onsets = test_example['true_onsets']

            with torch.no_grad():
                vocals_estimate, alphas, scores = model_to_test((audio, phoneme_idx))

            optimal_path_scores = optimal_alignment_path(scores, mode='max_numpy', init=200)

            phoneme_onsets = compute_phoneme_onsets(optimal_path_scores, hop_length=nhop, sampling_rate=samplerate)

            np.save(os.path.join(path_to_save_alignment, name + '_onsets'), np.array(phoneme_onsets))

            print(name)
            abs_errors_onsets = abs(np.array(phoneme_onsets, dtype=np.float) - np.array(true_onsets, dtype=np.float))
            print('onset error', np.mean(abs_errors_onsets), np.median(abs_errors_onsets))
            mean_onset_errors.append(np.mean(abs_errors_onsets))
            median_onset_errors.append(np.median(abs_errors_onsets))

    else:
        # word level alignment

        path_to_save_alignment = 'evaluation/{}/alignments/{}'.format(args.eval_tag, test_set)
        if not os.path.isdir(path_to_save_alignment):
            os.makedirs(path_to_save_alignment)
        for idx in range(len(dataset)):
            test_example = dataset[idx]
            name = test_example['name']
            audio = test_example['audio'].unsqueeze(dim=0).unsqueeze(dim=1)
            phoneme_idx = test_example['text_phoneme_idx'].unsqueeze(dim=0)
            true_onsets = test_example['true_onsets']
            phoneme_symbols = test_example['text_phoneme_symbols']

            if dataset == 'Hansen':
                true_offsets = test_example['true_offsets']

            with torch.no_grad():
                vocals_estimate, alphas, scores = model_to_test((audio, phoneme_idx))

            # vocal activity detection
            vocals_estimate = vocals_estimate[:, 0, 0, :].numpy().T
            vocals_mag = np.sum(vocals_estimate, axis=0)

            predicted_silence = np.nonzero(vocals_mag < args.vad_threshold)
            is_space_token = torch.nonzero(phoneme_idx == 3, as_tuple=True)

            score_vocals = scores
            for n in predicted_silence[0]:
                score_vocals[:, n, is_space_token[1]] = score_vocals.max()
            optimal_path_scores_vocals = model.optimal_alignment_path(score_vocals)

            phoneme_onsets = compute_phoneme_onsets(optimal_path_scores_vocals, hop_length=256, sampling_rate=16000)
            word_onsets, word_offsets = compute_word_alignment(phonemes=phoneme_symbols, phoneme_onsets=phoneme_onsets)

            np.save(os.path.join(path_to_save_alignment, name + '_onsets'), np.array(word_onsets))
            np.save(os.path.join(path_to_save_alignment, name + '_offsets'), np.array(word_offsets))

            print(name)
            abs_errors_onsets = abs(np.array(word_onsets, dtype=np.float) - np.array(true_onsets, dtype=np.float))
            print('onset error', np.mean(abs_errors_onsets), np.median(abs_errors_onsets))
            mean_onset_errors.append(np.mean(abs_errors_onsets))
            median_onset_errors.append(np.median(abs_errors_onsets))

    print("Mean mean absolute error onsets:", np.mean(np.array(mean_onset_errors)))
    print("Mean median absolute error onsets:", np.mean(np.array(median_onset_errors)))
