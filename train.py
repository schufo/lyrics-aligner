"""
Generates .txt-files with phoneme and/or word onsets.
"""
import copy
from datetime import datetime
import torch
import numpy as np
import wandb as wandb
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import model
import argparse
import os
import pickle
import warnings
from datahandler import AriaDataset

warnings.filterwarnings('ignore')


def compute_phoneme_onsets(optimal_path_matrix, hop_length, sampling_rate, return_skipped_idx=False):
    """

    Args:
        optimal_path_matrix: binary numpy array with shape (N, M)
        hop_length: int, hop length of the STFT
        sampling_rate: int, sampling frequency of the audio files

    Returns:
        phoneme_onsets: list
    """

    phoneme_indices = np.argmax(optimal_path_matrix, axis=1)  # Not Differentiable

    # find positions that have been skiped:
    skipped_idx = [x + 1 for i, (x, y) in
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
            word_onsets.append(phoneme_onsets[idx + 1])  # word onset is phoneme onset after space character
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
    for (m, n), (m_m1, n_m1) in zip(model.MatrixDiagonalIndexIterator(m=M + 1, n=N + 1, k_start=1),
                                    model.MatrixDiagonalIndexIterator(m=M, n=N, k_start=0)):
        d1 = dtw_matrix[n_m1, m]  # shape(number_of_considered_values)
        d2 = dtw_matrix[n_m1, m_m1]
        max_values = np.maximum(d1, d2)
        dtw_matrix[n, m] = score_matrix[0, n_m1, m_m1] + max_values
    return dtw_matrix[1:N + 1, 1:M + 1]


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
    optimal_path_matrix[0:n + 1, 0] = 1

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


def compute_hard_alignment(scores, lyrics_phoneme_symbols):
    optimal_path = optimal_alignment_path(scores)
    phoneme_onsets = compute_phoneme_onsets(optimal_path, hop_length=256, sampling_rate=16000)
    word_onsets, word_offsets = compute_word_alignment(lyrics_phoneme_symbols, phoneme_onsets)
    return phoneme_onsets, word_onsets, word_offsets


def save_hard_alignment(scores, lyrics_phoneme_symbols, file_name, word_list):
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


# Function to save model checkpoints
def save_checkpoint(model, run_name, epoch, steps, wandb):
    path = os.path.join("checkpoint", run_name)
    os.makedirs(path, exist_ok=True)
    filename = f'model_epoch_{epoch}_step_{steps}.pth' if epoch \
        else 'model_final.pth'
    full_path = os.path.join(path, filename)
    # Save locally
    torch.save(model.state_dict(), full_path)
    # Save on wandb - make sure the file is in the current directory or subdirectory.
    wandb.save(full_path)


# Function to perform training step and return loss
def train_step(lyrics_aligner, audio, phonemes_idx, alpha_tensor, optimizer, loss_fn):
    alpha_t_hat, scores = lyrics_aligner((audio, phonemes_idx))
    alpha_tensor = alpha_tensor.to_dense()
    alpha_tensor = alpha_tensor.permute((0, 2, 1))
    alpha_t_hat_flattened = alpha_t_hat.view(-1, alpha_t_hat.shape[-1])
    alpha_tensor_flattened = alpha_tensor.view(-1, alpha_tensor.shape[-1])
    loss = loss_fn(torch.log(alpha_t_hat_flattened + 1e-6), alpha_tensor_flattened)
    loss.backward()
    optimizer.step()
    return loss, scores


# Function to compute alignment MSE
def compute_alignment_mse(scores, phonemes, start_time):
    if type(start_time[0]) == torch.Tensor:
        start_time = list(map(lambda x: x.item(), start_time))
    detached_scores = scores.detach()
    lyrics_phoneme_symbols = list(map(lambda x: x[0], phonemes))
    h_phoneme_onsets, h_word_onsets, h_word_offsets = compute_hard_alignment(detached_scores, lyrics_phoneme_symbols)
    alignment_mse = np.mean((np.array(h_word_onsets) - np.array(start_time)) ** 2)
    return alignment_mse


def train(args):
    wandb.login(key="d2a2655d23be9c5fbe4d08ec428930c2c887d09f")
    num_epochs = args.epochs
    save_steps = args.save_steps
    run_name = args.run_name
    learning_rate = 1e-5  # Define learning rate as a variable for logging

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running model on {}.'.format(device))

    # Init Wandb
    project_name = 'lyrics_aligner' if torch.cuda.is_available() else 'lyrics_aligner_dev_macbook'
    run = wandb.init(project=project_name, name=args.run_name, config={
        'num_epochs': num_epochs,
        'save_steps': save_steps,
        'learning_rate': learning_rate,
    })

    # Load model
    lyrics_aligner = model.InformedOpenUnmix3().to(device)
    state_dict = torch.load('checkpoint/base/model_parameters.pth', map_location=device)
    lyrics_aligner.load_state_dict(state_dict)

    # Watch the model
    wandb.watch(lyrics_aligner, log='all')  # Log all gradients and model parameters

    # Define the baseline model for training comparison
    baseline = copy.deepcopy(lyrics_aligner)

    # Load dataset
    full_dataset = AriaDataset(path="dataset/")

    # Perform the split
    train_size = len(full_dataset) - 1
    test_size = 1
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Loss function
    loss_fn = nn.KLDivLoss()

    # Optimizer
    optimizer = Adam(lyrics_aligner.parameters(), lr=learning_rate)

    # Load (phoneme -> index) dictionary
    pickle_in = open('files/phoneme2idx.pickle', 'rb')
    phoneme2idx = pickle.load(pickle_in)
    steps = 0

    for epoch in range(num_epochs):
        for name, audio, phonemes, sr, audio_duration, words, start_time, end_time, word_start_indexes, word_end_indexes, w2ph_dict, alpha_tensor in dataloader:
            if name[0] == "aria_violetta":
                continue
            lyrics_phoneme_idx = [phoneme2idx[ph[0]] for ph in phonemes]
            phonemes_idx = torch.tensor(lyrics_phoneme_idx, dtype=torch.float32, device=device)[None, :]

            # Training step
            loss, scores = train_step(lyrics_aligner, audio, phonemes_idx, alpha_tensor, optimizer, loss_fn)

            # Save checkpoint
            steps += 1
            if steps % save_steps == 0:
                save_checkpoint(lyrics_aligner, run_name, epoch, steps, wandb)

            # Compute alignment MSE
            alignment_mse = compute_alignment_mse(scores, phonemes, start_time)

            # Log metrics to wandb
            wandb.log({
                'Training Loss': loss.item(),
                'Training Alignment MSE': alignment_mse,
                'Epoch': epoch,
                'Step': steps,
            })

    # Comparison with baseline model on test set
    with torch.no_grad():
        trained_model_mse_total = 0
        baseline_model_mse_total = 0
        total_samples = 0
        for name, audio, phonemes, sr, audio_duration, words, start_time, end_time, word_start_indexes, word_end_indexes, w2ph_dict, alpha_tensor in test_dataset:
            audio = torch.Tensor(audio)
            audio = audio[None, :]
            # Get model statistics over the test also comparing with the baseline
            lyrics_phoneme_idx = [phoneme2idx[ph] for ph in phonemes]
            phonemes_idx = torch.tensor(lyrics_phoneme_idx, dtype=torch.float32, device=device)[None, :]

            # Predict using the trained model
            _, scores = lyrics_aligner((audio.to(device), phonemes_idx))

            # Predict using the baseline model
            _, baseline_scores = baseline((audio.to(device), phonemes_idx))

            # Compute alignment MSE for the trained model and the baseline model and add to total
            trained_model_mse_total += compute_alignment_mse(scores, phonemes, start_time)
            baseline_model_mse_total += compute_alignment_mse(baseline_scores, phonemes, start_time)

            total_samples += 1

        # Compute average MSE
        avg_trained_model_mse = trained_model_mse_total / total_samples
        avg_baseline_model_mse = baseline_model_mse_total / total_samples

        # Log the average MSE scores to wandb
        wandb.log({
            'Average Trained Model MSE': avg_trained_model_mse,
            'Average Baseline Model MSE': avg_baseline_model_mse,
            'Total Samples': total_samples,
        })

    # Save final model
    # save_checkpoint(lyrics_aligner, run_name, None, None, wandb)

    # End the wandb run
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lyrics aligner')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=float('inf'))
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%m%d_%H%M"))
    args = parser.parse_args()

    train(args)
