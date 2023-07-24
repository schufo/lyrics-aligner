import numpy as np
import librosa


def spec_augment(mel_spectrogram, frequency_masking_para=27, time_masking_para=100, frequency_mask_num=1,
                 time_mask_num=1):
    """
    Perform SpecAugment on the given mel spectrogram.

    Args:
        mel_spectrogram (np.ndarray): The input mel spectrogram.
        frequency_masking_para (int, optional): Frequency masking parameter. Defaults to 27.
        time_masking_para (int, optional): Time masking parameter. Defaults to 100.
        frequency_mask_num (int, optional): The number of frequency masks to apply. Defaults to 1.
        time_mask_num (int, optional): The number of time masks to apply. Defaults to 1.

    Returns:
        np.ndarray: The augmented mel spectrogram.
    """

    tau = mel_spectrogram.shape[1]  # Get time dimension
    v = mel_spectrogram.shape[0]  # Get frequency dimension

    # Frequency masking
    for i in range(frequency_mask_num):
        # Randomly choose the width of the frequency mask (f) and its starting point (f0)
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)  # convert to integer
        f0 = np.random.randint(low=0, high=v - f)  # ensure that the starting point is within bounds
        mel_spectrogram[f0:f0 + f, :] = 0  # set a band of frequencies to zero

    # Time masking
    for i in range(time_mask_num):
        # Randomly choose the width of the time mask (t) and its starting point (t0)
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)  # convert to integer
        t0 = np.random.randint(low=0, high=tau - t)  # ensure that the starting point is within bounds
        mel_spectrogram[:, t0:t0 + t] = 0  # set a portion of the time sequence to zero

    return mel_spectrogram  # return augmented mel spectrogram


if __name__ == "__main__":
    # Load an audio file
    y, sr = librosa.load("your-audio-file.wav")  # 'y' is the audio time-series and 'sr' is the sample rate

    # Convert the audio to a mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)

    # Apply SpecAugment data augmentation to the mel spectrogram
    augmented_mel_spectrogram = spec_augment(mel_spectrogram)
