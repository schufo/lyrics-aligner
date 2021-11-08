# Phoneme level lyrics aligner

This repository can be used to align lyrics transcripts with the corresponding audio signals. The audio signals may contain solo singing or singing voice mixed with other instruments.
It contains a trained deep neural network which performs alignment and singing voice separation jointly.
Details about the model, training, and data are described in the associated paper
> Schulze-Forster, K., Doire, C., Richard, G., & Badeau, R. "Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation." IEEE/ACM Transactions on Audio, Speech and Language Processing (2021). doi: [10.1109/TASLP.2021.3091817](https://doi.org/10.1109/TASLP.2021.3091817). public version [available here](https://hal.telecom-paris.fr/hal-03255334/file/2021_Phoneme_level_lyrics_alignment_and_text-informed_singing_voice_separation.pdf).

If you use the model or code, please cite the paper:
<pre>
@article{schulze2021phoneme,
    author={Schulze-Forster, Kilian and Doire, Clement S. J. and Richard, Gaël and Badeau, Roland},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    title={Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation}, 
    year={2021},
    volume={29},
    number={},
    pages={2382-2395},
    doi={10.1109/TASLP.2021.3091817}
    }
</pre>

## Installation
1.  Clone the repository:
    <pre>
    git clone https://github.com/schufo/lyrics-aligner.git
    </pre>
2.  Install the conda environment:

    - If you want to run the model on a CPU:
      <pre>
      conda env create -f environment_cpu.yml
      </pre>
    - If you want to run the model on a GPU:
      <pre>
      conda env create -f environment_gpu.yml
      </pre>

Remember to activate the conda environment.

## Data preparation
### Audio
Please prepare one directory with all audio files. We load the audio files using librosa, so all formats supported by librosa can be used. This includes for example .wav and .mp3. See [the documentation](https://librosa.org/doc/latest/index.html) for more details.
### Lyrics
Please prepare a separate directory with all lyrics files in .txt-format. Each lyrics file must have the same name as the corresponding audio file (e.g. song1.wav --> song1.txt).

You can provide the lyrics as words or as phonemes.

If your lyrics are already decomposed into phonemes, please consider the following:
- We support only the 39 phonemes in ARPAbet notation listed on [website](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) of the CMU Pronouncing Dictionary.
- The provided .txt-file should contain one phoneme per line.
- The first and the last symbol should be the space character: >. It should also be placed between each word or at positions where silence between phonemes is expected in the singing voice signal.
- In this case only phoneme onsets and no word onsets can be computed.

If the lyrics are provided as words, they must be processed as follows to be used with the alignment model:
1. Generate a .txt-file with a list of unique words:
    <pre>
    python make_word_list.py PATH/TO/LYRICS/DIRECTORY --dataset-name NAME
    </pre>
    The `--dataset-name` flag is optional. It can be used if several datasets should be aligned with this model. The output files will contain the dataset name which defaults to 'dataset1'.
    This command generates the files `NAME_word_list.txt` and `NAME_word2phoneme.txt` in the `files` directory.
2. Go to [http://www.speech.cs.cmu.edu/tools/lextool.html](http://www.speech.cs.cmu.edu/tools/lextool.html), upload `NAME_word_list.txt` as word file, and click COMPILE.
3. Click on the link to see the list of output files. Then, click on the .dict-file. You should now see a list of all words with their corresponding phoneme decomposition.
4. Copy the whole list and paste it into `NAME_word2phoneme.txt` in the `files` directory.
5. Run the following command:
    <pre>
    python make_word2phoneme_dict.py --dataset-name NAME
    </pre>
    Use the same dataset name as in step 1. This will generate a Python dictionary to translate each word into phonemes and save it as `NAME_word2phonemes.pickle` in `files`.
6. Done!

## Usage
The model has been trained on the [MUSDB18 dataset](https://zenodo.org/record/1117372#.YYgpfy9XZQI) using the [lyrics extension](https://zenodo.org/record/3989267#.YYgpdS9XZQI). Therefore, it will probably work best with similar music. However, we also found it works well on solo singing. Some errors can be expected in challenging mixtures with long instrumental sections.

You can compute phoneme onsets and/or word onsets as follows:
<pre>
python align.py PATH/TO/AUDIO/DIRECTORY PATH/TO/LYRICS/DIRECTORY \
--lyrics-format w --onsets p --dataset-name dataset1 --vad-threshold 0
</pre>
Optional flags (defaults are shown above):

`--lyrics-format` Must be `w` if the lyrics are provided as words (and has been processed as descrived above) and `p` if the lyrics are provided as phonemes.

`--onsets` If phoneme onsets should be computed, set to `p`. If word onsets should be computed, set to `w`. If phoneme and word onsets should be computed, set to `pw` (only possible if lyrics are provided as words).

`--dataset-name` Should be the same as used for data preparation above.

`--vad-threshold` The model also computes an estimate of the isolated singing voice which can be used as Voice Activity Detector (VAD). This may be useful in challenging scenarios where long pauses are made by the singer while instruments are playing (e.g. intro, soli, outro). The magnitude of the vocals estimate is computed. Here a threshold (float) can be set to discriminate between active and inactive voice given the magnitude. The default is 0 which means that no VAD is used. The optimal value for a given audio signal may be difficult to determine as it depends on the loudness of the voice. In our experiments we used values between 0 and 30. You could print or plot the voice magnitude (computed in line 235) to get an intuition for an appropriate value. We recommend to use the option only if large errors are made on audio files with long instrumental sections. 

## Acknowledgment
This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

## Copyright
Copyright 2021 Kilian Schulze-Forster of Télécom Paris, Institut Polytechnique de Paris. All rights reserved.