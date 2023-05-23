import pickle
from typing import List, Optional
import textgrid
from util import clean_words


def _word_timestamps_from_phoneme_timestamps(phoneme_onset, word2phoneme, lyrics) -> List[tuple]:
    """
    Args:
        phoneme_onset: List of phoneme onset tuples. List[(ARPABET PHONEME, time in float seconds)]
        word2phoneme: word to corresponding ARPAbet phoneme dictionary
        lyrics: song lyrics as a list of words

    Returns: List[(str, float, float)]

    """
    phoneme_idx = 0
    word_timestamps = []
    for word in lyrics:
        phonemes_in_word = word2phoneme[word].split(" ")
        start_time = -1
        while phonemes_in_word:
            curr_phoneme = phonemes_in_word.pop(0)
            # TODO: Bug somewhere here, the pointers are not moving correctly. Check the phoneme model output file.
            while curr_phoneme != phoneme_onset[phoneme_idx][0]:
                phoneme_idx += 1
            if start_time == -1:
                start_time = phoneme_onset[phoneme_idx][1]
        phoneme_idx += 1
        end_time = phoneme_onset[phoneme_idx][1]
        word_timestamps.append((word, start_time, end_time))
    return word_timestamps


def _model_output_from_files(filepath: str, w2p_dict_file_path, lyrics_file_path) -> List[tuple]:
    """
    Returns:
        object: List[(str, float, float)]
    """
    with open(filepath, "r") as handle:
        phoneme_timestamps = [line.rstrip() for line in handle]
    phoneme_timestamps = list(map(lambda x: tuple(x.split('\t')), phoneme_timestamps))
    phoneme_onset = list(map(lambda x: (x[0], float(x[1])), phoneme_timestamps))

    with open(w2p_dict_file_path, "rb") as handle:
        word2phoneme = pickle.load(handle)

    with open(lyrics_file_path, "r") as handle:
        lyrics = [line.rstrip() for line in handle]
    lyrics = list(map(lambda x: x.split(" "), lyrics))
    lyrics = [word for line in lyrics for word in line]  # Creating a list of words from lyrics file
    lyrics = list(map(clean_words, lyrics))  # Removing unwanted symbols
    lyrics = list(filter(lambda x: x, lyrics))  # Removing Nones

    word_timestamps = _word_timestamps_from_phoneme_timestamps(phoneme_onset, word2phoneme, lyrics)
    return word_timestamps


def create_textgrid_file_from_model_output(filepath: str,
                                           word_timestamps: Optional[List[tuple]] = None,
                                           model_phoneme_output_file: Optional[str] = None,
                                           w2p_dict_file_path: Optional[str] = None,
                                           lyrics_file_path: Optional[str] = None) -> None:
    """

    Args:
        filepath:
        word_timestamps:
        model_phoneme_output_file:
        w2p_dict_file_path:
        lyrics_file_path:

    Returns:

    """
    WORD, WORD_START_TIME, WORD_END_TIME, LAST_ENTRY = 0, 1, 2, -1
    if not word_timestamps:
        if not model_phoneme_output_file or not w2p_dict_file_path or not lyrics_file_path:
            raise FileNotFoundError("Provide the location of files which contains the model phoneme output, "
                                    "lyrics and word to phoneme dictionary ")
        word_timestamps = _model_output_from_files(model_phoneme_output_file, w2p_dict_file_path, lyrics_file_path)
    max_time = word_timestamps[LAST_ENTRY][WORD_END_TIME]
    tg = textgrid.TextGrid(name=None,
                           minTime=0.0,
                           maxTime=max_time)
    model_output_tier = textgrid.IntervalTier(name="Standard Voice 1", minTime=0.0, maxTime=max_time)
    for word, start_time, end_time in word_timestamps:
        interval = textgrid.Interval(minTime=start_time, maxTime=end_time, mark=word)
        model_output_tier.addInterval(interval)
    tg.append(tier=model_output_tier)
    with open(filepath, 'w') as f:
        tg.write(f)
    print(f"Text grid file created at {filepath}")


def create_textgrid_file_from_model_output_resume(filepath: str, ass_obj,
                                                  word_timestamps: Optional[List[tuple]] = None,
                                                  model_phoneme_output_file: Optional[str] = None,
                                                  w2p_dict_file_path: Optional[str] = None,
                                                  lyrics_file_path: Optional[str] = None) -> None:
    """

    Args:
        filepath:
        word_timestamps:
        model_phoneme_output_file:
        w2p_dict_file_path:
        lyrics_file_path:

    Returns:

    """
    WORD, WORD_START_TIME, WORD_END_TIME, LAST_ENTRY = 0, 1, 2, -1
    if not word_timestamps:
        if not model_phoneme_output_file or not w2p_dict_file_path or not lyrics_file_path:
            raise FileNotFoundError("Provide the location of files which contains the model phoneme output, "
                                    "lyrics and word to phoneme dictionary ")
        word_timestamps = _model_output_from_files(model_phoneme_output_file, w2p_dict_file_path, lyrics_file_path)
    max_time = word_timestamps[LAST_ENTRY][WORD_END_TIME]
    tg = textgrid.TextGrid(name=None,
                           minTime=0.0,
                           maxTime=max_time)
    model_output_tier = textgrid.IntervalTier(name="Standard Voice 1", minTime=0.0, maxTime=max_time)
    aegi_sub_lines = ass_obj.events._lines
    prev_max = 0.0
    for i, (word, start_time, end_time) in enumerate(word_timestamps):
        aegi_sub_dialogue = aegi_sub_lines[i]
        if int(aegi_sub_dialogue.start.total_seconds()) < 100:
            start_time = aegi_sub_dialogue.start.total_seconds()
            end_time = aegi_sub_dialogue.end.total_seconds()
        prev_max = max(end_time, prev_max)
        interval = textgrid.Interval(minTime=start_time, maxTime=end_time, mark=word)
        try:
            model_output_tier.addInterval(interval)
        except ValueError as e:
            print(prev_max)
            print(interval.maxTime)
            print(interval.maxTime < prev_max)
            if interval.maxTime < prev_max:
                interval.maxTime = prev_max + 0.5
            interval.minTime = prev_max + 0.1
            model_output_tier.addInterval(interval)

    tg.append(tier=model_output_tier)
    with open(filepath, 'w') as f:
        tg.write(f)
    print(f"Text grid file created at {filepath}")


def create_dataset_from_textgrid_files(filepaths: List[str]) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    filepath = "/Users/pushkarjajoria/Desktop/Violetta/violetta_james.TextGrid"
    phoneme_output_file = "../arias/violetta/phoneme_model_output.txt"
    word_to_phoneme_dict = "../arias/violetta/word2phonemes.pickle"
    sung_text = "../arias/violetta/sung_text.txt"
    import ass

    with open("/Users/pushkarjajoria/Downloads/Archive/word_timestamps.ass", encoding='utf_8_sig') as handle:
        ass_obj = ass.parse(handle)
    create_textgrid_file_from_model_output_resume(filepath, ass_obj, model_phoneme_output_file=phoneme_output_file,
                                                  w2p_dict_file_path=word_to_phoneme_dict,
                                                  lyrics_file_path=sung_text)
