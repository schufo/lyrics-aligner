import datetime
import sys
import time
from itertools import zip_longest
from typing import Tuple, List

from textgrid import TextGrid, IntervalTier
import ass
import textgrid
import csv
import openpyxl


def create_tsv_file(data, file_path):
    """Creates a TSV file from a list of tuples (text, start_time, end_time)"""
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header
        writer.writerow(['Text', 'Start Time', 'End Time'])

        # Write the data rows
        for row in data:
            writer.writerow(row)


def create_excel_file(words1, words2, file_path):
    """Creates an Excel file from two lists of words"""
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # Write the words from list 1 to the first column
    for i, word in enumerate(words1, start=1):
        sheet.cell(row=i, column=1).value = word

    # Write the words from list 2 to the second column
    for i, word in enumerate(words2, start=1):
        sheet.cell(row=i, column=2).value = word

    # Save the workbook to the specified file path
    workbook.save(file_path)


def clean_words(s: str) -> str:
    """Strips punctuations from words"""
    s = s.replace(".", "").replace("…", "").replace(";", "").replace("?", "").replace(",", "") \
        .replace("!", "").replace(":", "").strip("\"") \
        .strip("(").strip(")").replace("–", "")
    return s.lower()


def read_lyrics_file(file_path):
    """Reads lyrics from a file and converts them to a list of words"""
    words = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual words
            line_words = line.strip().split()

            # Clean each word and add it to the list
            cleaned_words = [clean_words(word) for word in line_words]
            words.extend(cleaned_words)

    return words


def sec2srttime(sec: float) -> str:
    """Converts float(sec) into a hh:mm:ss,mss format"""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    ms = int((sec % max(1, int(sec))) * 1000)
    hours = str(h).zfill(2)
    min = str(m).zfill(2)
    sec = str(s).zfill(2)
    millisec = str(ms).zfill(3)
    return f"{hours}:{min}:{sec},{millisec}"


def grouper(n, iterable, fillvalue=None):
    """Iterating a python iterable in a group of 'n'. Used to add multiple words to a single srt line"""
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def create_srt_line(*args):
    """Combining multiple tuples(can be None) into a single line of srt string"""
    out = ""
    for arg in args:
        if arg:
            out += arg[0]
            out += " "
    return out.strip()


def print_timed_lyrics(_word_timestamps):
    time_counter = 0
    for w, start_time, end_time in _word_timestamps:
        time_to_wait = max(0, start_time - time_counter)
        time.sleep(time_to_wait)
        convert = str(datetime.timedelta(seconds=start_time))
        print(f"[{w}] -------> {convert}")
        sys.stdout.flush()
        time_counter = start_time


def print_timed_subtitles(_word_timestamps):
    time_counter = 0
    for t1, t2, t3, t4 in grouper(4, _word_timestamps):
        start_time = t1[1]
        time_to_wait = max(0, start_time - time_counter)
        time.sleep(time_to_wait)
        convert = str(datetime.timedelta(seconds=start_time))
        print(f"[{create_srt_line(t1, t2, t3, t4)}] -------> {convert}")
        sys.stdout.flush()
        time_counter = start_time


def create_textgrid(word_tuples, tier_name='words'):
    # Create an empty TextGrid
    grid = TextGrid()

    # Calculate the max end time to set the TextGrid maxTime
    max_end_time = max(word_tuples, key=lambda x: x[2])[2]
    grid.maxTime = max_end_time

    # Create an IntervalTier
    tier = IntervalTier(name=tier_name, maxTime=max_end_time)

    # Add each word to the IntervalTier
    for word, start_time, end_time in word_tuples:
        tier.add(start_time, end_time, word)

    # Add the IntervalTier to the TextGrid
    grid.append(tier)

    return grid


from typing import List, Tuple


def resolve_conflicts(sorted_word_tuples: List[Tuple[str, float, float]]):
    # Unpack the first word into separate variables
    text1, start_time1, end_time1 = sorted_word_tuples[0]

    # Initialize the list with the first word
    resolved_word_tuples = [(text1, start_time1, end_time1)]

    for current_word in sorted_word_tuples[1:]:
        # Get the last word in the resolved list
        last_word_text, last_word_start, last_word_end = resolved_word_tuples[-1]

        # Unpack the current word into separate variables
        current_word_text, current_word_start, current_word_end = current_word

        # If the start time of the current word is less than or equal to
        # the end time of the last word plus 0.1
        if current_word_start <= last_word_end:
            # Adjust the start time of the current word
            new_start_time = last_word_end + 0.1
            print(f"{current_word_start, last_word_end} -> {new_start_time} | Adjusting the start_time of {current_word_text} because of conflict with {last_word_text}")

            # If new duration is less than or equal to 0.1, adjust the end time
            if current_word_end - new_start_time <= 0.1:
                print(f"{current_word_text} became invalid after adjustment. Increasing the endtime.")
                new_end_time = new_start_time + 0.1
            else:
                new_end_time = current_word_end

            # Append the word with the new start and end times
            resolved_word_tuples.append((current_word_text, new_start_time, new_end_time))
        else:
            # If there's no need for adjustment, simply append the current word
            resolved_word_tuples.append(current_word)

    return resolved_word_tuples


def is_compound_word(text: str):
    return not text.isspace() and len(text.split(" ")) > 1


def subdivide(text, start_time, end_time):
    words = text.split()
    num_words = len(words)
    total_time = end_time - start_time
    time_per_word = total_time / num_words
    subdivisions = []
    for i in range(num_words):
        word_start_time = round(start_time + (i * time_per_word), 2)
        word_end_time = round(word_start_time + time_per_word, 2)
        subdivisions.append((words[i], word_start_time, word_end_time))
        start_time += 0.01
    return subdivisions


def combine_textgrid_files(file_list):
    def check_conflict(word1, word2, conflict_threshold=0.01):
        # unpack the tuples
        text1, start_time1, end_time1 = word1
        text2, start_time2, end_time2 = word2

        # check if the texts are the same
        same_text = (text1 == text2)

        # check for overlapping intervals and calculate the overlap duration
        if (start_time1 <= end_time2) and (end_time1 >= start_time2):
            overlap_duration = min(end_time1, end_time2) - max(start_time1, start_time2)
        else:
            overlap_duration = 0

        # check if the overlap duration is greater than conflict_threshold
        significant_overlap = (overlap_duration > conflict_threshold)

        # check if both conditions are met
        if same_text and significant_overlap:
            # calculate the durations of the intervals
            duration1 = end_time1 - start_time1
            duration2 = end_time2 - start_time2

            # return the bigger interval along with True
            if duration1 > duration2:
                return True, word1
            else:
                return True, word2
        else:
            print("No Conflict found")
            sys.stdout.flush()
            # if no conflict, return False and None
            return False, None

    word_labels = []
    james_annotation_time = 179.0
    with open("/Users/pushkarjajoria/Downloads/Archive/word_timestamps.ass", encoding='utf_8_sig') as handle:
        ass_obj = ass.parse(handle)
    aegi_sub_lines = ass_obj.events._lines
    time_start, i = 0.0, 0
    while time_start < james_annotation_time:
        curr_line = aegi_sub_lines[i]
        word_labels.append((curr_line.text, curr_line.start.total_seconds(), curr_line.end.total_seconds()))
        time_start = aegi_sub_lines[i+1].start.total_seconds()
        i += 1
    for i, file_path in enumerate(file_list):
        tg = textgrid.TextGrid.fromFile(file_path)
        for tier in tg:
            for interval in tier:
                text = interval.mark
                start_time = interval.minTime
                end_time = interval.maxTime
                if is_compound_word(text):
                    # Equally divide the interval for each word in the compound word
                    subdivisions = subdivide(text, start_time, end_time)
                    for w, s, e in subdivisions:
                        word_labels.append((w, s, e))
                else:
                    text = text.replace(" ", "")
                    if not text:
                        print(f"Empty annotation.")
                        continue
                    word_labels.append((text, start_time, end_time))

    return word_labels


def find_triple_repeating_words(word_list):
    """Finds all triple repeating words in a sorted list"""
    triple_repeating_words = []
    double_repeating_words = []

    prev_word = None
    repeat_count = 1

    for word, start_time, end_time in word_list:
        if word == prev_word:
            repeat_count += 1
        else:
            repeat_count = 1

        if repeat_count == 2:
            double_repeating_words.append((word, start_time, end_time))

        if repeat_count == 3:
            triple_repeating_words.append((word, start_time, end_time))

        prev_word = word

    return triple_repeating_words, double_repeating_words


def create_lyrics_from_labels(words: List[str], file_path: str):
    """Creates a text file with five words per line from a list of words"""
    with open(file_path, 'w') as file:
        for i in range(0, len(words), 5):
            line = ' '.join(words[i:i+5])
            file.write(line + '\n')


if __name__ == "__main__":
    textgrid_files = ["/Users/pushkarjajoria/Downloads/Music alignment/Violetta_2.TextGrid",
                      "/Users/pushkarjajoria/Downloads/Music alignment/Violetta_3_249_318.TextGrid",
                      "/Users/pushkarjajoria/Downloads/Music alignment/Violetta_4.TextGrid",
                      "/Users/pushkarjajoria/Downloads/Music alignment/gold_annotations_GC.TextGrid"]
    lyrics_file = "/Users/pushkarjajoria/Desktop/Violetta/inputs/violetta_text/violetta.txt"
    excel_file_path = "/Users/pushkarjajoria/Downloads/Music alignment/lyrics_diff.xlsx"
    time_stamps = [(180, 248),
                   (249, 318),
                   (319, 450),
                   (450, 570)]
    error_words = [('misterioso', 185.14184, 185.17501),
                   ]
    """
    Processing steps:
    1. Combine .ass annotation, followed by the text grid files.
    2. Sort them based on starting time
    3. Resolve conflicts by changing the start time such that it's strictly greater than the previous end time.
        If the resulting interval is smaller than 0.1, change the end time such that the resulting interval is at least 
        0.1 seconds.
    4. Clean each word in this list to remove punctuations that are not valid. Round the time to 2 decimal points.
    """
    word_tuples = combine_textgrid_files(textgrid_files)
    word_tuples = sorted(word_tuples, key=lambda x: x[1])
    resolved_word_tuples = resolve_conflicts(word_tuples)
    final_labels = list(map(lambda x: (clean_words(x[0]), round(x[1], 2), round(x[2], 2)), resolved_word_tuples))

    # "Save Excel file for side-by-side comparison"
    # tg_word_list = list(map(lambda x: clean_words(x[0]), resolved_word_tuples))
    # lyrics_words = read_lyrics_file(lyrics_file)
    # create_excel_file(tg_word_list, lyrics_words, excel_file_path)
    # print(len(tg_word_list))
    # print(len(lyrics_words))

    # "Saving textgrid file"
    # filepath = "/Users/pushkarjajoria/Downloads/Music alignment/Violetta_combined.TextGrid"
    # combined_tg = create_textgrid(resolved_word_tuples, tier_name="combined text_grid")
    # with open(filepath, 'w') as handle:
    #     combined_tg.write(handle)

    "Saving a tsv file of labels"
    filepath = "/Users/pushkarjajoria/Git/pushkarjajoria/lyrics-aligner/dataset/aria_violetta/labels.tsv"
    create_tsv_file(final_labels, filepath)

    # "Print double and triple repeating words"
    # triple, double = find_triple_repeating_words(final_labels)
    # print(double)
    # print(triple)

    create_lyrics_from_labels(list(map(lambda x: x[0], final_labels)), "/Users/pushkarjajoria/Git/pushkarjajoria/lyrics-aligner/dataset/aria_violetta/lyrics.txt")
