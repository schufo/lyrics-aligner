import unittest
import os


class TestLyricsLabelsConsistency(unittest.TestCase):

    def test_consistency(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
        datapoints = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        for datapoint in datapoints:
            lyrics_path = os.path.join(base_dir, datapoint, 'text', 'song.txt')
            labels_path = os.path.join(base_dir, datapoint, 'labels.tsv')

            if not os.path.exists(lyrics_path):
                self.fail(f"Lyrics file not found: {lyrics_path}")
            if not os.path.exists(labels_path):
                self.fail(f"Labels file not found: {labels_path}")

            lyrics_words = load_lyrics_from_txt(lyrics_path)
            label_texts = load_labels_from_tsv(labels_path)

            if len(lyrics_words) != len(label_texts) or any(lw != lt for lw, lt in zip(lyrics_words, label_texts)):
                mismatch_positions = [(i, lw, lt) for i, (lw, lt) in enumerate(zip(lyrics_words, label_texts)) if
                                      lw != lt]
                error_details = '\n'.join(
                    [f"Position {i}: Lyrics='{lw}', Label='{lt}'" for i, lw, lt in mismatch_positions])

                error_msg = (f"\nOrder/content mismatch found for {datapoint}:\n"
                             f"Lyrics file: {lyrics_path}\n"
                             f"Labels file: {labels_path}\n\n"
                             f"Details:\n{error_details}")
                self.fail(error_msg)


def load_lyrics_from_txt(filepath):
    with open(filepath, 'r') as f:
        return [word for line in f for word in line.strip().split()]


def load_labels_from_tsv(filepath):
    with open(filepath, 'r') as f:
        return [line.strip().split('\t')[0] for idx, line in enumerate(f) if
                line.strip() and idx > 0]  # Skipping header


if __name__ == "__main__":
    unittest.main()
