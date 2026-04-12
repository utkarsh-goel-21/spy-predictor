import unittest

import numpy as np

from backend.models.lstm_train import SPYDataset


class SPYDatasetAlignmentTests(unittest.TestCase):
    def test_sequence_label_matches_last_row_target(self):
        X = np.arange(12, dtype=float).reshape(-1, 1)
        y = np.arange(12)
        dataset = SPYDataset(X, y, seq_len=10)

        self.assertEqual(len(dataset), 3)

        first_seq, first_label = dataset[0]
        self.assertEqual(int(first_seq[0][0].item()), 0)
        self.assertEqual(int(first_seq[-1][0].item()), 9)
        self.assertEqual(int(first_label.item()), 9)

        last_seq, last_label = dataset[2]
        self.assertEqual(int(last_seq[0][0].item()), 2)
        self.assertEqual(int(last_seq[-1][0].item()), 11)
        self.assertEqual(int(last_label.item()), 11)


if __name__ == "__main__":
    unittest.main()
