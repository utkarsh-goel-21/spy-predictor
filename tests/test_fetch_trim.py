import unittest

import pandas as pd

from backend.data.fetch import trim_df_to_period


class FetchTrimTests(unittest.TestCase):
    def test_trim_df_to_period_keeps_recent_window(self):
        idx = pd.date_range("2026-01-01", periods=120, freq="B")
        df = pd.DataFrame({"Close": range(len(idx))}, index=idx)

        trimmed = trim_df_to_period(df, "3mo")

        self.assertGreaterEqual(len(trimmed), 50)
        self.assertLess(len(trimmed), len(df))
        self.assertEqual(trimmed.index.max(), df.index.max())


if __name__ == "__main__":
    unittest.main()
