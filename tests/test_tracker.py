from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "submit_and_record.py"
SPEC = importlib.util.spec_from_file_location("submit_and_record", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRACKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKER
SPEC.loader.exec_module(TRACKER)


SUCCESS_OUTPUT = """\
valid: submission.py (2742 bytes)
queued: 27071dce-9e87-4a51-800e-12c4b16b6f5b
tier:   easy
data:   e1
left:   59 today
view:   https://onelayerdeeper.ai/submissions/27071dce-9e87-4a51-800e-12c4b16b6f5b
[queued] submission.py
[running] submission.py
[succeeded] submission.py
submission  27071dce-9e87-4a51-800e-12c4b16b6f5b
file        submission.py
status      succeeded
score       3.83%
max T       <1
OOD N max T <1
tier        easy
dataset     E1 · Fixed N=323, T=1/2/3
suite       h100_easy_e1.json
run         977b46be-8f9c-4d6a-b4f4-89b17105d31c
modal call  fc-01KYAZXPSEF37TWAPXMKBN8HM5
"""


class ParseOutputTests(unittest.TestCase):
    def test_parses_complete_success(self) -> None:
        result = TRACKER.parse_output(SUCCESS_OUTPUT, "easy", "e1")

        self.assertTrue(result.valid)
        self.assertEqual(result.validated_bytes, 2742)
        self.assertTrue(result.queued)
        self.assertEqual(result.attempts_left, 59)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.score_pct, 3.83)
        self.assertEqual(result.max_t, "<1")
        self.assertEqual(result.ood_n_max_t, "<1")
        self.assertEqual(result.dataset_id, "e1")
        self.assertEqual(result.dataset_label, "E1 · Fixed N=323, T=1/2/3")
        self.assertEqual(result.suite, "h100_easy_e1.json")

    def test_preserves_failed_status(self) -> None:
        result = TRACKER.parse_output(
            "valid: submission.py (10 bytes)\nsubmission rejected: bad key\n",
            "medium",
            "m1",
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.queued)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.tier, "medium")
        self.assertEqual(result.dataset_id, "m1")


class ScoreCellTests(unittest.TestCase):
    def test_formats_missing_failed_and_successful_results(self) -> None:
        self.assertEqual(TRACKER.score_cell(None, None, None), "—")
        self.assertEqual(TRACKER.score_cell("failed", None, None), "failed")
        self.assertEqual(
            TRACKER.score_cell("succeeded", 7.83, "https://example.com/result"),
            "[7.83%](https://example.com/result)",
        )


if __name__ == "__main__":
    unittest.main()
