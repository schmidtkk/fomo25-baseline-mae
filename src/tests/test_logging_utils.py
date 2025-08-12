import os
import csv
import tempfile

from utils.logging_utils import write_subject_csv


def test_write_subject_csv_creates_file_with_rows():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "subj", "val_subject_probs_epoch_0001.csv")
        probs = {"A": 0.7, "B": 0.2}
        tgts = {"A": 1, "B": 0}
        write_subject_csv(out, probs, tgts, epoch=1)

        assert os.path.isfile(out)
        with open(out, "r") as f:
            rows = list(csv.reader(f))
        # header + 2 rows
        assert len(rows) == 3
        assert rows[0] == ["subject_id", "prob_pos", "target", "epoch"]
        ids = {rows[1][0], rows[2][0]}
        assert ids == {"A", "B"}


