"""Shared arg parsing for the fault fixtures. Mirrors the pipeline CLI contract."""
import argparse


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--subsample", type=float, default=None)
    return p.parse_args()


def write_submission(out_dir, rows, header="row_id,user_id,video_id,score",
                     score=lambda i: 0.5):
    import os
    with open(os.path.join(out_dir, "submission.csv"), "w") as fh:
        fh.write(header + "\n")
        for i in range(rows):
            fh.write("%d,u%d,v%d,%s\n" % (i, i % 7, i, score(i)))
