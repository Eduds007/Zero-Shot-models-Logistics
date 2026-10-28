"""
Scores the manual review exported from the GT vs SAM3 viewer
(build_sam3_gt_review.py -> index.html -> "Exportar CSV").

Only images whose every FP/FN was tagged count, so a partially reviewed image
can't bias the result; since the image order in the viewer is the random
sample order, reviewing images 1..N gives a random subset of images. Reports,
per class and overall:

- automatic precision/recall against the raw LOCO GT
- corrected precision/recall: FPs tagged 'missing_gt' count as TPs (the
  object is real, LOCO just didn't annotate it); FNs tagged 'gt_wrong' are
  dropped from the GT
- 95% bootstrap CI over images for the corrected numbers
- breakdown of the FP/FN tags

Usage (from dataset/):
    python score_sam3_gt_review.py sam3_gt_review_seed42.csv [--review-dir sam3_gt_review]
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict


def load_tp_counts(review_dir):
    with open(os.path.join(review_dir, "data.js")) as f:
        data = json.loads(f.read().split("=", 1)[1].rstrip().rstrip(";"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    tp = defaultdict(Counter)  # image_id -> class -> TP count
    for im in data["images"]:
        for d in im["det"]:
            if d["status"] == "TP":
                tp[str(im["id"])][names[d["cat"]]] += 1
    return tp, list(names.values()), len(data["images"])


def counts(image_ids, tp, rows_by_image, cls=None):
    c = Counter()
    for iid in image_ids:
        c["tp"] += sum(v for k, v in tp[iid].items() if cls in (None, k))
        for r in rows_by_image[iid]:
            if cls not in (None, r["category"]):
                continue
            c[r["kind"]] += 1
            c[r["tag"]] += 1
    return c


def pr(c):
    tp, fp, fn = c["tp"], c["FP"], c["FN"]
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    tp_c = tp + c["missing_gt"]
    p_c = tp_c / (tp + fp) if tp + fp else float("nan")
    r_c = tp_c / (tp_c + fn - c["gt_wrong"]) if tp_c + fn - c["gt_wrong"] else float("nan")
    return p, r, p_c, r_c


def bootstrap(image_ids, tp, rows_by_image, cls, n=2000, seed=0):
    rng = random.Random(seed)
    ps, rs = [], []
    for _ in range(n):
        sample = [rng.choice(image_ids) for _ in image_ids]
        _, _, p_c, r_c = pr(counts(sample, tp, rows_by_image, cls))
        ps.append(p_c)
        rs.append(r_c)

    def ci(xs):
        xs = sorted(x for x in xs if x == x)
        return (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs else (float("nan"),) * 2
    return ci(ps), ci(rs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--review-dir", default="sam3_gt_review")
    args = parser.parse_args()

    tp, classes, n_images = load_tp_counts(args.review_dir)
    rows_by_image = defaultdict(list)
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            rows_by_image[r["image_id"]].append(r)

    done = [iid for iid in tp.keys() | rows_by_image.keys()
            if all(r["tag"] for r in rows_by_image[iid])]
    n_tagged = sum(1 for rs in rows_by_image.values() for r in rs if r["tag"])
    print(f"{len(done)}/{n_images} images fully reviewed ({n_tagged} tagged items in total)\n")
    if not done:
        return

    pct = lambda x: f"{100 * x:5.1f}%" if x == x else "   – "
    rng = lambda lo, hi: f"{pct(lo).strip()}-{pct(hi).strip()}" if lo == lo else "–"
    print(f"{'class':<20}{'TP':>6}{'FP':>6}{'FN':>6}   {'P auto':>7} {'P corr':>7} {'95% CI':>15}"
          f"   {'R auto':>7} {'R corr':>7} {'95% CI':>15}")
    for cls in classes + [None]:
        c = counts(done, tp, rows_by_image, cls)
        p, r, p_c, r_c = pr(c)
        (pl, ph), (rl, rh) = bootstrap(done, tp, rows_by_image, cls)
        print(f"{cls or 'OVERALL':<20}{c['tp']:>6}{c['FP']:>6}{c['FN']:>6}   {pct(p)} {pct(p_c)} "
              f"{rng(pl, ph):>15}   {pct(r)} {pct(r_c)} {rng(rl, rh):>15}")

    for kind in ("FP", "FN"):
        tags = Counter(r["tag"] for iid in done for r in rows_by_image[iid] if r["kind"] == kind)
        total = sum(tags.values())
        print(f"\n{kind} tags ({total}):")
        for tag, n in tags.most_common():
            print(f"  {tag:<14}{n:>6}  {pct(n / total)}")


if __name__ == "__main__":
    main()
