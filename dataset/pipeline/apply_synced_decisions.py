"""
Folds review decisions made on the robot (via review_node.py, uploaded by
review_sync_node.py into dataset/pipeline/review_decisions/<class>/{approved,rejected}/)
back into pipeline.db, and moves approved images+labels into
<class>'s replay_dir (approved_pending_training/<class>/) -- the same
destination build_training_batch.py already reads from.

Matches uploaded files back to DB rows by filename stem (captured_at is
embedded in the filename, e.g. capture_20260917_095438_918436.jpg), since
the robot's local queue doesn't carry the row id around.

Run automatically by review_sync_node.py after every upload; safe to run
manually too (idempotent -- already-processed files are gone from
review_decisions/ once handled).

Usage:
    python apply_synced_decisions.py
"""

import os
import shutil

import db

DECISIONS_ROOT = os.path.join(db.PIPELINE_ROOT, "review_decisions")


def main():
    conn = db.connect()
    registry = db.load_class_registry()
    n_approved = n_rejected = n_unmatched = 0

    for cls in db.pipeline_classes(registry):
        cls_name = cls["name"]
        for decision in ("approved", "rejected"):
            images_dir = os.path.join(DECISIONS_ROOT, cls_name, decision, "images")
            labels_dir = os.path.join(DECISIONS_ROOT, cls_name, decision, "labels")
            if not os.path.isdir(images_dir):
                continue

            for fname in sorted(os.listdir(images_dir)):
                stem = os.path.splitext(fname)[0]
                src_img = os.path.join(images_dir, fname)
                src_lbl = os.path.join(labels_dir, stem + ".txt")

                row = conn.execute(
                    "SELECT id FROM frames WHERE class_name = ? AND local_path LIKE ?",
                    (cls_name, f"%{stem}%"),
                ).fetchone()

                if row is None:
                    print(f"WARNING: no matching DB row for {cls_name}/{decision}/{fname} -- discarding")
                    os.remove(src_img)
                    if os.path.exists(src_lbl):
                        os.remove(src_lbl)
                    n_unmatched += 1
                    continue

                row_id = row[0]
                if decision == "approved":
                    dest_dir = os.path.join(db.PROJECT_ROOT, cls["replay_dir"])
                    dest_img = os.path.join(dest_dir, "images", fname)
                    dest_lbl = os.path.join(dest_dir, "labels", stem + ".txt")
                    os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                    os.makedirs(os.path.dirname(dest_lbl), exist_ok=True)
                    shutil.move(src_img, dest_img)
                    if os.path.exists(src_lbl):
                        shutil.move(src_lbl, dest_lbl)
                    db.set_review_status(conn, row_id, "approved", new_local_path=dest_img)
                    n_approved += 1
                else:
                    os.remove(src_img)
                    if os.path.exists(src_lbl):
                        os.remove(src_lbl)
                    db.set_review_status(conn, row_id, "rejected")
                    n_rejected += 1

    print(f"approved={n_approved} rejected={n_rejected} unmatched={n_unmatched}")


if __name__ == "__main__":
    main()
