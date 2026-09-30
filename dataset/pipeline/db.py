"""
Shared SQLite access for the dataset pipeline (dataset/pipeline/pipeline.db)
and the class_registry.json reader. Every pipeline script (ingest,
sam3_quality_gate, review_bridge, build_training_batch, promote_if_better,
auto_train_trigger) imports from here instead of touching the schema
directly, so the "what classes exist" and "what's the frame lifecycle"
logic only lives in one place.

Frame lifecycle (one row per (captured image, pipeline class) pair -- a
frame is checked against every non-LOCO class the registry knows about, so
adding a class later needs no schema change):
  sam3_status:    pending -> pass | reject
  review_status:  NULL (while sam3_status != 'pass') -> pending -> approved | rejected
      -> held_out_test (build_fixed_testset.py carves a fixed slice of the
      approved pool out into dataset/pipeline/fixed_testset/ permanently;
      this status exists only so approved_pool() naturally excludes those
      rows forever, without losing the row's history the way a DELETE
      would)
  incorporated_in_batch: NULL until the first build_training_batch.py run
      after approval considers it (does NOT mean "consumed forever" -- see
      build_training_batch.py, approved images stay eligible for replay
      sampling in future rounds too).
"""

import json
import os
import sqlite3

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.dirname(PIPELINE_ROOT)
PROJECT_ROOT = os.path.dirname(DATASET_ROOT)

DB_PATH = os.path.join(PIPELINE_ROOT, "pipeline.db")
CLASS_REGISTRY_PATH = os.path.join(PIPELINE_ROOT, "class_registry.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    robot_id TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    local_path TEXT NOT NULL,
    class_name TEXT NOT NULL,
    blur_score REAL,
    sam3_mean_confidence REAL,
    sam3_status TEXT NOT NULL DEFAULT 'pending',
    review_status TEXT,
    reviewed_at TEXT,
    incorporated_in_batch TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_sam3_status ON frames (sam3_status);
CREATE INDEX IF NOT EXISTS idx_frames_review_status ON frames (review_status, class_name);
CREATE INDEX IF NOT EXISTS idx_frames_incorporated ON frames (incorporated_in_batch, class_name);
"""


def connect(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def load_class_registry(path=CLASS_REGISTRY_PATH):
    with open(path) as f:
        return json.load(f)["classes"]


def pipeline_classes(registry=None):
    """Classes discovered/added through this pipeline (not the original LOCO 5)."""
    registry = registry if registry is not None else load_class_registry()
    return [c for c in registry if c["source"] == "pipeline"]


def loco_classes(registry=None):
    registry = registry if registry is not None else load_class_registry()
    return [c for c in registry if c["source"] == "loco"]


def class_by_name(name, registry=None):
    registry = registry if registry is not None else load_class_registry()
    for c in registry:
        if c["name"] == name:
            return c
    raise KeyError(f"'{name}' is not in class_registry.json")


def frame_id(robot_id, captured_at, class_name):
    return f"{robot_id}_{captured_at}_{class_name}"


def count_unincorporated_approved(conn, class_name):
    """Approved rows for this class not yet considered by any
    build_training_batch.py round (NOT the same as "awaiting human
    review" -- see count_pending_review_all for that)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM frames WHERE review_status = 'approved' AND incorporated_in_batch IS NULL "
        "AND class_name = ?",
        (class_name,),
    ).fetchone()
    return row[0]


def count_unincorporated_approved_total(conn, registry=None):
    """Summed across every pipeline class -- what auto_train_trigger.py
    compares against APPROVAL_THRESHOLD to decide whether a new round is
    worth kicking off."""
    return sum(count_unincorporated_approved(conn, c["name"]) for c in pipeline_classes(registry))


def approved_pool(conn, class_name):
    """Every approved row for this class, ever -- eligible for replay sampling
    regardless of incorporated_in_batch (see module docstring).

    ORDER BY id is required, not cosmetic: build_training_batch.py shuffles
    this list with a fixed random.seed(SEED) expecting a reproducible split
    across rounds -- without a deterministic base order, SQLite doesn't
    guarantee row order for an unordered SELECT, so the same seed could
    silently shuffle a differently-ordered list each run."""
    return conn.execute(
        "SELECT id, local_path FROM frames WHERE class_name = ? AND review_status = 'approved' ORDER BY id",
        (class_name,),
    ).fetchall()


def get_next_pending_review(conn):
    """Oldest not-yet-reviewed row that passed the SAM3 gate, across every
    class -- review_bridge.py shows reviewers one queue, not one per class."""
    row = conn.execute(
        "SELECT id, robot_id, captured_at, local_path, class_name, blur_score, sam3_mean_confidence "
        "FROM frames WHERE sam3_status = 'pass' AND review_status = 'pending' "
        "ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    keys = ("id", "robot_id", "captured_at", "local_path", "class_name", "blur_score", "sam3_mean_confidence")
    return dict(zip(keys, row))


def count_pending_review_all(conn):
    return conn.execute(
        "SELECT COUNT(*) FROM frames WHERE sam3_status = 'pass' AND review_status = 'pending'"
    ).fetchone()[0]


def set_review_status(conn, row_id, status, new_local_path=None):
    """new_local_path: pass when the reviewer's decision also moved the file
    on disk (approve_current moves it into approved_pending_training/<class>/)
    so local_path always reflects where the file actually is."""
    import datetime as _dt
    if new_local_path is not None:
        conn.execute(
            "UPDATE frames SET review_status = ?, reviewed_at = ?, local_path = ? WHERE id = ?",
            (status, _dt.datetime.now().isoformat(), new_local_path, row_id),
        )
    else:
        conn.execute(
            "UPDATE frames SET review_status = ?, reviewed_at = ? WHERE id = ?",
            (status, _dt.datetime.now().isoformat(), row_id),
        )
    conn.commit()


def mark_incorporated(conn, ids, batch_name):
    conn.executemany(
        "UPDATE frames SET incorporated_in_batch = ? WHERE id = ?",
        [(batch_name, i) for i in ids],
    )
    conn.commit()
