"""
Scans dataset/incoming/<robot_id>/*.jpg (where sync_node.py on the robot
rsyncs captured frames) and registers new ones in pipeline.db, one row per
(image, pipeline class) pair -- every frame gets checked against every
class_registry.json class with source="pipeline", so adding a new target
class later needs no change here.

Run manually, or on a simple cron on the workstation (this side is assumed
to be reachable whenever it's powered on, unlike the robot).

Usage:
    python ingest_from_robot.py [--robot-id fml-22]
"""

import argparse
import glob
import os

import db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming-dir", default=os.path.join(db.DATASET_ROOT, "incoming"))
    parser.add_argument("--robot-id", default=None,
                         help="Ingest only this robot's subdirectory; default: every subdirectory found")
    args = parser.parse_args()

    conn = db.connect()
    classes = db.pipeline_classes(db.load_class_registry())
    if not classes:
        print("class_registry.json has no source=pipeline classes -- nothing to check frames against")
        return

    if args.robot_id:
        robot_dirs = [os.path.join(args.incoming_dir, args.robot_id)]
    else:
        robot_dirs = sorted(d for d in glob.glob(os.path.join(args.incoming_dir, "*")) if os.path.isdir(d))

    total_new = 0
    for robot_dir in robot_dirs:
        robot_id = os.path.basename(robot_dir)
        image_paths = sorted(glob.glob(os.path.join(robot_dir, "*.jpg")))
        new_for_robot = 0
        for path in image_paths:
            captured_at = os.path.splitext(os.path.basename(path))[0]
            for cls in classes:
                row_id = db.frame_id(robot_id, captured_at, cls["name"])
                exists = conn.execute("SELECT 1 FROM frames WHERE id = ?", (row_id,)).fetchone()
                if exists:
                    continue
                conn.execute(
                    "INSERT INTO frames (id, robot_id, captured_at, local_path, class_name, "
                    "sam3_status, created_at) VALUES (?, ?, ?, ?, ?, 'pending', datetime('now'))",
                    (row_id, robot_id, captured_at, path, cls["name"]),
                )
                new_for_robot += 1
        conn.commit()
        if image_paths:
            print(f"{robot_id}: {len(image_paths)} image(s) on disk, {new_for_robot} new row(s) ingested")
        total_new += new_for_robot

    print(f"\ntotal new rows ingested: {total_new}")


if __name__ == "__main__":
    main()
