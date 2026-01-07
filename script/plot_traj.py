#
# Visualize ego/objects trajectories with IDs and export an animated plot.
#
import json
import os
import sys
from argparse import ArgumentParser

import imageio
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.traj_schema import (collect_frame_times, extract_scene_tracks, is_scene_traj,
                               sample_positions_by_time)


def _load_json(path):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _sorted_frames(frames_dict):
    return sorted(frames_dict.keys(), key=lambda x: int(x))


def _last_point(points):
    if not points:
        return None
    return np.array(points[-1], dtype=np.float32)


def _collect_obj_ids(obj_frames):
    obj_ids = set()
    for entry in obj_frames.values():
        others = entry.get("others", {})
        for obj_id in others.keys():
            obj_ids.add(str(obj_id))
    return sorted(obj_ids, key=lambda x: int(x) if x.isdigit() else x)


def _extract_ego_positions(frames, ego_frames):
    positions = []
    for frame_id in frames:
        entry = ego_frames.get(frame_id, {})
        pts = entry.get("ego")
        positions.append(_last_point(pts) if pts else None)
    return positions


def _extract_obj_positions(frames, obj_frames, obj_ids):
    positions = {obj_id: [] for obj_id in obj_ids}
    for frame_id in frames:
        entry = obj_frames.get(frame_id, {})
        others = entry.get("others", {})
        for obj_id in obj_ids:
            pts = others.get(obj_id)
            positions[obj_id].append(_last_point(pts) if pts else None)
    return positions


def _axis_indices(axes):
    if axes == "xy":
        return 0, 1
    if axes == "xz":
        return 0, 2
    if axes == "yz":
        return 1, 2
    raise ValueError(f"Unsupported axes: {axes}")


def _compute_bounds(paths, pad=0.05):
    xs, ys = [], []
    for path in paths:
        for p in path:
            if p is None:
                continue
            xs.append(p[0])
            ys.append(p[1])
    if not xs or not ys:
        return (-1, 1), (-1, 1)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max_x - min_x
    dy = max_y - min_y
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0
    return (min_x - dx * pad, max_x + dx * pad), (min_y - dy * pad, max_y + dy * pad)


def _load_scene_positions(scene_payload, ego_track, obj_track, pred_mode):
    _, ego_series, obj_series, _ = extract_scene_tracks(
        scene_payload,
        ego_track=ego_track,
        obj_track=obj_track,
        pred_mode=pred_mode,
    )
    frame_times = collect_frame_times(ego_series, obj_series)
    ego_positions = sample_positions_by_time(ego_series, frame_times)
    obj_positions_map = {
        obj_id: sample_positions_by_time(series, frame_times)
        for obj_id, series in obj_series.items()
    }
    frame_labels = [f"{t:.3f}" for t in frame_times]
    return frame_labels, ego_positions, obj_positions_map


def main():
    parser = ArgumentParser(description="Plot trajectories with IDs.")
    parser.add_argument("--ego_json", type=str, default=None)
    parser.add_argument("--obj_json", type=str, default=None)
    parser.add_argument("--traj_json", type=str, default=None)
    parser.add_argument("--output", type=str, default="traj_overview.gif")
    parser.add_argument("--axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--trail_length", type=int, default=0)
    parser.add_argument("--show_history", action="store_true", help="显示他车轨迹历史（默认）。")
    parser.add_argument("--no_show_history", action="store_false", dest="show_history",
                        help="隐藏他车轨迹历史。")
    parser.add_argument("--label_ego", action="store_true", help="显示 ego 标号。")
    parser.add_argument("--label_objects", action="store_true", help="显示物体 ID（默认）。")
    parser.add_argument("--no_label_objects", action="store_false", dest="label_objects",
                        help="隐藏物体 ID。")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--ego_track", type=str, default=None)
    parser.add_argument("--obj_track", type=str, default=None)
    parser.add_argument("--pred_mode", type=str, default=None)
    parser.set_defaults(label_objects=True, show_history=True)
    args = parser.parse_args()

    scene_payload = None
    if args.traj_json:
        scene_payload = _load_json(args.traj_json)
        if not is_scene_traj(scene_payload):
            raise ValueError("traj_json must be a scene trajectory JSON with an agents list.")
    else:
        if not args.ego_json or not args.obj_json:
            raise ValueError("Provide --traj_json or both --ego_json and --obj_json.")
        ego = _load_json(args.ego_json)
        obj = _load_json(args.obj_json)
        if is_scene_traj(ego) or is_scene_traj(obj):
            scene_payload = ego if is_scene_traj(ego) else obj

    if scene_payload:
        frames, ego_positions, obj_positions_map = _load_scene_positions(
            scene_payload, args.ego_track, args.obj_track, args.pred_mode
        )
        if args.frame_stride > 1:
            idxs = list(range(0, len(frames), args.frame_stride))
            frames = [frames[i] for i in idxs]
            ego_positions = [ego_positions[i] for i in idxs]
            obj_positions_map = {
                obj_id: [positions[i] for i in idxs]
                for obj_id, positions in obj_positions_map.items()
            }
    else:
        frames = _sorted_frames(ego["frames"])
        if args.frame_stride > 1:
            frames = frames[::args.frame_stride]

        obj_ids = _collect_obj_ids(obj["frames"])
        ego_positions = _extract_ego_positions(frames, ego["frames"])
        obj_positions_map = _extract_obj_positions(frames, obj["frames"], obj_ids)

    ax_i, ax_j = _axis_indices(args.axes)
    ego_xy = [p[[ax_i, ax_j]] if p is not None else None for p in ego_positions]
    obj_xy_map = {
        obj_id: [p[[ax_i, ax_j]] if p is not None else None for p in positions]
        for obj_id, positions in obj_positions_map.items()
    }
    all_paths = [ego_xy] + list(obj_xy_map.values())
    xlim, ylim = _compute_bounds(all_paths)

    writer = imageio.get_writer(args.output, fps=args.fps)
    for i, frame_id in enumerate(frames):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Frame {frame_id}")
        ax.set_xlabel(args.axes[0])
        ax.set_ylabel(args.axes[1])
        if args.grid:
            ax.grid(True, linewidth=0.5, alpha=0.4)

        ego_path = [p for p in ego_xy[:i + 1] if p is not None]
        if ego_path:
            ego_path = np.stack(ego_path, axis=0)
            ax.plot(ego_path[:, 0], ego_path[:, 1], color="red", linewidth=2)
            ax.scatter(ego_path[-1, 0], ego_path[-1, 1], color="red", s=40)
            if args.label_ego:
                ax.text(ego_path[-1, 0], ego_path[-1, 1], "ego", color="red", fontsize=9)

        for obj_id, obj_xy in obj_xy_map.items():
            if args.show_history:
                start = 0
                if args.trail_length > 0:
                    start = max(0, i + 1 - args.trail_length)
                obj_path = [p for p in obj_xy[start:i + 1] if p is not None]
                if obj_path:
                    obj_path = np.stack(obj_path, axis=0)
                    ax.plot(obj_path[:, 0], obj_path[:, 1], color="gray", linewidth=1, alpha=0.6)
            cur = obj_xy[i] if i < len(obj_xy) else None
            if cur is not None:
                ax.scatter(cur[0], cur[1], color="gray", s=16, alpha=0.8)
                if args.label_objects:
                    ax.text(cur[0], cur[1], obj_id, color="black", fontsize=8)

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        writer.append_data(image)
        plt.close(fig)

    writer.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
