#
# Edit a target object's trajectory to create a collision scenario and export an animated plot.
#
import json
import math
from argparse import ArgumentParser

import imageio
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.traj_schema import (is_scene_traj, resolve_states_ref, sample_positions_by_time,
                               select_track, state_to_pos, state_to_time)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _sorted_frames(frames_dict):
    return sorted(frames_dict.keys(), key=lambda x: int(x))


def _last_point(points):
    if not points:
        return None
    return np.array(points[-1], dtype=np.float32)


def _extract_per_frame_positions(frames, obj_frames, obj_id):
    positions = []
    for frame_id in frames:
        entry = obj_frames.get(frame_id, {})
        others = entry.get("others", {})
        pts = others.get(obj_id)
        positions.append(_last_point(pts) if pts else None)
    return positions


def _collect_obj_ids(obj_frames):
    obj_ids = set()
    for entry in obj_frames.values():
        others = entry.get("others", {})
        for obj_id in others.keys():
            obj_ids.add(str(obj_id))
    return sorted(obj_ids, key=lambda x: int(x) if x.isdigit() else x)


def _extract_all_obj_positions(frames, obj_frames, obj_ids):
    positions = {}
    for obj_id in obj_ids:
        positions[obj_id] = _extract_per_frame_positions(frames, obj_frames, obj_id)
    return positions


def _extract_ego_positions(frames, ego_frames):
    positions = []
    for frame_id in frames:
        entry = ego_frames.get(frame_id, {})
        pts = entry.get("ego")
        positions.append(_last_point(pts) if pts else None)
    return positions


def _extract_state_series(states, dt):
    if not states:
        return [], []
    times = []
    positions = []
    for idx, state in enumerate(states):
        fallback = float(idx)
        if dt is not None:
            fallback = float(idx) * dt
        t_val = state_to_time(state, fallback)
        pos = state_to_pos(state)
        times.append(float(t_val))
        positions.append(pos)
    return times, positions


def _resolve_time_index(frame_times, key):
    try:
        value = float(key)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid frame key '{key}'. Use index or time value.")
    if float(value).is_integer():
        idx = int(value)
        if 0 <= idx < len(frame_times):
            return idx
    if not frame_times:
        raise ValueError("No frame times available.")
    deltas = np.abs(np.array(frame_times, dtype=np.float32) - float(value))
    idx = int(np.argmin(deltas))
    if deltas[idx] > 1e-4:
        raise ValueError(f"Frame time '{key}' not found.")
    return idx


def _find_agent(scene_payload, agent_id):
    for agent in scene_payload.get("agents", []):
        if str(agent.get("agent_id")) == str(agent_id):
            return agent
    return None


def _set_state_pose(state, pos):
    if state is None or pos is None:
        return
    target = None
    if isinstance(state.get("pose"), dict):
        target = state["pose"]
    elif isinstance(state.get("position"), dict):
        target = state["position"]
    else:
        target = {}
        state["pose"] = target
    target["x"] = float(pos[0])
    target["y"] = float(pos[1])
    target["z"] = float(pos[2]) if len(pos) > 2 else 0.0


def _axis_indices(axes):
    if axes == "xy":
        return 0, 1
    if axes == "xz":
        return 0, 2
    if axes == "yz":
        return 1, 2
    raise ValueError(f"Unsupported axes: {axes}")


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _modify_positions(orig_positions, ego_positions, start_idx, impact_idx,
                      impact_offset, keep_z, method, post_impact):
    new_positions = []
    impact_ego = ego_positions[impact_idx]
    if impact_ego is None:
        raise ValueError("Ego position missing at impact frame.")
    target = impact_ego + impact_offset
    for i, pos in enumerate(orig_positions):
        if pos is None:
            new_positions.append(None)
            continue
        if i < start_idx:
            new_p = pos
        elif i <= impact_idx:
            t = (i - start_idx) / max(1, (impact_idx - start_idx))
            if method == "smoothstep":
                t = _smoothstep(t)
            new_p = (1.0 - t) * pos + t * target
        else:
            if post_impact == "hold":
                new_p = target.copy()
            else:
                new_p = pos
        if keep_z:
            new_p[2] = pos[2]
        new_positions.append(new_p)
    return new_positions


def _rebuild_cumulative_positions(positions):
    cumulative = []
    out = []
    for pos in positions:
        if pos is not None:
            cumulative.append(pos)
        out.append([p.tolist() for p in cumulative])
    return out


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


def _render_animation(frames, ego_positions, obj_positions_map, target_id, axes, output_path,
                      fps, show_all, collision_dist):
    ax_i, ax_j = _axis_indices(axes)
    ego_xy = [p[[ax_i, ax_j]] if p is not None else None for p in ego_positions]
    obj_xy_map = {}
    for obj_id, pos_list in obj_positions_map.items():
        obj_xy_map[obj_id] = [p[[ax_i, ax_j]] if p is not None else None for p in pos_list]
    all_paths = [ego_xy] + list(obj_xy_map.values())
    xlim, ylim = _compute_bounds(all_paths)

    images = []
    for i, frame_id in enumerate(frames):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.set_title(f"Frame {frame_id}")
        ax.set_xlabel(axes[0])
        ax.set_ylabel(axes[1])

        ego_path = [p for p in ego_xy[:i + 1] if p is not None]
        if ego_path:
            ego_path = np.stack(ego_path, axis=0)
            ax.plot(ego_path[:, 0], ego_path[:, 1], color="red", linewidth=2, label="ego")
            ax.scatter(ego_path[-1, 0], ego_path[-1, 1], color="red", s=40)

        for obj_id, obj_xy in obj_xy_map.items():
            obj_path = [p for p in obj_xy[:i + 1] if p is not None]
            if not obj_path:
                continue
            obj_path = np.stack(obj_path, axis=0)
            if obj_id == target_id:
                ax.plot(obj_path[:, 0], obj_path[:, 1], color="blue", linewidth=2, label="target")
                ax.scatter(obj_path[-1, 0], obj_path[-1, 1], color="blue", s=36)
            else:
                ax.plot(obj_path[:, 0], obj_path[:, 1], color="gray", linewidth=1, alpha=0.4)
                ax.scatter(obj_path[-1, 0], obj_path[-1, 1], color="gray", s=16, alpha=0.5)

        if collision_dist > 0 and target_id in obj_xy_map:
            ego_p = ego_positions[i]
            obj_p = obj_positions_map[target_id][i]
            if ego_p is not None and obj_p is not None:
                dist = np.linalg.norm(ego_p - obj_p)
                if dist <= collision_dist:
                    mid = (ego_p[[ax_i, ax_j]] + obj_p[[ax_i, ax_j]]) * 0.5
                    ax.scatter(mid[0], mid[1], color="gold", s=80, marker="x")
                    ax.text(mid[0], mid[1], "collision", color="gold", fontsize=9)

        if show_all:
            ax.legend(loc="best")

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        images.append(image)
        plt.close(fig)

    imageio.mimsave(output_path, images, fps=fps)


def main():
    parser = ArgumentParser(description="Edit trajectory and render animated plot.")
    parser.add_argument("--ego_json", type=str, default="ego_traj.json")
    parser.add_argument("--obj_json", type=str, default="obj_traj.json")
    parser.add_argument("--out_obj_json", type=str, default="obj_traj_modified.json")
    parser.add_argument("--obj_id", type=str, default="3")
    parser.add_argument("--start_frame", type=str, default="015")
    parser.add_argument("--impact_frame", type=str, default="027")
    parser.add_argument("--impact_offset", type=str, default="0,0,0")
    parser.add_argument("--axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--keep_z", action="store_true")
    parser.add_argument("--method", choices=["linear", "smoothstep"], default="smoothstep")
    parser.add_argument("--post_impact", choices=["hold", "continue"], default="hold")
    parser.add_argument("--fill_missing", action="store_true")
    parser.add_argument("--only_target", action="store_true")
    parser.add_argument("--collision_dist", type=float, default=0.15)
    parser.add_argument("--output_anim", type=str, default="traj_anim.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--show_all", action="store_true")
    parser.add_argument("--ego_track", type=str, default=None)
    parser.add_argument("--obj_track", type=str, default=None)
    parser.add_argument("--pred_mode", type=str, default=None)
    args = parser.parse_args()

    ego = _load_json(args.ego_json)
    obj = _load_json(args.obj_json)
    scene_payload = None
    if is_scene_traj(ego):
        scene_payload = ego
    elif is_scene_traj(obj):
        scene_payload = obj

    if scene_payload:
        time_base = scene_payload.get("time_base") or {}
        dt = time_base.get("dt")
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            dt = None
        ego_agent = _find_agent(scene_payload, "ego")
        if ego_agent is None:
            raise ValueError("ego agent not found in scene trajectory.")
        obj_agent = _find_agent(scene_payload, args.obj_id)
        if obj_agent is None:
            raise ValueError(f"obj_id '{args.obj_id}' not found in scene trajectory.")

        ego_pref = [args.ego_track] if args.ego_track else ["history", "observed", "planned", "predicted"]
        obj_pref = [args.obj_track] if args.obj_track else ["observed", "history", "predicted", "planned"]
        _, ego_track_entry = select_track(ego_agent.get("tracks", {}), ego_pref)
        _, obj_track_entry = select_track(obj_agent.get("tracks", {}), obj_pref)
        ego_states = resolve_states_ref(ego_track_entry, pred_mode=args.pred_mode)
        obj_states = resolve_states_ref(obj_track_entry, pred_mode=args.pred_mode)
        if not ego_states or not obj_states:
            raise ValueError("Missing states for ego or target object track.")

        frame_times, obj_positions = _extract_state_series(obj_states, dt)
        if not frame_times:
            raise ValueError("Target object track has no timestamps.")
        ego_times, ego_positions_raw = _extract_state_series(ego_states, dt)
        ego_series = [
            (t_val, pos)
            for t_val, pos in zip(ego_times, ego_positions_raw)
            if pos is not None
        ]
        ego_positions = sample_positions_by_time(ego_series, frame_times)
        frames = [f"{t:.3f}" for t in frame_times]

        start_idx = _resolve_time_index(frame_times, args.start_frame)
        impact_idx = _resolve_time_index(frame_times, args.impact_frame)

        if args.fill_missing:
            last = None
            for i, p in enumerate(obj_positions):
                if p is None and last is not None:
                    obj_positions[i] = last.copy()
                elif p is not None:
                    last = p
    else:
        frames = _sorted_frames(ego["frames"])
        if args.start_frame not in frames or args.impact_frame not in frames:
            raise ValueError("start_frame/impact_frame not found in ego frames.")

        ego_positions = _extract_ego_positions(frames, ego["frames"])
        obj_ids = _collect_obj_ids(obj["frames"])
        if args.obj_id not in obj_ids:
            raise ValueError(f"obj_id '{args.obj_id}' not found in obj_traj.json")
        obj_positions_map = _extract_all_obj_positions(frames, obj["frames"], obj_ids)
        obj_positions = obj_positions_map[args.obj_id]

    if not scene_payload:
        if args.fill_missing:
            last = None
            for i, p in enumerate(obj_positions):
                if p is None and last is not None:
                    obj_positions[i] = last.copy()
                elif p is not None:
                    last = p
            obj_positions_map[args.obj_id] = obj_positions

        start_idx = frames.index(args.start_frame)
        impact_idx = frames.index(args.impact_frame)

    impact_offset = np.array([float(x) for x in args.impact_offset.split(",")], dtype=np.float32)

    new_positions = _modify_positions(
        obj_positions, ego_positions, start_idx, impact_idx,
        impact_offset, args.keep_z, args.method, args.post_impact
    )
    if scene_payload:
        for state, pos in zip(obj_states, new_positions):
            if pos is not None:
                _set_state_pose(state, pos)
        if args.only_target:
            obj_positions_map = {args.obj_id: new_positions}
        else:
            obj_positions_map = {}
            for agent in scene_payload.get("agents", []):
                agent_id = str(agent.get("agent_id"))
                if agent_id == "ego":
                    continue
                if agent_id == args.obj_id:
                    obj_positions_map[agent_id] = new_positions
                    continue
                _, track_entry = select_track(agent.get("tracks", {}), obj_pref)
                states = resolve_states_ref(track_entry, pred_mode=args.pred_mode)
                if not states:
                    continue
                other_times, other_positions = _extract_state_series(states, dt)
                series = [
                    (t_val, pos)
                    for t_val, pos in zip(other_times, other_positions)
                    if pos is not None
                ]
                obj_positions_map[agent_id] = sample_positions_by_time(series, frame_times)

        _save_json(args.out_obj_json, scene_payload)
    else:
        obj_positions_map[args.obj_id] = new_positions

        cumulative = _rebuild_cumulative_positions(new_positions)
        for i, frame_id in enumerate(frames):
            entry = obj["frames"].setdefault(frame_id, {"others": {}})
            entry.setdefault("others", {})
            entry["others"][args.obj_id] = cumulative[i]

        _save_json(args.out_obj_json, obj)
        if args.only_target:
            obj_positions_map = {args.obj_id: new_positions}
    _render_animation(
        frames, ego_positions, obj_positions_map, args.obj_id, args.axes,
        args.output_anim, args.fps, args.show_all, args.collision_dist
    )

    print(f"Saved updated obj traj to {args.out_obj_json}")
    print(f"Saved animation to {args.output_anim}")


if __name__ == "__main__":
    main()
