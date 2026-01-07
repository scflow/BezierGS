#
# Edit a target object's trajectory to create a collision scenario and export an animated plot.
#
import json
import math
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

from utils.traj_schema import (is_scene_traj, resolve_states_ref, sample_positions_by_time,
                               select_multi_modal, select_track, state_to_pos, state_to_time)

EPS = 1e-6


def _load_json(path):
    if not path:
        return None
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
    if isinstance(key, str):
        stripped = key.strip()
        if stripped and ("." not in stripped and "e" not in stripped.lower()):
            if stripped.lstrip("+-").isdigit():
                idx = int(stripped)
                if 0 <= idx < len(frame_times):
                    return idx
    try:
        value = float(key)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid frame key '{key}'. Use index or time value.")
    if not frame_times:
        raise ValueError("No frame times available.")
    deltas = np.abs(np.array(frame_times, dtype=np.float32) - float(value))
    idx = int(np.argmin(deltas))
    if isinstance(key, str) and ("." in key or "e" in key.lower()):
        return idx
    if deltas[idx] > 1e-4:
        raise ValueError(f"Frame time '{key}' not found.")
    return idx


def _nearest_time_index(frame_times, time_val):
    if not frame_times:
        raise ValueError("No frame times available.")
    deltas = np.abs(np.array(frame_times, dtype=np.float32) - float(time_val))
    return int(np.argmin(deltas))


def _parse_time_or_index(frame_times, key):
    if isinstance(key, str):
        stripped = key.strip()
        if stripped and ("." in stripped or "e" in stripped.lower()):
            return True, float(stripped), None
        if stripped.lstrip("+-").isdigit():
            idx = int(stripped)
            if 0 <= idx < len(frame_times):
                return False, frame_times[idx], idx
    if isinstance(key, (int, np.integer)):
        idx = int(key)
        if 0 <= idx < len(frame_times):
            return False, frame_times[idx], idx
        return True, float(key), None
    value = float(key)
    if not float(value).is_integer():
        return True, value, None
    idx = int(value)
    if 0 <= idx < len(frame_times):
        return False, frame_times[idx], idx
    return True, float(value), None


def _time_in_list(times, t_val, tol=1e-6):
    return any(abs(t - t_val) <= tol for t in times)


def _clamp_time_range(value, t_min, t_max):
    return max(t_min, min(t_max, value))


def _find_agent(scene_payload, agent_id):
    for agent in scene_payload.get("agents", []):
        if str(agent.get("agent_id")) == str(agent_id):
            return agent
    return None


def _get_states_container(track_entry, pred_mode=None):
    if not isinstance(track_entry, dict):
        return None
    if isinstance(track_entry.get("states"), list):
        return track_entry
    trajectory = track_entry.get("trajectory")
    if isinstance(trajectory, dict):
        return trajectory
    modes = track_entry.get("multi_modal")
    if isinstance(modes, list):
        mode = select_multi_modal(modes, pred_mode=pred_mode)
        if isinstance(mode, dict):
            return mode
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


def _estimate_velocity(positions, times, idx):
    if not positions:
        return np.zeros(3, dtype=np.float32)
    pos = positions[idx]
    if pos is None:
        for j in range(idx - 1, -1, -1):
            if positions[j] is not None:
                pos = positions[j]
                idx = j
                break
    if pos is None:
        for j in range(idx + 1, len(positions)):
            if positions[j] is not None:
                pos = positions[j]
                idx = j
                break
    if pos is None:
        return np.zeros(3, dtype=np.float32)
    prev_idx = idx - 1
    next_idx = idx + 1
    while prev_idx >= 0 and positions[prev_idx] is None:
        prev_idx -= 1
    while next_idx < len(positions) and positions[next_idx] is None:
        next_idx += 1
    if prev_idx >= 0 and next_idx < len(positions):
        dt = float(times[next_idx] - times[prev_idx]) if times is not None else float(next_idx - prev_idx)
        if dt <= EPS:
            return np.zeros(3, dtype=np.float32)
        return (positions[next_idx] - positions[prev_idx]) / dt
    if prev_idx >= 0:
        dt = float(times[idx] - times[prev_idx]) if times is not None else float(idx - prev_idx)
        if dt <= EPS:
            return np.zeros(3, dtype=np.float32)
        return (pos - positions[prev_idx]) / dt
    if next_idx < len(positions):
        dt = float(times[next_idx] - times[idx]) if times is not None else float(next_idx - idx)
        if dt <= EPS:
            return np.zeros(3, dtype=np.float32)
        return (positions[next_idx] - pos) / dt
    return np.zeros(3, dtype=np.float32)


def _compute_quintic(p0, v0, a0, p1, v1, a1, t0, t1, t_vals):
    T = float(max(EPS, t1 - t0))
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    A = p1 - (c0 + c1 * T + c2 * T2)
    B = v1 - (c1 + 2.0 * c2 * T)
    C = a1 - (2.0 * c2)
    c3 = (10.0 * A - 4.0 * B * T + 0.5 * C * T2) / T3
    c4 = (-15.0 * A + 7.0 * B * T - C * T2) / T4
    c5 = (6.0 * A - 3.0 * B * T + 0.5 * C * T2) / T5
    out = []
    for t_val in t_vals:
        tau = float(t_val - t0)
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau
        out.append(c0 + c1 * tau + c2 * tau2 + c3 * tau3 + c4 * tau4 + c5 * tau5)
    return out


def _find_x_intersection_time(times, ego_positions, obj_positions, start_idx, end_idx):
    if end_idx is None:
        end_idx = len(times) - 1
    best_idx = None
    best_abs = None
    for i in range(start_idx, end_idx + 1):
        if ego_positions[i] is None or obj_positions[i] is None:
            continue
        diff = float(obj_positions[i][0] - ego_positions[i][0])
        abs_diff = abs(diff)
        if best_abs is None or abs_diff < best_abs:
            best_abs = abs_diff
            best_idx = i
        if i == start_idx:
            continue
        prev = i - 1
        if ego_positions[prev] is None or obj_positions[prev] is None:
            continue
        diff_prev = float(obj_positions[prev][0] - ego_positions[prev][0])
        if diff_prev == 0.0:
            return float(times[prev])
        if diff == 0.0:
            return float(times[i])
        if diff_prev * diff < 0.0:
            t0 = float(times[prev])
            t1 = float(times[i])
            denom = diff_prev - diff
            if abs(denom) < EPS:
                return float(times[i])
            ratio = diff_prev / denom
            return t0 + (t1 - t0) * ratio
    if best_idx is not None:
        return float(times[best_idx])
    return None


def _apply_lateral_shift_collision(orig_positions, ego_positions, start_idx, impact_idx, times,
                                   impact_offset, keep_z, post_impact, damp_time):
    new_positions = list(orig_positions)
    start_pos = orig_positions[start_idx]
    if start_pos is None:
        for i in range(start_idx - 1, -1, -1):
            if orig_positions[i] is not None:
                start_pos = orig_positions[i]
                break
    if start_pos is None:
        for i in range(start_idx + 1, len(orig_positions)):
            if orig_positions[i] is not None:
                start_pos = orig_positions[i]
                break
    if start_pos is None:
        raise ValueError("Target object start position missing.")

    impact_ego = ego_positions[impact_idx]
    if impact_ego is None:
        raise ValueError("Ego position missing at impact frame.")
    target = impact_ego + impact_offset
    impact_obj = orig_positions[impact_idx]
    if impact_obj is None:
        impact_obj = new_positions[impact_idx]
    if impact_obj is None:
        impact_obj = start_pos
    target_pos = impact_obj.copy()
    target_pos[1] = target[1]

    v0 = _estimate_velocity(orig_positions, times, start_idx)
    v1 = _estimate_velocity(orig_positions, times, impact_idx)
    y0 = float(start_pos[1])
    y1 = float(target_pos[1])
    t0 = times[start_idx] if times is not None else float(start_idx)
    t1 = times[impact_idx] if times is not None else float(impact_idx)
    t_vals = [times[i] if times is not None else float(i) for i in range(start_idx, impact_idx + 1)]
    y_segment = _compute_quintic(y0, float(v0[1]), 0.0, y1, float(v1[1]), 0.0, t0, t1, t_vals)

    for offset, y_val in enumerate(y_segment):
        idx = start_idx + offset
        base = orig_positions[idx]
        if base is None:
            continue
        new_p = base.copy()
        new_p[1] = float(y_val)
        if keep_z:
            new_p[2] = base[2]
        new_positions[idx] = new_p
    new_positions[impact_idx] = target_pos.copy()

    if post_impact == "continue":
        delta_y = target_pos[1] - impact_obj[1]
        for i in range(impact_idx + 1, len(new_positions)):
            base = orig_positions[i]
            if base is None:
                new_positions[i] = None
                continue
            new_p = base.copy()
            new_p[1] = base[1] + delta_y
            if keep_z:
                new_p[2] = base[2]
            new_positions[i] = new_p
    elif post_impact in ("hold", "damp"):
        for i in range(impact_idx + 1, len(new_positions)):
            if times is None:
                tau = float(i - impact_idx)
            else:
                tau = float(times[i] - t1)
            if post_impact == "hold":
                new_p = target_pos.copy()
            else:
                if damp_time <= EPS:
                    new_p = target_pos.copy()
                else:
                    if tau <= damp_time:
                        new_p = target_pos + v1 * (tau - 0.5 * (tau * tau) / damp_time)
                    else:
                        new_p = target_pos + v1 * (0.5 * damp_time)
            if keep_z and orig_positions[i] is not None:
                new_p[2] = orig_positions[i][2]
            new_positions[i] = new_p

    return new_positions


def _apply_direct_collision(orig_positions, ego_positions, start_idx, impact_idx, times,
                            impact_offset, keep_z, post_impact, damp_time):
    new_positions = list(orig_positions)
    start_pos = orig_positions[start_idx]
    if start_pos is None:
        for i in range(start_idx - 1, -1, -1):
            if orig_positions[i] is not None:
                start_pos = orig_positions[i]
                break
    if start_pos is None:
        for i in range(start_idx + 1, len(orig_positions)):
            if orig_positions[i] is not None:
                start_pos = orig_positions[i]
                break
    if start_pos is None:
        raise ValueError("Target object start position missing.")

    impact_pos = ego_positions[impact_idx]
    if impact_pos is None:
        raise ValueError("Ego position missing at impact frame.")
    target = impact_pos + impact_offset

    v0 = _estimate_velocity(orig_positions, times, start_idx)
    v1 = _estimate_velocity(orig_positions, times, impact_idx)
    a0 = np.zeros(3, dtype=np.float32)
    a1 = np.zeros(3, dtype=np.float32)

    t0 = times[start_idx] if times is not None else float(start_idx)
    t1 = times[impact_idx] if times is not None else float(impact_idx)
    t_vals = [times[i] if times is not None else float(i) for i in range(start_idx, impact_idx + 1)]
    segment = _compute_quintic(start_pos, v0, a0, target, v1, a1, t0, t1, t_vals)
    for offset, pos in enumerate(segment):
        idx = start_idx + offset
        if keep_z and orig_positions[idx] is not None:
            pos = pos.copy()
            pos[2] = orig_positions[idx][2]
        new_positions[idx] = pos

    if post_impact == "continue":
        impact_orig = orig_positions[impact_idx]
        if impact_orig is None:
            impact_orig = new_positions[impact_idx]
        delta = target - impact_orig
        for i in range(impact_idx + 1, len(new_positions)):
            if orig_positions[i] is None:
                new_positions[i] = None
            else:
                new_p = orig_positions[i] + delta
                if keep_z:
                    new_p[2] = orig_positions[i][2]
                new_positions[i] = new_p
    elif post_impact in ("hold", "damp"):
        for i in range(impact_idx + 1, len(new_positions)):
            if times is None:
                tau = float(i - impact_idx)
            else:
                tau = float(times[i] - t1)
            if post_impact == "hold":
                new_p = target.copy()
            else:
                if damp_time <= EPS:
                    new_p = target.copy()
                else:
                    if tau <= damp_time:
                        new_p = target + v1 * (tau - 0.5 * (tau * tau) / damp_time)
                    else:
                        new_p = target + v1 * (0.5 * damp_time)
            if keep_z and orig_positions[i] is not None:
                new_p[2] = orig_positions[i][2]
            new_positions[i] = new_p

    return new_positions, v1, target


def _check_dynamics(positions, times, max_speed=None, max_accel=None):
    if not positions or len(positions) < 2:
        return None
    max_v = 0.0
    max_a = 0.0
    velocities = []
    for i in range(1, len(positions)):
        if positions[i] is None or positions[i - 1] is None:
            velocities.append(None)
            continue
        dt = float(times[i] - times[i - 1]) if times is not None else 1.0
        if dt <= EPS:
            velocities.append(None)
            continue
        v = (positions[i] - positions[i - 1]) / dt
        velocities.append(v)
        max_v = max(max_v, float(np.linalg.norm(v)))
    for i in range(1, len(velocities)):
        v0 = velocities[i - 1]
        v1 = velocities[i]
        if v0 is None or v1 is None:
            continue
        dt = float(times[i] - times[i - 1]) if times is not None else 1.0
        if dt <= EPS:
            continue
        a = (v1 - v0) / dt
        max_a = max(max_a, float(np.linalg.norm(a)))
    if max_speed is not None and max_v > max_speed + 1e-6:
        print(f"[WARN] 速度超限: {max_v:.3f} > {max_speed:.3f}")
    if max_accel is not None and max_a > max_accel + 1e-6:
        print(f"[WARN] 加速度超限: {max_a:.3f} > {max_accel:.3f}")
    return {"max_speed": max_v, "max_accel": max_a}
def _build_upsampled_times(times, start_time, impact_time, hz):
    if hz <= 0:
        return list(times)
    dt = 1.0 / float(hz)
    before = [t for t in times if t < start_time - EPS]
    after = [t for t in times if t > impact_time + EPS]
    dense = list(np.arange(start_time, impact_time + dt * 0.5, dt))
    merged = before + dense + after
    merged.sort()
    out = []
    for t_val in merged:
        if not out or abs(t_val - out[-1]) > 1e-6:
            out.append(float(t_val))
    return out


def _resample_positions_cubic(times, positions, new_times, extrapolate=False):
    valid = [(t, pos) for t, pos in zip(times, positions) if pos is not None]
    if len(valid) < 2:
        fallback = valid[-1][1] if valid else None
        return [fallback.copy() if fallback is not None else None for _ in new_times]
    valid.sort(key=lambda x: x[0])
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as exc:
        raise RuntimeError("需要 scipy 来做三次样条插值，请先安装 scipy。") from exc
    t_vals = np.array([v[0] for v in valid], dtype=np.float32)
    coords = np.stack([v[1] for v in valid], axis=0)
    splines = [
        CubicSpline(t_vals, coords[:, 0], bc_type="natural"),
        CubicSpline(t_vals, coords[:, 1], bc_type="natural"),
        CubicSpline(t_vals, coords[:, 2], bc_type="natural"),
    ]
    t_min = float(t_vals[0])
    t_max = float(t_vals[-1])
    p_min = np.array([s(t_min) for s in splines], dtype=np.float32)
    p_max = np.array([s(t_max) for s in splines], dtype=np.float32)
    v_min = np.array([s(t_min, 1) for s in splines], dtype=np.float32)
    v_max = np.array([s(t_max, 1) for s in splines], dtype=np.float32)
    out = []
    for t_val in new_times:
        if t_val < t_min:
            if extrapolate:
                out.append(p_min + v_min * (t_val - t_min))
            else:
                out.append(p_min.copy())
            continue
        if t_val > t_max:
            if extrapolate:
                out.append(p_max + v_max * (t_val - t_max))
            else:
                out.append(p_max.copy())
            continue
        out.append(np.array([s(t_val) for s in splines], dtype=np.float32))
    return out


def _build_states_with_times(states, positions, times):
    state_by_key = {}
    for state in states:
        t_val = state_to_time(state, None)
        if t_val is None:
            continue
        state_by_key[f"{float(t_val):.6f}"] = state
    out = []
    for t_val, pos in zip(times, positions):
        if pos is None:
            continue
        key = f"{float(t_val):.6f}"
        state = state_by_key.get(key, {"t": float(t_val), "pose": {}})
        state["t"] = float(t_val)
        _set_state_pose(state, pos)
        out.append(state)
    return out


def _modify_positions(orig_positions, ego_positions, start_idx, impact_idx,
                      impact_offset, keep_z, method, post_impact, times=None,
                      damp_time=1.0, impact_scale=1.0, impact_velocity=None):
    new_positions = []
    impact_ego = ego_positions[impact_idx]
    if impact_ego is None:
        raise ValueError("Ego position missing at impact frame.")
    target = impact_ego + impact_offset
    impact_obj = orig_positions[impact_idx] if impact_idx < len(orig_positions) else None
    if impact_obj is None:
        for i in range(impact_idx - 1, -1, -1):
            if orig_positions[i] is not None:
                impact_obj = orig_positions[i]
                break
    if impact_obj is None and start_idx < len(orig_positions):
        impact_obj = orig_positions[start_idx]
    if impact_obj is not None:
        scale = float(impact_scale)
        if scale <= 0.0:
            scale = 1.0
        delta = (target - impact_obj) * scale
        target = impact_obj + delta
    else:
        delta = target * 0.0
    for i, pos in enumerate(orig_positions):
        if pos is None:
            new_positions.append(None)
            continue
        if i < start_idx:
            new_p = pos
        elif i <= impact_idx:
            if times is None:
                t = (i - start_idx) / max(1, (impact_idx - start_idx))
            else:
                denom = max(EPS, (times[impact_idx] - times[start_idx]))
                t = (times[i] - times[start_idx]) / denom
            if method == "smoothstep":
                t = _smoothstep(t)
            new_p = pos + delta * t
        else:
            if post_impact == "hold":
                new_p = target.copy()
            elif post_impact == "damp":
                if times is None:
                    tau = float(i - impact_idx)
                else:
                    tau = float(times[i] - times[impact_idx])
                if impact_velocity is None:
                    prev_idx = impact_idx - 1 if impact_idx > 0 else impact_idx
                    prev_pos = new_positions[prev_idx] if prev_idx >= 0 else None
                    if prev_pos is None:
                        v0 = np.zeros(3, dtype=np.float32)
                    else:
                        if times is None:
                            dt = float(max(1, impact_idx - prev_idx))
                        else:
                            dt = float(max(EPS, times[impact_idx] - times[prev_idx]))
                        v0 = (new_positions[impact_idx] - prev_pos) / dt
                else:
                    v0 = impact_velocity
                if damp_time <= EPS:
                    new_p = target.copy()
                else:
                    if tau <= damp_time:
                        new_p = target + v0 * (tau - 0.5 * (tau * tau) / damp_time)
                    else:
                        new_p = target + v0 * (0.5 * damp_time)
            else:
                new_p = pos + delta
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
                      fps, show_all, collision_dist, label_ego, label_objects):
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
            if label_ego:
                ax.text(ego_path[-1, 0], ego_path[-1, 1], "ego", color="red", fontsize=9)

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
            if label_objects:
                ax.text(obj_path[-1, 0], obj_path[-1, 1], obj_id, color="black", fontsize=8)

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
    parser.add_argument("--traj_json", type=str, default=None)
    parser.add_argument("--out_obj_json", type=str, default="obj_traj_modified.json")
    parser.add_argument("--obj_id", type=str, default="3")
    parser.add_argument("--start_frame", type=str, default="015")
    parser.add_argument("--impact_frame", type=str, default="027")
    parser.add_argument("--impact_offset", type=str, default="0,0,0")
    parser.add_argument("--impact_scale", type=float, default=1.0,
                        help="碰撞强度系数，>1 更激进，<1 更温和。")
    parser.add_argument("--axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--keep_z", action="store_true")
    parser.add_argument("--plan_mode", choices=["warp", "direct", "x_align"], default="direct",
                        help="轨迹生成方式：warp=偏移变形，direct=直接生成撞击轨迹，x_align=按x对齐侧向平移。")
    parser.add_argument("--method", choices=["linear", "smoothstep"], default="smoothstep")
    parser.add_argument("--post_impact", choices=["hold", "continue", "damp"], default="damp",
                        help="碰撞后处理：hold=停住，continue=沿原轨迹，damp=阻尼停下。")
    parser.add_argument("--fill_missing", action="store_true")
    parser.add_argument("--only_target", action="store_true")
    parser.add_argument("--collision_dist", type=float, default=0.15)
    parser.add_argument("--output_anim", type=str, default="traj_anim.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--show_all", action="store_true")
    parser.add_argument("--label_ego", action="store_true", help="显示 ego 标号。")
    parser.add_argument("--label_objects", action="store_true", help="显示物体 ID（默认）。")
    parser.add_argument("--no_label_objects", action="store_false", dest="label_objects",
                        help="隐藏物体 ID。")
    parser.add_argument("--ego_track", type=str, default=None)
    parser.add_argument("--obj_track", type=str, default=None)
    parser.add_argument("--pred_mode", type=str, default=None)
    parser.add_argument("--upsample_hz", type=float, default=50.0, help="插帧频率（Hz），仅在碰撞窗口生效。")
    parser.add_argument("--no_upsample", action="store_false", dest="upsample", help="关闭插帧。")
    parser.add_argument("--damp_time", type=float, default=1.0, help="阻尼停下时间（秒）。")
    parser.add_argument("--allow_extrapolate", action="store_true",
                        help="允许时间超出轨迹范围（可能导致外推不稳定）。")
    parser.add_argument("--max_speed", type=float, default=None, help="速度上限（m/s），用于检查。")
    parser.add_argument("--max_accel", type=float, default=None, help="加速度上限（m/s^2），用于检查。")
    parser.set_defaults(upsample=True, label_objects=True)
    args = parser.parse_args()

    scene_payload = None
    ego = None
    obj = None
    if args.traj_json:
        scene_payload = _load_json(args.traj_json)
        if not is_scene_traj(scene_payload):
            raise ValueError("traj_json must be a scene trajectory JSON with an agents list.")
    else:
        if not args.ego_json or not args.obj_json:
            raise ValueError("Provide --traj_json or both --ego_json and --obj_json.")
        ego = _load_json(args.ego_json)
        obj = _load_json(args.obj_json)
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
        ego_container = _get_states_container(ego_track_entry, pred_mode=args.pred_mode)
        obj_container = _get_states_container(obj_track_entry, pred_mode=args.pred_mode)
        ego_states = resolve_states_ref(ego_track_entry, pred_mode=args.pred_mode)
        obj_states = resolve_states_ref(obj_track_entry, pred_mode=args.pred_mode)
        if not ego_states or not obj_states:
            raise ValueError("Missing states for ego or target object track.")

        frame_times, obj_positions = _extract_state_series(obj_states, dt)
        if not frame_times:
            raise ValueError("Target object track has no timestamps.")
        obj_times = list(frame_times)
        ego_times, ego_positions_raw = _extract_state_series(ego_states, dt)
        obj_pairs = sorted(zip(frame_times, obj_positions), key=lambda x: x[0])
        frame_times = [p[0] for p in obj_pairs]
        obj_positions = [p[1] for p in obj_pairs]
        obj_times = list(frame_times)
        ego_pairs = sorted(zip(ego_times, ego_positions_raw), key=lambda x: x[0])
        ego_times = [p[0] for p in ego_pairs]
        ego_positions_raw = [p[1] for p in ego_pairs]

        _, start_time, _ = _parse_time_or_index(frame_times, args.start_frame)
        _, impact_time, _ = _parse_time_or_index(frame_times, args.impact_frame)
        t_min = max(min(ego_times), min(frame_times))
        t_max = min(max(ego_times), max(frame_times))
        if t_min > t_max:
            raise ValueError("Ego/obj 时间范围没有交集，无法规划碰撞。")
        if not args.allow_extrapolate:
            clamped_start = _clamp_time_range(start_time, t_min, t_max)
            clamped_impact = _clamp_time_range(impact_time, t_min, t_max)
            if abs(clamped_start - start_time) > 1e-6:
                print(f"[WARN] start_time 超出范围，已调整为 {clamped_start:.3f}")
            if abs(clamped_impact - impact_time) > 1e-6:
                print(f"[WARN] impact_time 超出范围，已调整为 {clamped_impact:.3f}")
            start_time = clamped_start
            impact_time = clamped_impact
        if start_time > impact_time:
            print("[WARN] start_time > impact_time，已交换")
            start_time, impact_time = impact_time, start_time
        need_resample = args.upsample or not _time_in_list(frame_times, start_time) or not _time_in_list(frame_times, impact_time)

        if need_resample:
            if args.upsample:
                frame_times = _build_upsampled_times(
                    frame_times, start_time, impact_time, args.upsample_hz
                )
            else:
                frame_times = sorted(set(frame_times + [start_time, impact_time]))
            ego_positions = _resample_positions_cubic(
                ego_times, ego_positions_raw, frame_times, extrapolate=args.allow_extrapolate
            )
            obj_positions = _resample_positions_cubic(
                obj_times, obj_positions, frame_times, extrapolate=args.allow_extrapolate
            )
        else:
            ego_series = [
                (t_val, pos)
                for t_val, pos in zip(ego_times, ego_positions_raw)
                if pos is not None
            ]
            ego_positions = sample_positions_by_time(ego_series, frame_times)
        if args.plan_mode == "x_align":
            end_idx = _nearest_time_index(frame_times, impact_time)
            start_idx = _nearest_time_index(frame_times, start_time)
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            x_time = _find_x_intersection_time(frame_times, ego_positions, obj_positions, start_idx, end_idx)
            if x_time is None:
                raise ValueError("找不到 x 轴交汇点。")
            impact_time = x_time
            if not _time_in_list(frame_times, impact_time):
                frame_times = sorted(set(frame_times + [impact_time]))
                ego_positions = _resample_positions_cubic(
                    ego_times, ego_positions_raw, frame_times, extrapolate=args.allow_extrapolate
                )
                obj_positions = _resample_positions_cubic(
                    obj_times, obj_positions, frame_times, extrapolate=args.allow_extrapolate
                )
        frames = [f"{t:.3f}" for t in frame_times]
        start_idx = _nearest_time_index(frame_times, start_time)
        impact_idx = _nearest_time_index(frame_times, impact_time)

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

    frame_time_values = None
    if scene_payload:
        frame_time_values = frame_times
    else:
        frame_time_values = [float(i) for i in range(len(frames))]

    if args.plan_mode == "direct":
        new_positions, impact_velocity, _ = _apply_direct_collision(
            obj_positions, ego_positions, start_idx, impact_idx, frame_time_values,
            impact_offset, args.keep_z, args.post_impact, args.damp_time
        )
        _check_dynamics(new_positions, frame_time_values, args.max_speed, args.max_accel)
    elif args.plan_mode == "x_align":
        new_positions = _apply_lateral_shift_collision(
            obj_positions, ego_positions, start_idx, impact_idx, frame_time_values,
            impact_offset, args.keep_z, args.post_impact, args.damp_time
        )
        _check_dynamics(new_positions, frame_time_values, args.max_speed, args.max_accel)
    else:
        new_positions = _modify_positions(
            obj_positions, ego_positions, start_idx, impact_idx,
            impact_offset, args.keep_z, args.method, args.post_impact,
            times=frame_time_values, damp_time=args.damp_time,
            impact_scale=args.impact_scale
        )
        _check_dynamics(new_positions, frame_time_values, args.max_speed, args.max_accel)
    if scene_payload:
        obj_states = _build_states_with_times(obj_states, new_positions, frame_times)
        if obj_container is None:
            raise ValueError("Target track container not found for scene trajectory.")
        obj_container["states"] = obj_states
        if args.upsample and ego_container is not None:
            ego_states = _build_states_with_times(ego_states, ego_positions, frame_times)
            ego_container["states"] = ego_states
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
                if args.upsample:
                    obj_positions_map[agent_id] = _resample_positions_cubic(
                        other_times, other_positions, frame_times
                    )
                else:
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
        args.output_anim, args.fps, args.show_all, args.collision_dist,
        args.label_ego, args.label_objects
    )

    print(f"Saved updated obj traj to {args.out_obj_json}")
    print(f"Saved animation to {args.output_anim}")


if __name__ == "__main__":
    main()
