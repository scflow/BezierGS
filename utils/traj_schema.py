import numpy as np

EPS = 1e-6


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_scene_traj(payload):
    return isinstance(payload, dict) and isinstance(payload.get("agents"), list)


def state_to_pos(state):
    if not isinstance(state, dict):
        return None
    if "pose" in state:
        pos = _coerce_xyz(state.get("pose"))
        if pos is not None:
            return pos
    if "position" in state:
        pos = _coerce_xyz(state.get("position"))
        if pos is not None:
            return pos
    if "pos" in state:
        pos = _coerce_xyz(state.get("pos"))
        if pos is not None:
            return pos
    if "x" in state and "y" in state:
        z = state.get("z", 0.0)
        return np.array([float(state["x"]), float(state["y"]), float(z)], dtype=np.float32)
    return None


def state_to_time(state, fallback):
    if isinstance(state, dict) and "t" in state:
        t_val = _coerce_float(state.get("t"))
        if t_val is not None:
            return t_val
    return fallback


def select_track(tracks, preferred=None):
    if not isinstance(tracks, dict) or not tracks:
        return None, None
    if preferred:
        if isinstance(preferred, str):
            preferred = [preferred]
        for name in preferred:
            if name and name in tracks:
                return name, tracks[name]
    for name, entry in tracks.items():
        return name, entry
    return None, None


def select_multi_modal(modes, pred_mode=None):
    if not modes:
        return None
    if pred_mode is not None:
        for mode in modes:
            if str(mode.get("mode_id")) == str(pred_mode):
                return mode
    best = None
    best_prob = None
    for mode in modes:
        prob = _coerce_float(mode.get("probability"))
        if prob is None:
            if best is None:
                best = mode
            continue
        if best_prob is None or prob > best_prob:
            best_prob = prob
            best = mode
    return best or modes[0]


def resolve_states_ref(track_entry, pred_mode=None):
    if not isinstance(track_entry, dict):
        return None
    states = track_entry.get("states")
    if isinstance(states, list):
        return states
    trajectory = track_entry.get("trajectory")
    if isinstance(trajectory, dict):
        states = trajectory.get("states")
        if isinstance(states, list):
            return states
    modes = track_entry.get("multi_modal")
    if isinstance(modes, list):
        mode = select_multi_modal(modes, pred_mode=pred_mode)
        if isinstance(mode, dict):
            states = mode.get("states")
            if isinstance(states, list):
                return states
    return None


def extract_states(track_entry, pred_mode=None):
    states = resolve_states_ref(track_entry, pred_mode=pred_mode)
    if not states:
        return []
    return states


def build_series(states, dt=None):
    if not states:
        return []
    dt_val = _coerce_float(dt)
    series = []
    for idx, state in enumerate(states):
        fallback = float(idx)
        if dt_val is not None:
            fallback = float(idx) * dt_val
        t_val = state_to_time(state, fallback)
        pos = state_to_pos(state)
        if pos is None:
            continue
        series.append((float(t_val), pos))
    series.sort(key=lambda x: x[0])
    return series


def extract_scene_tracks(scene_payload, ego_track=None, obj_track=None, pred_mode=None, ego_id="ego"):
    coord_mode = scene_payload.get("frame_id") or scene_payload.get("coord") or "world"
    time_base = scene_payload.get("time_base") or {}
    dt_val = _coerce_float(time_base.get("dt"))
    ego_series = []
    obj_series = {}
    agents = scene_payload.get("agents") or []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = agent.get("agent_id")
        if agent_id is None:
            continue
        agent_id = str(agent_id)
        tracks = agent.get("tracks") or {}
        if agent_id == ego_id:
            preferred = [ego_track] if ego_track else ["history", "observed", "planned", "predicted"]
        else:
            preferred = [obj_track] if obj_track else ["observed", "history", "predicted", "planned"]
        _, track_entry = select_track(tracks, preferred)
        if track_entry is None:
            continue
        states = extract_states(track_entry, pred_mode=pred_mode)
        if not states:
            continue
        series = build_series(states, dt=dt_val)
        if agent_id == ego_id:
            ego_series = series
        else:
            obj_series[agent_id] = series
    return coord_mode, ego_series, obj_series, dt_val


def collect_frame_times(ego_series, obj_series):
    times = []
    for t_val, _ in ego_series:
        times.append(t_val)
    for series in obj_series.values():
        for t_val, _ in series:
            times.append(t_val)
    if not times:
        return []
    return sorted(set(times))


def sample_positions_by_time(series, frame_times, eps=EPS):
    if not frame_times:
        return []
    if not series:
        return [None] * len(frame_times)
    series_sorted = sorted(series, key=lambda x: x[0])
    out = []
    idx = 0
    last = None
    for t_val in frame_times:
        while idx < len(series_sorted) and series_sorted[idx][0] <= t_val + eps:
            last = series_sorted[idx][1]
            idx += 1
        out.append(None if last is None else last.copy())
    return out


def cumulative_points_by_time(series, frame_times, eps=EPS):
    if not frame_times:
        return []
    if not series:
        return [[] for _ in frame_times]
    series_sorted = sorted(series, key=lambda x: x[0])
    out = []
    points = []
    idx = 0
    for t_val in frame_times:
        while idx < len(series_sorted) and series_sorted[idx][0] <= t_val + eps:
            points.append(series_sorted[idx][1])
            idx += 1
        out.append(list(points))
    return out


def _coerce_xyz(value):
    if isinstance(value, dict):
        x_val = value.get("x")
        y_val = value.get("y")
        z_val = value.get("z", 0.0)
        if x_val is None or y_val is None:
            return None
        return np.array([float(x_val), float(y_val), float(z_val)], dtype=np.float32)
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            return None
        x_val = value[0]
        y_val = value[1]
        z_val = value[2] if len(value) > 2 else 0.0
        return np.array([float(x_val), float(y_val), float(z_val)], dtype=np.float32)
    return None
