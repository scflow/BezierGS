#
# Render a nuScenes-style multi-camera layout video with optional 2D boxes and trajectories.
#
import glob
import json
import os
from argparse import ArgumentParser

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from gaussian_renderer import render
from scene import Scene, GaussianModel, EnvLight, ColorCorrection, PoseCorrection
from utils.general_utils import seed_everything

EPS = 1e-6

# Names expected by the user; layout is defined separately.
CAM_NAME_ORDER = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_RIGHT", "BACK_LEFT"]
CAM_LAYOUT = [
    ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
    ["BACK_LEFT", "BACK", "BACK_RIGHT"],
]


def _parse_int_list(value, expected_len=None):
    items = [int(x) for x in value.split(",") if x.strip() != ""]
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(items)} in '{value}'")
    return items


def _parse_float_list(value, expected_len=None):
    items = [float(x) for x in value.split(",") if x.strip() != ""]
    if expected_len is not None and len(items) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(items)} in '{value}'")
    return items


def _parse_color_bgr(value):
    # Accept "r,g,b" in 0-255.
    rgb = _parse_int_list(value, expected_len=3)
    return (rgb[2], rgb[1], rgb[0])


def _load_json(path):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _normalize_frame_key(frame_id):
    return f"{int(frame_id):03d}"


def _get_frame_timestamp(frame_cams):
    for cam in frame_cams.values():
        return float(cam.timestamp)
    return 0.0


def _build_frames(cameras):
    frames = {}
    for cam in cameras:
        frame_id = int(cam.colmap_id // 10)
        cam_id = int(cam.colmap_id % 10)
        frames.setdefault(frame_id, {})[cam_id] = cam
    return frames


def _resize_letterbox(image, target_w, target_h, bg_color):
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _draw_text(image, text, org, color, scale, thickness):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _load_box_frames(box_json):
    if not box_json:
        return {}
    frames = box_json.get("frames", box_json)
    out = {}
    for frame_key, cams in frames.items():
        out[str(frame_key)] = {}
        if not isinstance(cams, dict):
            continue
        for cam_key, boxes in cams.items():
            out[str(frame_key)][str(cam_key)] = boxes
    return out


def _load_traj_frames(traj_json):
    if not traj_json:
        return None, None
    coord_mode = traj_json.get("coord", "world")
    frames = traj_json.get("frames", traj_json)
    return frames, coord_mode


def _extract_axes(pos, axes):
    if axes == "xy":
        return float(pos[0]), float(pos[1])
    if axes == "xz":
        return float(pos[0]), float(pos[2])
    if axes == "yz":
        return float(pos[1]), float(pos[2])
    raise ValueError(f"Unsupported axes '{axes}'")


def _build_source_ego_track(frames, frame_ids, cam_id):
    positions = []
    for frame_id in frame_ids:
        cam = frames[frame_id].get(cam_id)
        if cam is None:
            positions.append(None)
            continue
        positions.append(cam.camera_center.detach().cpu().numpy())
    return positions


def _build_source_obj_tracks(scene, frame_ids, frame_timestamps, max_tracks):
    tracks = {}
    traj_dict = getattr(scene.gaussians, "trajectory_dict", {})
    if not traj_dict:
        return tracks
    obj_ids = sorted(traj_dict.keys())
    if max_tracks is not None:
        obj_ids = obj_ids[:max_tracks]
    for obj_id in obj_ids:
        time_pos = traj_dict[obj_id]
        if not time_pos:
            continue
        times = np.array([float(t) for t in time_pos.keys()], dtype=np.float32)
        positions = [time_pos[t].detach().cpu().numpy() for t in time_pos.keys()]
        positions = np.stack(positions, axis=0)
        track = []
        for frame_id in frame_ids:
            t = frame_timestamps[frame_id]
            idx = int(np.argmin(np.abs(times - t)))
            track.append(positions[idx])
        tracks[str(obj_id)] = track
    return tracks


def _collect_traj_bounds(tracks, axes):
    xs, ys = [], []
    for points in tracks.values():
        for pos in points:
            if pos is None:
                continue
            x, y = _extract_axes(pos, axes)
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < EPS:
        max_x += 1.0
    if abs(max_y - min_y) < EPS:
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def _map_traj_point(pos, bounds, axes, panel, flip_y, pad_frac):
    x0, y0, w, h = panel
    min_x, max_x, min_y, max_y = bounds
    dx = max_x - min_x
    dy = max_y - min_y
    min_x -= dx * pad_frac
    max_x += dx * pad_frac
    min_y -= dy * pad_frac
    max_y += dy * pad_frac
    dx = max_x - min_x
    dy = max_y - min_y
    x, y = _extract_axes(pos, axes)
    nx = (x - min_x) / dx
    ny = (y - min_y) / dy
    if flip_y:
        ny = 1.0 - ny
    px = int(x0 + np.clip(nx, 0.0, 1.0) * (w - 1))
    py = int(y0 + np.clip(ny, 0.0, 1.0) * (h - 1))
    return px, py


def _draw_trajectory_panel(image, panel, ego_track, obj_tracks, traj_bounds, traj_axes,
                           traj_pad, flip_y, ego_color, obj_color, thickness, point_radius):
    x0, y0, w, h = panel
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (64, 64, 64), 1)
    if traj_bounds is None:
        return
    # Ego trajectory.
    if ego_track:
        points = []
        for pos in ego_track:
            if pos is None:
                continue
            points.append(_map_traj_point(pos, traj_bounds, traj_axes, panel, flip_y, traj_pad))
        if len(points) >= 2:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], False, ego_color, thickness)
        if points:
            cv2.circle(image, points[-1], point_radius, ego_color, -1)
    # Object trajectories.
    for _, track in obj_tracks.items():
        if not track:
            continue
        points = []
        for pos in track:
            if pos is None:
                continue
            points.append(_map_traj_point(pos, traj_bounds, traj_axes, panel, flip_y, traj_pad))
        if len(points) >= 2:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], False, obj_color, max(1, thickness - 1))
        if points:
            cv2.circle(image, points[-1], point_radius, obj_color, -1)


def _normalize_traj_entry(entry):
    if entry is None:
        return None
    if isinstance(entry, list):
        if len(entry) == 0:
            return []
        if isinstance(entry[0], (int, float)):
            return [entry]
        return entry
    return None


def _to_numpy_points(points):
    if points is None:
        return None
    out = []
    for p in points:
        if p is None:
            continue
        out.append(np.array(p, dtype=np.float32))
    return out


def _flatten_track_frames(track_frames):
    flat = []
    for pts in track_frames:
        if pts:
            flat.extend(pts)
    return flat


@torch.no_grad()
def main():
    parser = ArgumentParser(description="Render nuScenes-layout video")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base_config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--canvas_w", type=int, default=1920)
    parser.add_argument("--canvas_h", type=int, default=1080)
    parser.add_argument("--frame_margin", type=int, default=24)
    parser.add_argument("--frame_border", type=int, default=6)
    parser.add_argument("--frame_color", type=str, default="255,255,255")
    parser.add_argument("--bg_color", type=str, default="0,0,0")
    parser.add_argument("--cell_padding", type=int, default=8)
    parser.add_argument("--cam_order", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--show_cam_names", action="store_true")
    parser.add_argument("--box_json", type=str, default=None)
    parser.add_argument("--show_boxes", action="store_true")
    parser.add_argument("--show_ids", action="store_true")
    parser.add_argument("--box_color", type=str, default="0,255,0")
    parser.add_argument("--box_thickness", type=int, default=2)
    parser.add_argument("--id_color", type=str, default="255,255,255")
    parser.add_argument("--id_scale", type=float, default=0.6)
    parser.add_argument("--ego_traj_json", type=str, default=None)
    parser.add_argument("--obj_traj_json", type=str, default=None)
    parser.add_argument("--use_source_traj", action="store_true")
    parser.add_argument("--no_source_traj", action="store_false", dest="use_source_traj")
    parser.add_argument("--max_obj_tracks", type=int, default=0)
    parser.add_argument("--traj_panel", type=str, default="0.68,0.62,0.30,0.30")
    parser.add_argument("--traj_axes", type=str, default="xy", choices=["xy", "xz", "yz"])
    parser.add_argument("--traj_pad", type=float, default=0.05)
    parser.add_argument("--traj_flip_y", action="store_true")
    parser.add_argument("--traj_ego_color", type=str, default="255,64,64")
    parser.add_argument("--traj_obj_color", type=str, default="64,200,255")
    parser.add_argument("--traj_thickness", type=int, default=2)
    parser.add_argument("--traj_point_radius", type=int, default=3)
    parser.add_argument("--ego_cam_name", type=str, default="FRONT")
    parser.set_defaults(use_source_traj=True)
    args, unknown = parser.parse_known_args()

    base_conf = OmegaConf.load(args.base_config)
    second_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(base_conf, second_conf, cli_conf)
    cfg.resolution_scales = cfg.resolution_scales[:1]

    seed_everything(cfg.seed)

    gaussians = GaussianModel(cfg)
    scene = Scene(cfg, gaussians, shuffle=False)

    if cfg.env_map_res > 0:
        env_map = EnvLight(resolution=cfg.env_map_res).cuda()
        env_map.training_setup(cfg)
    else:
        env_map = None

    if cfg.use_color_correction:
        color_correction = ColorCorrection(cfg)
        color_correction.training_setup(cfg)
    else:
        color_correction = None

    if cfg.use_pose_correction:
        pose_correction = PoseCorrection(cfg)
        pose_correction.training_setup(cfg)
    else:
        pose_correction = None

    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint is None or str(checkpoint).strip() == "":
        checkpoints = glob.glob(os.path.join(cfg.model_path, "chkpnt*.pth"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found in model_path.")
        checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
    checkpoint = os.path.expanduser(str(checkpoint))
    model_params, first_iter = torch.load(checkpoint)
    gaussians.restore(model_params, cfg)

    if env_map is not None:
        env_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                      os.path.basename(checkpoint).replace("chkpnt", "env_light_chkpnt"))
        if os.path.exists(env_checkpoint):
            light_params, _ = torch.load(env_checkpoint)
            env_map.restore(light_params)
    if color_correction is not None:
        cc_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                     os.path.basename(checkpoint).replace("chkpnt", "color_correction_chkpnt"))
        if os.path.exists(cc_checkpoint):
            cc_params, _ = torch.load(cc_checkpoint)
            color_correction.restore(cc_params)
    if pose_correction is not None:
        pc_checkpoint = os.path.join(os.path.dirname(checkpoint),
                                     os.path.basename(checkpoint).replace("chkpnt", "pose_correction_chkpnt"))
        if os.path.exists(pc_checkpoint):
            pc_params, _ = torch.load(pc_checkpoint)
            pose_correction.restore(pc_params)

    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cams = scene.getTestCameras(scale=scene.resolution_scales[0]) if args.split == "test" else scene.getTrainCameras()
    frames = _build_frames(cams)
    frame_ids = sorted(frames.keys())
    if args.frame_stride > 1:
        frame_ids = frame_ids[::args.frame_stride]
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]

    cam_order = _parse_int_list(args.cam_order, expected_len=6)
    cam_name_to_id = dict(zip(CAM_NAME_ORDER, cam_order))

    ego_cam_name = args.ego_cam_name.upper()
    if ego_cam_name not in cam_name_to_id:
        raise ValueError(f"Unknown ego_cam_name '{args.ego_cam_name}'. Use one of {CAM_NAME_ORDER}.")
    ego_cam_id = cam_name_to_id[ego_cam_name]

    frame_timestamps = {frame_id: _get_frame_timestamp(frames[frame_id]) for frame_id in frame_ids}

    box_frames = _load_box_frames(_load_json(args.box_json))
    ego_traj_frames, ego_coord = _load_traj_frames(_load_json(args.ego_traj_json))
    obj_traj_frames, obj_coord = _load_traj_frames(_load_json(args.obj_traj_json))

    use_source_traj = args.use_source_traj
    ego_track_frames = []
    obj_track_frames = {}
    ego_mode = "world"
    obj_mode = "world"

    if ego_traj_frames:
        ego_mode = ego_coord
        for frame_id in frame_ids:
            key = _normalize_frame_key(frame_id)
            entry = ego_traj_frames.get(key)
            if isinstance(entry, dict):
                entry = entry.get("ego")
            pts = _normalize_traj_entry(entry)
            ego_track_frames.append(_to_numpy_points(pts))
    elif use_source_traj:
        ego_positions = _build_source_ego_track(frames, frame_ids, ego_cam_id)
        accum = []
        for pos in ego_positions:
            if pos is not None:
                accum.append(pos)
            ego_track_frames.append(list(accum))

    if obj_traj_frames:
        obj_mode = obj_coord
        for frame_idx, frame_id in enumerate(frame_ids):
            key = _normalize_frame_key(frame_id)
            entry = obj_traj_frames.get(key, {})
            if isinstance(entry, dict):
                others = entry.get("others", {})
            else:
                others = {}
            seen = set()
            for obj_id, pts in others.items():
                obj_id = str(obj_id)
                if obj_id not in obj_track_frames:
                    obj_track_frames[obj_id] = [None] * frame_idx
                obj_track_frames[obj_id].append(_to_numpy_points(_normalize_traj_entry(pts)))
                seen.add(obj_id)
            for obj_id in list(obj_track_frames.keys()):
                if obj_id not in seen and len(obj_track_frames[obj_id]) == frame_idx:
                    obj_track_frames[obj_id].append(None)
    elif use_source_traj:
        max_tracks = args.max_obj_tracks if args.max_obj_tracks > 0 else None
        obj_positions = _build_source_obj_tracks(scene, frame_ids, frame_timestamps, max_tracks)
        for obj_id, positions in obj_positions.items():
            accum = []
            frames_seq = []
            for pos in positions:
                if pos is not None:
                    accum.append(pos)
                frames_seq.append(list(accum))
            obj_track_frames[obj_id] = frames_seq

    traj_bounds = None
    if (ego_track_frames or obj_track_frames) and (ego_mode != "normalized" or obj_mode != "normalized"):
        merged_tracks = {}
        if ego_mode != "normalized" and ego_track_frames:
            merged_tracks["ego"] = _flatten_track_frames(ego_track_frames)
        if obj_mode != "normalized":
            for k, frames_seq in obj_track_frames.items():
                merged_tracks[k] = _flatten_track_frames(frames_seq)
        traj_bounds = _collect_traj_bounds(merged_tracks, args.traj_axes)

    canvas_w, canvas_h = args.canvas_w, args.canvas_h
    frame_margin = args.frame_margin
    frame_border = args.frame_border
    cell_padding = args.cell_padding
    frame_color = _parse_color_bgr(args.frame_color)
    bg_color_bgr = _parse_color_bgr(args.bg_color)
    box_color = _parse_color_bgr(args.box_color)
    id_color = _parse_color_bgr(args.id_color)
    ego_color = _parse_color_bgr(args.traj_ego_color)
    obj_color = _parse_color_bgr(args.traj_obj_color)

    panel_ratio = _parse_float_list(args.traj_panel, expected_len=4)
    panel = (
        int(panel_ratio[0] * canvas_w),
        int(panel_ratio[1] * canvas_h),
        int(panel_ratio[2] * canvas_w),
        int(panel_ratio[3] * canvas_h),
    )

    rows, cols = 2, 3
    inner_w = canvas_w - 2 * frame_margin - 2 * frame_border
    inner_h = canvas_h - 2 * frame_margin - 2 * frame_border
    cell_w = max(1, (inner_w - cell_padding * (cols - 1)) // cols)
    cell_h = max(1, (inner_h - cell_padding * (rows - 1)) // rows)
    grid_w = cell_w * cols + cell_padding * (cols - 1)
    grid_h = cell_h * rows + cell_padding * (rows - 1)
    grid_x0 = frame_margin + frame_border + (inner_w - grid_w) // 2
    grid_y0 = frame_margin + frame_border + (inner_h - grid_h) // 2

    if args.output:
        output_path = args.output
    else:
        output_name = f"nuscenes_layout_{args.split}_{first_iter}.mp4"
        output_path = os.path.join(cfg.model_path, "eval", output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = imageio.get_writer(output_path, fps=args.fps, codec="libx264", quality=8)

    for frame_idx, frame_id in enumerate(frame_ids):
        canvas = np.full((canvas_h, canvas_w, 3), bg_color_bgr, dtype=np.uint8)
        cv2.rectangle(canvas,
                      (frame_margin, frame_margin),
                      (canvas_w - frame_margin, canvas_h - frame_margin),
                      frame_color, frame_border)
        for r, row in enumerate(CAM_LAYOUT):
            for c, cam_name in enumerate(row):
                cam_id = cam_name_to_id.get(cam_name)
                cam = frames[frame_id].get(cam_id) if cam_id is not None else None
                cell = np.full((cell_h, cell_w, 3), bg_color_bgr, dtype=np.uint8)
                if cam is not None:
                    v, _ = scene.gaussians.get_inst_velocity(cam.timestamp)
                    render_pkg = render(cam, scene.gaussians, cfg, background,
                                        env_map=env_map, color_correction=color_correction,
                                        pose_correction=pose_correction, other=[v])
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    image = (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if args.show_boxes:
                        frame_key = _normalize_frame_key(frame_id)
                        cam_key = str(cam_id)
                        boxes = box_frames.get(frame_key, {}).get(cam_key, [])
                        for box in boxes:
                            bbox = box.get("bbox", box)
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, args.box_thickness)
                            if args.show_ids and "id" in box:
                                _draw_text(image, str(box["id"]), (x1, max(0, y1 - 4)),
                                           id_color, args.id_scale, 1)
                    cell = _resize_letterbox(image, cell_w, cell_h, bg_color_bgr)

                if args.show_cam_names:
                    _draw_text(cell, cam_name, (8, 24), (255, 255, 255), 0.7, 2)

                x0 = grid_x0 + c * (cell_w + cell_padding)
                y0 = grid_y0 + r * (cell_h + cell_padding)
                canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell

        if ego_track_frames or obj_track_frames:
            if ego_mode == "normalized" and ego_track_frames:
                ego_pts = []
                for p in ego_track_frames[frame_idx] or []:
                    px = int(panel[0] + np.clip(p[0], 0.0, 1.0) * (panel[2] - 1))
                    py = int(panel[1] + np.clip(1.0 - p[1], 0.0, 1.0) * (panel[3] - 1))
                    ego_pts.append((px, py))
                if ego_pts:
                    cv2.polylines(canvas, [np.array(ego_pts, dtype=np.int32)], False, ego_color, args.traj_thickness)
                    cv2.circle(canvas, ego_pts[-1], args.traj_point_radius, ego_color, -1)
            if obj_mode == "normalized" and obj_track_frames:
                for _, seq in obj_track_frames.items():
                    obj_pts = []
                    for p in seq[frame_idx] or []:
                        px = int(panel[0] + np.clip(p[0], 0.0, 1.0) * (panel[2] - 1))
                        py = int(panel[1] + np.clip(1.0 - p[1], 0.0, 1.0) * (panel[3] - 1))
                        obj_pts.append((px, py))
                    if obj_pts:
                        cv2.polylines(canvas, [np.array(obj_pts, dtype=np.int32)], False, obj_color,
                                      max(1, args.traj_thickness - 1))
                        cv2.circle(canvas, obj_pts[-1], args.traj_point_radius, obj_color, -1)

            if (ego_mode != "normalized" and ego_track_frames) or (obj_mode != "normalized" and obj_track_frames):
                ego_sub = ego_track_frames[frame_idx] if ego_mode != "normalized" else []
                obj_sub = {}
                if obj_mode != "normalized":
                    for k, seq in obj_track_frames.items():
                        obj_sub[k] = seq[frame_idx] if frame_idx < len(seq) else None
                _draw_trajectory_panel(canvas, panel, ego_sub, obj_sub, traj_bounds, args.traj_axes,
                                       args.traj_pad, args.traj_flip_y, ego_color, obj_color,
                                       args.traj_thickness, args.traj_point_radius)

        frame_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()
    torch.cuda.empty_cache()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
