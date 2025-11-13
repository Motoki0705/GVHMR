import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from einops import einsum
import hydra
from hydra import compose, initialize_config_module
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d.configs import register_store_gvhmr
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    read_video_np,
    save_video,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.vis.renderer import Renderer

# --- Torch 2.6+ の "weights_only=True" 既定への互換処理（Ultralyticsの重み読み込み用） ---
from ultralytics.nn import tasks as _utasks
def _torch_safe_load(file):
    # 公式の .pt を使う前提で weights_only=False で読み込む
    ckpt = torch.load(file, map_location="cpu", weights_only=False)
    return ckpt, str(file)  # ★ Ultralytics 側が (ckpt, weight) の2値を期待
_utasks.torch_safe_load = _torch_safe_load
# --------------------------------------------------------------------------------------------

CRF = 23


@dataclass
class PersonPaths:
    track_id: int
    preprocess_dir: Path
    output_dir: Path
    bbx: Path
    bbx_xyxy_video_overlay: Path
    vit_features: Path
    vitpose: Path
    vitpose_video_overlay: Path
    hmr4d_results: Path
    slam: Path


@dataclass
class PersonSequence:
    track_id: int
    verts_incam: torch.Tensor  # (L, V, 3) on CPU
    joints_incam: torch.Tensor  # (L, J, 3) on CPU
    K_fullimg: torch.Tensor  # (3, 3)


def parse_args_to_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc="Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


def build_person_paths(cfg, track_id: int) -> PersonPaths:
    preprocess_dir = Path(cfg.preprocess_dir) / f"id_{track_id}"
    output_dir = Path(cfg.output_dir) / f"id_{track_id}"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return PersonPaths(
        track_id=track_id,
        preprocess_dir=preprocess_dir,
        output_dir=output_dir,
        bbx=preprocess_dir / "bbx.pt",
        bbx_xyxy_video_overlay=preprocess_dir / "bbx_xyxy_video_overlay.mp4",
        vit_features=preprocess_dir / "vit_features.pt",
        vitpose=preprocess_dir / "vitpose.pt",
        vitpose_video_overlay=preprocess_dir / "vitpose_video_overlay.mp4",
        hmr4d_results=output_dir / "hmr4d_results.pt",
        slam=Path(cfg.paths.slam),
    )


def ensure_slam(cfg):
    paths = cfg.paths
    if cfg.static_cam:
        return
    slam_path = Path(paths.slam)
    if slam_path.exists():
        Log.info(f"[Preprocess] slam results from {slam_path}")
        return
    if not cfg.use_dpvo:
        simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
        vo_results = simple_vo.compute()
        torch.save(vo_results, slam_path)
    else:
        from hmr4d.utils.preproc.slam import SLAMModel

        length, width, height = get_video_lwh(cfg.video_path)
        K_fullimg = estimate_K(width, height)
        intrinsics = convert_K_to_K4(K_fullimg)
        slam = SLAMModel(cfg.video_path, width, height, intrinsics, buffer=4000, resize=0.5)
        bar = tqdm(total=length, desc="DPVO")
        while True:
            ret = slam.track()
            if ret:
                bar.update()
            else:
                break
        slam_results = slam.process()
        torch.save(slam_results, slam_path)


def run_preprocess_for_person(cfg, person_paths: PersonPaths, track_id: int, bbx_xyxy: torch.Tensor):
    video_path = cfg.video_path
    verbose = cfg.verbose

    if not person_paths.bbx.exists():
        bbx_xyxy = bbx_xyxy.float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
        torch.save({"id": track_id, "bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, person_paths.bbx)
    else:
        data = torch.load(person_paths.bbx, weights_only=False)
        bbx_xyxy = data["bbx_xyxy"]
        bbx_xys = data["bbx_xys"]
        Log.info(f"[Preprocess][id={track_id}] bbx from {person_paths.bbx}")

    if verbose and not person_paths.bbx_xyxy_video_overlay.exists():
        video = read_video_np(video_path)
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, str(person_paths.bbx_xyxy_video_overlay))

    if not person_paths.vitpose.exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, person_paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(person_paths.vitpose, weights_only=False)
        Log.info(f"[Preprocess][id={track_id}] vitpose from {person_paths.vitpose}")

    if verbose and not person_paths.vitpose_video_overlay.exists():
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, str(person_paths.vitpose_video_overlay))

    if not person_paths.vit_features.exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, person_paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess][id={track_id}] vit_features from {person_paths.vit_features}")


def load_data_dict(cfg, person_paths: PersonPaths):
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(person_paths.slam, weights_only=False)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    bbx_bundle = torch.load(person_paths.bbx, weights_only=False)
    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_bundle["bbx_xys"],
        "kp2d": torch.load(person_paths.vitpose, weights_only=False),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(person_paths.vit_features, weights_only=False),
    }
    return data


def prepare_person_sequences(person_paths_by_id: Dict[int, PersonPaths]) -> Tuple[Dict[int, PersonSequence], torch.Tensor]:
    if not person_paths_by_id:
        raise ValueError("person_paths_by_id must not be empty.")

    device = torch.device("cuda")
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to(device)
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).to(device)
    faces_raw = make_smplx("smpl").faces
    if isinstance(faces_raw, np.ndarray):
        faces_smpl = torch.from_numpy(faces_raw.astype(np.int64))
    else:
        faces_smpl = torch.as_tensor(faces_raw, dtype=torch.int64)

    sequences: Dict[int, PersonSequence] = {}

    for track_id, paths in person_paths_by_id.items():
        if not paths.hmr4d_results.exists():
            raise FileNotFoundError(f"HMR4D results not found for track {track_id} at {paths.hmr4d_results}")
        pred = torch.load(paths.hmr4d_results, weights_only=False)
        smpl_params_incam = to_cuda(pred["smpl_params_incam"])
        smplx_out = smplx(**smpl_params_incam)

        verts_smpl = torch.stack([torch.matmul(smplx2smpl, verts) for verts in smplx_out.vertices])
        joints_smpl = einsum(J_regressor, verts_smpl, "j v, l v i -> l j i")

        sequences[track_id] = PersonSequence(
            track_id=track_id,
            verts_incam=verts_smpl.detach().cpu(),
            joints_incam=joints_smpl.detach().cpu(),
            K_fullimg=torch.as_tensor(pred["K_fullimg"][0]).float(),
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sequences, faces_smpl

def assign_track_colors(track_ids: Iterable[int]) -> Dict[int, List[int]]:
    palette = [
        (231, 111, 81),
        (244, 162, 97),
        (233, 196, 106),
        (42, 157, 143),
        (38, 70, 83),
        (94, 129, 172),
        (129, 178, 154),
        (198, 70, 104),
    ]
    id_list = sorted(track_ids)
    return {track_id: list(palette[idx % len(palette)]) for idx, track_id in enumerate(id_list)}


def render_incam_multi(cfg, sequences: Dict[int, PersonSequence], faces_smpl: torch.Tensor):
    output_path = Path(cfg.output_dir) / "multi_incam.mp4"
    if output_path.exists():
        if output_path.stat().st_size == 0:
            Log.warn(f"[Render Incam] Removing empty video at {output_path}")
            output_path.unlink(missing_ok=True)
        else:
            Log.info(f"[Render Incam] Video already exists at {output_path}")
            return

    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    first_sequence = next(iter(sequences.values()))
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=first_sequence.K_fullimg)
    reader = get_video_reader(video_path)
    writer = get_writer(str(output_path), fps=30, crf=CRF)

    colors = assign_track_colors(sequences.keys())
    track_ids = sorted(sequences.keys())

    try:
        for frame_idx, img_raw in tqdm(enumerate(reader), total=length, desc="Rendering Multi Incam"):
            composed = img_raw
            for track_id in track_ids:
                verts = sequences[track_id].verts_incam[frame_idx].to("cuda")
                composed = renderer.render_mesh(verts, composed, colors[track_id])
            writer.write_frame(composed)
    finally:
        writer.close()
        try:
            reader.close()
        except Exception:
            pass
        

def run_prediction_for_person(cfg, person_paths: PersonPaths, model: DemoPL):
    if person_paths.hmr4d_results.exists():
        Log.info(f"[HMR4D][Skip] Existing prediction at {person_paths.hmr4d_results}")
        return
    data = load_data_dict(cfg, person_paths)
    tic = Log.sync_time()
    pred = model.predict(data, static_cam=cfg.static_cam)
    pred = detach_to_cpu(pred)
    data_time = data["length"] / 30
    Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
    torch.save(pred, person_paths.hmr4d_results)


def main():
    cfg = parse_args_to_cfg()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA environment is required. torch.cuda.is_available() returned False.")
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    ensure_slam(cfg)

    tracker = Tracker()
    bbx_multi: Dict[int, torch.Tensor] = tracker.get_multi_track(cfg.video_path)

    person_paths_by_id: Dict[int, PersonPaths] = {}
    pending_inference: List[int] = []

    for track_id, bbx_xyxy in bbx_multi.items():
        Log.info(f"[Pipeline] Preparing track id {track_id}")
        person_paths = build_person_paths(cfg, track_id)
        run_preprocess_for_person(cfg, person_paths, track_id, bbx_xyxy)
        if not person_paths.hmr4d_results.exists():
            pending_inference.append(track_id)
        person_paths_by_id[track_id] = person_paths

    if pending_inference:
        Log.info(f"[Pipeline] Running HMR4D inference for tracks: {pending_inference}")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        for track_id in pending_inference:
            run_prediction_for_person(cfg, person_paths_by_id[track_id], model)
        del model
        torch.cuda.empty_cache()
    else:
        Log.info("[Pipeline] All HMR4D results found. Skipping inference.")

    if not person_paths_by_id:
        Log.warn("[Pipeline] No person tracks processed. Exiting.")
        return

    sequences, faces_smpl = prepare_person_sequences(person_paths_by_id)

    render_incam_multi(cfg, sequences, faces_smpl)

if __name__ == "__main__":
    main()
