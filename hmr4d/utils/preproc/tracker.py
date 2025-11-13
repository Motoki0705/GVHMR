from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth
from hmr4d.utils.ui import select_track_ids


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def _build_track_tensor(self, frame_ids, bbx_xyxys, num_frames):
        frame_ids = torch.tensor(frame_ids)
        bbx_xyxys = torch.tensor(bbx_xyxys)
        mask = frame_id_to_mask(frame_ids, num_frames)
        bbx_xyxy_track = rearrange_by_mask(bbx_xyxys, mask)
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)
        bbx_xyxy_track = linear_interpolate_frame_ids(bbx_xyxy_track, missing_frame_id_list)
        assert (bbx_xyxy_track.sum(1) != 0).all()
        bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
        bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
        return bbx_xyxy_track

    def get_one_track(self, video_path):
        track_history = self.track(video_path)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        track_id = id_sorted[0]
        bbx_xyxy_one_track = self._build_track_tensor(
            id_to_frame_ids[track_id],
            id_to_bbx_xyxys[track_id],
            get_video_lwh(video_path)[0],
        )
        return bbx_xyxy_one_track

    def get_multi_track(self, video_path):
        track_history = self.track(video_path)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        selected_ids = select_track_ids(track_history, video_path, id_sorted)
        num_frames = get_video_lwh(video_path)[0]
        multi_track = {}
        for track_id in selected_ids:
            if track_id not in id_to_frame_ids:
                continue
            bbx_xyxy_track = self._build_track_tensor(id_to_frame_ids[track_id], id_to_bbx_xyxys[track_id], num_frames)
            multi_track[int(track_id)] = bbx_xyxy_track

        if not multi_track:
            raise RuntimeError("No valid tracks selected for multi-person inference.")
        return multi_track
