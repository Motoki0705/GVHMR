from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Any, Tuple
from einops import einsum

import torch
from hmr4d.utils.smplx_utils import make_smplx

HMR4D_RESULTS_PATH = Path("third_party/GVHMR/outputs/demo/tennis_clip/id_1/hmr4d_results.pt")

def to_cuda(data):
    """Move data in the batch to cuda(), carefully handle data that is not tensor"""
    if isinstance(data, torch.Tensor):
        return data.cuda()
    elif isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    else:
        return data


def _faces_to_tensor(faces_raw: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert model faces to an integer tensor on CPU."""
    if isinstance(faces_raw, np.ndarray):
        return torch.from_numpy(faces_raw.astype(np.int64))
    return torch.as_tensor(faces_raw, dtype=torch.int64)


def preprare_person(
    result_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load HMR4D outputs and prepare SMPL/SMPL-X meshes for visualization."""
    device = torch.device("cuda")
    smplx_model = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load("third_party/GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to(device)
    J_regressor = torch.load("third_party/GVHMR/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).to(device)
    # shape of smplx2smpl -> torch.Size([6890, 10475])
    # shape of J_regressor -> torch.Size([24, 6890])
    print(f"shape of smplx2smpl -> {smplx2smpl.shape}")
    print(f"shape of J_regressor -> {J_regressor.shape}")
    faces_smpl = _faces_to_tensor(make_smplx("smpl").faces)
    faces_smplx = _faces_to_tensor(smplx_model.faces)

    pred = torch.load(result_path, weights_only=False)
    smpl_params_incam = to_cuda(pred["smpl_params_incam"])
    smplx_out = smplx_model(**smpl_params_incam)
    
    verts_smpl = torch.stack([torch.matmul(smplx2smpl, verts) for verts in smplx_out.vertices])
    joints_smpl = einsum(J_regressor, verts_smpl, "j v, l v i -> l j i")
    
    K_fullimg = torch.as_tensor(pred["K_fullimg"][0]).float()
    return (
        verts_smpl.detach().cpu(),
        smplx_out.vertices.detach().cpu(),
        joints_smpl.detach().cpu(),
        K_fullimg,
        faces_smpl,
        faces_smplx,
    )


def _set_axes_equal(ax, vertices: np.ndarray) -> None:
    """Apply equal aspect ratio around the provided vertices."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_meshes_with_matplotlib(
    verts_smpl: torch.Tensor,
    faces_smpl: torch.Tensor,
    verts_smplx: torch.Tensor,
    faces_smplx: torch.Tensor,
    frame_idx: int = 0,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Render SMPL and SMPL-X meshes for one frame with matplotlib."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    num_frames = int(verts_smpl.shape[0])
    if num_frames == 0 or int(verts_smplx.shape[0]) == 0:
        raise ValueError("Mesh sequences must contain at least one frame.")
    if frame_idx < 0 or frame_idx >= num_frames or frame_idx >= int(verts_smplx.shape[0]):
        raise IndexError(f"frame_idx={frame_idx} is out of range for {num_frames} frames.")

    smpl_vertices = verts_smpl[frame_idx].detach().cpu().numpy()
    smplx_vertices = verts_smplx[frame_idx].detach().cpu().numpy()
    smpl_faces_np = faces_smpl.detach().cpu().numpy()
    smplx_faces_np = faces_smplx.detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 7))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    configs = [
        (axes[0], smpl_vertices, smpl_faces_np, "SMPL", "#4C78A8"),
        (axes[1], smplx_vertices, smplx_faces_np, "SMPL-X", "#F58518"),
    ]

    for ax, vertices, faces, title, color in configs:
        mesh = Poly3DCollection(vertices[faces], facecolor=color, edgecolor="none", alpha=0.9)
        ax.add_collection3d(mesh)
        _set_axes_equal(ax, vertices)
        ax.view_init(elev=20, azim=-70)
        ax.set_title(f"{title} frame={frame_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def animate_mesh_with_matplotlib(
    vertices_seq: torch.Tensor,
    faces: torch.Tensor,
    title: str,
    interval_ms: int = 30,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Render one mesh sequence as a matplotlib animation."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if int(vertices_seq.shape[0]) == 0:
        raise ValueError("Mesh sequence must contain at least one frame.")

    vertices_np = vertices_seq.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _set_axes_equal(ax, vertices_np.reshape(-1, 3))
    ax.view_init(elev=20, azim=-70)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    mesh = Poly3DCollection(vertices_np[0][faces_np], facecolor="#4C78A8", edgecolor="none", alpha=0.9)
    ax.add_collection3d(mesh)
    title_text = ax.set_title(f"{title} frame=0")

    def update(frame_idx: int):
        mesh.set_verts(vertices_np[frame_idx][faces_np])
        title_text.set_text(f"{title} frame={frame_idx}")
        return mesh, title_text

    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=vertices_np.shape[0],
        interval=interval_ms,
        blit=False,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

        
def main():
    verts_smpl, verts_smplx, joints_smpl, K_fullimg, faces_smpl, faces_smplx = preprare_person(HMR4D_RESULTS_PATH)
    # shape of verts_smpl -> torch.Size([325, 6890, 3])
    # shape of verts_smplx -> torch.Size([325, 10475, 3])
    # shape of joints_smpl -> torch.Size([325, 24, 3])
    # shape of K_fullimg -> torch.Size([3, 3])
    # shape of faces_smpl -> torch.Size([13776, 3])
    # shape of faces_smplx -> torch.Size([20908, 3])
    print(f"shape of verts_smpl -> {verts_smpl.shape}")
    print(f"shape of verts_smplx -> {verts_smplx.shape}")
    print(f"shape of joints_smpl -> {joints_smpl.shape}")
    print(f"shape of K_fullimg -> {K_fullimg.shape}")
    print(f"shape of faces_smpl -> {faces_smpl.shape}")
    print(f"shape of faces_smplx -> {faces_smplx.shape}")

    animate_mesh_with_matplotlib(
        vertices_seq=verts_smpl,
        faces=faces_smpl,
        title="SMPL",
        interval_ms=30,
        save_path="third_party/GVHMR/analysis/inference_analysis/verts_smpl_animation.gif",
        show=False
    )
        
    

if __name__ == "__main__":
    main()
