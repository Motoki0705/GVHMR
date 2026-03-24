import logging
from pathlib import Path
from typing import Any

import torch

HMR4D_RESULTS_PATH = Path("third_party/GVHMR/outputs/demo/tennis_clip/id_1/hmr4d_results.pt")
logger = logging.getLogger(name=__name__)
logging.basicConfig(level=logging.INFO)

def load_hmr4d_result(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")

def load_smpl_model(path: Path) -> dict[str, Any]:
    return 0

def tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device)
    }
    return meta

def main():
    obj = load_hmr4d_result(HMR4D_RESULTS_PATH)
    logger.info(
        f"keys of hmr4d results -> {obj.keys()} \n"
    ) # ['smpl_params_global', 'smpl_params_incam', 'K_fullimg', 'net_outputs']
    
    # トップレベルのパラメータについて、shapeを調べる。
    # INFO:__main__:body_pose -> {'shape': [325, 63], 'dtype': 'torch.float32', 'device': 'cpu'}
    # INFO:__main__:betas -> {'shape': [325, 10], 'dtype': 'torch.float32', 'device': 'cpu'}
    # INFO:__main__:global_orient -> {'shape': [325, 3], 'dtype': 'torch.float32', 'device': 'cpu'}
    # INFO:__main__:transl -> {'shape': [325, 3], 'dtype': 'torch.float32', 'device': 'cpu'}
    smpl_params_global = obj["smpl_params_global"]
    logger.info("meta of smpl_params_global")
    for name, param in smpl_params_global.items():
        param_meta = tensor_meta(param)
        logger.info(
            f"{name} -> {param_meta}"
        )
    
    # betasに関して一貫性を調べる
    batas_std = smpl_params_global["betas"].std(dim=0)
    print()
    logger.info(
        f"batas std -> {batas_std}\n"
    ) # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    
    # body_poseを用いて可視化する

if __name__ == "__main__":
    main()