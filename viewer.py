"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import os
import time

import nerfview
import numpy as np
import torch
from arguments import GroupParams, ModelParams, PipelineParams
from gaussian_renderer.ever import splinerender
from scene.cameras import MiniCam
from scene.dataset_readers import ProjectionType
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.system_utils import searchForMaxIteration

import viser


def get_gaussian_model(dataset: ModelParams) -> GaussianModel:
    print("Getting Gaussian Model for dataset: ", dataset.model_path)
    selected_3dgs = GaussianModel(
        dataset.sh_degree, dataset.use_neural_network, dataset.max_opacity
    )

    loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    print("Loading trained model at iteration {}".format(loaded_iter))
    selected_3dgs.load_ply(
        os.path.join(
            dataset.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.ply",
        )
    )
    print("Loaded Gaussian Model")
    return selected_3dgs


def main(dataset: ModelParams, pp: GroupParams, port: int = 8080):
    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    server = viser.ViserServer(port=port, verbose=False)

    selected_3dgs = get_gaussian_model(dataset)

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()  # Original view matrix calculation

        if selected_3dgs is None:
            return np.zeros((height, width, 3))

        fovy = camera_state.fov  # Assuming camera_state.fov is the vertical FoV
        aspect_ratio = width / float(height)
        fovx = 2 * np.arctan(np.tan(fovy / 2.0) * aspect_ratio)

        # Calculate projection matrix
        projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)
            .transpose(0, 1)
            .to(device)
        )  # Using default znear/zfar from Camera class
        full_proj_transform = viewmat @ projection_matrix

        view = MiniCam(
            width, height, fovy, fovx, 0.01, 100.0, viewmat, full_proj_transform
        )  # Pass full_proj_transform and use actual znear/zfar
        #! Super hacky, but need to add some fields to the MiniCam class
        view.model = ProjectionType.PERSPECTIVE

        net_image = splinerender(
            view,
            selected_3dgs,
            pp,
            torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
        )["render"]
        net_image = (
            (torch.clamp(net_image, min=0, max=1.0) * 255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        # net_image = cv2.resize(net_image, (width, height))
        return net_image

    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    dataset = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    # parser.add_argument("--ckpt", type=str, default=None, help="path to the .pt file")
    parser.add_argument(
        "--port", type=int, default=8083, help="port for the viewer server"
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    # cli(main, args, verbose=True)
    main(dataset.extract(args), pp.extract(args), args.port)
