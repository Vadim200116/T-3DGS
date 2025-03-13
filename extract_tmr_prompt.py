import os

from scene import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from dataclasses import dataclass
from scene.dataset_readers import sceneLoadTypeCallbacks
from argparse import ArgumentParser
from arguments import PipelineParams
import torch
from utils.general_utils import prep_img
from utils.transient_utils import DinoFeatureExatractor, dilate_mask, FeatUp_FeatureExtractor, ConvDPT
from tqdm import tqdm 
from gaussian_renderer import render
from PIL import Image
import torch.nn.functional as F

from copy import deepcopy

@dataclass
class Arguments:
    resolution: int
    data_device: str = "cuda"
    train_test_exp = False


class GaussianSplattingScene:
    def __init__(self, source_path, resolution):
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", "", "", eval=True, train_test_exp=False)
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1, Arguments(resolution), False)


class GaussianSplattingModel:
    @property
    def num_predictions(self):
        return len(self.train_cams)

    def __init__(self, model_path, scene: GaussianSplattingScene, iteration, dino_version):
        self.gaussians = GaussianModel(sh_degree=3)
        self.iteration = iteration
        if iteration:
            self.gaussians.load_ply(os.path.join(model_path, "point_cloud", "iteration_" + str(iteration),"point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene.info.point_cloud, scene.info.train_cameras, scene.info.nerf_normalization["radius"])

        self.train_cams = scene.train_cameras

        parser = ArgumentParser(description="Training script parameters")
        self.pipeline = PipelineParams(parser)

        self.feature_extractor = DinoFeatureExatractor(dino_version)
        transient_path = model_path + f"/transient_{iteration}.pth"
        self.featup_extractor = FeatUp_FeatureExtractor()
        
        self.transient_model = ConvDPT(self.feature_extractor.dino_model.embed_dim).cuda()
        self.transient_model.load_state_dict(torch.load(transient_path))
        self.transient_model.eval()

    def render_image(self, idx, do_prep=False):
        view  = deepcopy(self.train_cams[idx])
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        rendering = render(view, self.gaussians, self.pipeline, background)["render"]

        return prep_img(rendering) if do_prep else rendering

    def save_diffs(self, diffs_dir):
        for frame_idx, cam in tqdm(enumerate(self.train_cams)):
            gt_image = cam.original_image
            image = self.render_image(frame_idx)
            diff = torch.abs(gt_image - image)

            diff_np = prep_img(diff)
            Image.fromarray(diff_np).save(os.path.join(diffs_dir, f"{frame_idx:05d}.png"))

    def save_masks(self, masks_dir):
        for frame_idx, cam in tqdm(enumerate(self.train_cams)):
            gt_image = cam.original_image
            image = self.render_image(frame_idx)

            transient_input = torch.cat((gt_image.unsqueeze(0), image.detach().unsqueeze(0)))
            features = self.featup_extractor.extract(transient_input)
            d_cos =  1-F.cosine_similarity(features[0].permute(1,2,0), features[1].permute(1,2,0), dim=2).reshape(-1, 1, 256, 256)
            d_cos = F.interpolate(
                d_cos,
                size=(gt_image.shape[1], gt_image.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            transient_maps = self.transient_model(transient_input, self.feature_extractor)
            sigma_1 = transient_maps[0][0]
            sigma_2 = transient_maps[0][1]
            rho     = transient_maps[0][2]
            s_1 = transient_maps[1][0]
            s_2 = transient_maps[1][1]
            r   = transient_maps[1][2]

            det = sigma_1.square()*sigma_2.square()*(1.-rho.square())
            det_render = torch.clamp(s_1.square()*s_2.square()*(1 - r.square()), min=1e-6)

            KL = ((sigma_1.square()/torch.clamp(s_1.square(), min=1e-6) \
                + sigma_2.square()/torch.clamp(s_2.square(), min=1e-6) \
                - rho*r*sigma_1*sigma_2/(s_1*s_2))/torch.clamp(1. - rho.square(), min=1e-6) \
                - torch.log(det/det_render) - 2)*0.5

            mask = (KL > 30).detach().cpu().numpy()    
            Image.fromarray(~mask).save(os.path.join(masks_dir, f"{frame_idx:05d}.png"))


def main():

    parser = ArgumentParser(description="Gaussian Splatting Script")
    parser.add_argument("--source_path", type=str, required=True,
                      help="Path to the source directory")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the model directory")
    parser.add_argument("--resolution", type=int, required=True,
                      help="Resolution value")
    parser.add_argument("--iteration", type=int, required=True,
                      help="Iteration number")
    parser.add_argument("--output_path", type=str, default="",
                      help="Output path (optional)")
    parser.add_argument("--dino_version", type=str, default="dinov2_vits14",
                      help="DINO version (default: dinov2_vits14)")

    args = parser.parse_args()

    output_path = args.output_path or os.path.join(args.model_path, "prompt")

    masks_dir = os.path.join(output_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    diffs_dir = os.path.join(output_path, "diffs")
    os.makedirs(diffs_dir, exist_ok=True)

    scene = GaussianSplattingScene(args.source_path, args.resolution)

    model = GaussianSplattingModel(
        model_path=args.model_path,
        scene=scene,
        iteration=args.iteration,
        dino_version=args.dino_version,
    )

    model.save_masks(masks_dir)
    model.save_diffs(diffs_dir)

if __name__ == "__main__":
    main()
