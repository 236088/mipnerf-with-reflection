import torch
import os
from os import path
from config import get_config
from model import ReflectNeRF
import imageio
from datasets import get_dataloader
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_fake_normals, visualize_real_normals, to8b


def visualize(config):
    data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)

    model = ReflectNeRF(
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()

    img_dir = path.join(config.log_dir, "img")
    if not path.exists(img_dir):
        os.makedirs(img_dir)
    if config.visualize_depth:
        depth_dir = path.join(config.log_dir, "depth")
        if not path.exists(depth_dir):
            os.makedirs(depth_dir)
    if config.visualize_normals:
        normal_dir = path.join(config.log_dir, "normal")
        if not path.exists(normal_dir):
            os.makedirs(normal_dir)
    
    print("Generating Video using", len(data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []
    for i, ray in enumerate(tqdm(data)):
        img, dist, normal, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)
        imageio.imwrite(path.join(img_dir, f"{i:03}.png"), img)
        rgb_frames.append(img)
        if config.visualize_depth:
            depth = to8b(visualize_depth(dist, acc, data.near, data.far))
            depth_frames.append(depth)
            imageio.imwrite(path.join(depth_dir, f"{i:03}.png"), depth)
        if config.visualize_normals:
            normal = to8b(visualize_real_normals(normal, acc))
            normal_frames.append(normal)
            imageio.imwrite(path.join(normal_dir, f"{i:03}.png"), normal)

    imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=30, quality=10, codecs="hvec")
    if config.visualize_depth:
        imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=30, quality=10, codecs="hvec")
    if config.visualize_normals:
        imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=30, quality=10, codecs="hvec")


if __name__ == "__main__":
    config = get_config()
    visualize(config)
