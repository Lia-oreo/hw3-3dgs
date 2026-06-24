import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from PIL import Image
from torchvision.utils import save_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render

def get_rotation_matrix(axis, angle):
    # 绕 axis (x/y/z) 旋转 angle 弧度
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.eye(3)

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/merge")
parser.add_argument("--output_path", type=str, default="/root/autodl-tmp/merge/video")
parser.add_argument("--num_frames", type=int, default=120)
parser.add_argument("--radius", type=float, default=4.0)
parser.add_argument("--height", type=float, default=1.5)
args = parser.parse_args()

gaussians = GaussianModel(sh_degree=3)
gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_7000", "point_cloud.ply"))

bg_color = [1, 1, 1]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

pipe = Namespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False, antialiasing=False)

os.makedirs(args.output_path, exist_ok=True)

for i in range(args.num_frames):
    angle = 2 * np.pi * i / args.num_frames
    cam_pos = np.array([args.radius * np.cos(angle), args.height, args.radius * np.sin(angle)])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(np.array([0, 1, 0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R_w2c = np.stack([right, up, forward], axis=0)
    
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    
    cam = Camera(
        colmap_id=i,
        R=R_w2c,
        T=cam_pos,
        FoVx=np.deg2rad(60),
        FoVy=np.deg2rad(60),
        image=image,
        
        image_name=f"frame_{i:04d}",
        uid=i,
        data_device="cuda",
        resolution=(512, 512),
        depth_params=None,
        invdepthmap=None
    )
    
    with torch.no_grad():
        rendering = render(cam, gaussians, pipe, background)["render"]
    save_image(rendering, os.path.join(args.output_path, f"frame_{i:04d}.png"))
    print(f"Rendered {i+1}/{args.num_frames}")

# 合成视频
os.system(f"ffmpeg -y -framerate 30 -i {args.output_path}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {args.output_path}/video.mp4")
print("Video saved to", os.path.join(args.output_path, "video.mp4"))
