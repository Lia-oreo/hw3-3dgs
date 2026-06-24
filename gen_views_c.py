import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

model_path = "/root/autodl-tmp/object_b/sd-2-1-base/stabilityai/stable-diffusion-2-1-base"
image_path = "/root/autodl-tmp/object_c/input.png"
out_dir = "/root/autodl-tmp/object_c/views"
os.makedirs(out_dir, exist_ok=True)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
)
pipe.to("cuda")

cond = Image.open(image_path).convert("RGB").resize((512, 512))

prompts = [
    "milk carton, front view, white background",
    "milk carton, front-right view, white background",
    "milk carton, right side view, white background",
    "milk carton, back-right view, white background",
    "milk carton, back view, white background",
    "milk carton, back-left view, white background",
    "milk carton, left side view, white background",
    "milk carton, front-left view, white background",
]

for i, p in enumerate(prompts):
    print(f"Generating {i}: {p}")
    img = pipe(p, image=cond, strength=0.7, num_inference_steps=30, guidance_scale=7.5).images[0]
    img.save(os.path.join(out_dir, f"view_{i:02d}.png"))

print("Done:", out_dir)
