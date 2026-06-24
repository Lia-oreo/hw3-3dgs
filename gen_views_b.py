import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

model_path = "/root/autodl-tmp/object_b/sd-2-1-base/stabilityai/stable-diffusion-2-1-base"
out_dir = "/root/autodl-tmp/object_b/views"
os.makedirs(out_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
)
pipe.to("cuda")

base_prompt = "a red apple"
prompts = [
    f"{base_prompt}, front view, white background",
    f"{base_prompt}, front-right view, white background",
    f"{base_prompt}, right side view, white background",
    f"{base_prompt}, back-right view, white background",
    f"{base_prompt}, back view, white background",
    f"{base_prompt}, back-left view, white background",
    f"{base_prompt}, left side view, white background",
    f"{base_prompt}, front-left view, white background",
]

for i, p in enumerate(prompts):
    print(f"Generating {i}: {p}")
    img = pipe(p, num_inference_steps=30, guidance_scale=7.5, height=512, width=512).images[0]
    img.save(os.path.join(out_dir, f"view_{i:02d}.png"))

print("Done:", out_dir)
