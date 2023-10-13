import argparse
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Text to Image Generation")
parser.add_argument("text", type=str, help="Input text for image generation")
args = parser.parse_args()

# Download stable diffusion model from Hugging Face

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
try:
    stable_diffusion_model = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16")
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load the model:", str(e))

# Generate image from text
with autocast(device):
    image = stable_diffusion_model(args.text, guidance_scale=8.5)["sample"][0]

# Save the generated image with a unique filename
num = 1
while True:
    output_filename = f"output{num}.png"
    if not os.path.exists(output_filename):
        break
    num += 1

image.save(output_filename)

print(f"Image saved as {output_filename}")
