from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

print("Loading InstructPix2Pix model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
)

pipe = pipe.to(device)

print(f"Using device: {device}")

# Load input image
image = Image.open("input.jpg").convert("RGB")

# Translation instruction
prompt = """
Transform this image into a futuristic cyberpunk version.
Add neon lights, glowing buildings, advanced technology,
vibrant colors, and a sci-fi atmosphere while preserving
the main structure of the original image.
"""

print("Generating translated image...")

result = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=25,
    image_guidance_scale=1.5,
    guidance_scale=7.5
).images[0]

# Save output
result.save("output.png")

print("✅ Image translation completed!")
print("✅ Output saved as output.png")