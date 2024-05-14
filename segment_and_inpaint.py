import torch
import argparse
from transformers import SamModel, SamProcessor

from PIL import Image
import requests
import numpy as np
import cv2
from torchvision import transforms

import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

print("STARTED SEG SUBPROCESS")

parser = argparse.ArgumentParser(description="Process an image with bounding boxes")
parser.add_argument("--image_path", type=str, help="Path to the input image")
parser.add_argument("--bounding_boxes", type=str, help="Bounding boxes in the format x1_y1_x2_y2")
parser.add_argument("--prompt", type=str, help="Prompt for Inpainting", default="")
args = parser.parse_args()


MASK_IMAGES_DIR = "./data/seg_masks/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Reshape the bounding boxes list into sets of four coordinates
bounding_boxes = [float(i) for i in args.bounding_boxes.split("_")]

print("SEGMENTING NOW")
print(bounding_boxes)

raw_image = Image.open(args.image_path).convert("RGB")

inputs = processor(raw_image, return_tensors="pt").to(device)
image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

input_boxes = [[bounding_boxes]]
 

inputs = processor(raw_image, input_boxes=[input_boxes], return_tensors="pt").to(device)
inputs["input_boxes"].shape

inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

print(scores)
print(masks[0][0][torch.argmax(scores)])

print("SEGMENTING DONE")

image_tensor = masks[0][0][torch.argmax(scores)].type(torch.uint8) * 255
image_tensor = image_tensor.unsqueeze(0)

to_pil = transforms.ToPILImage()
image_pil = to_pil(image_tensor[0])

# Save the PIL image
image_pil.save(MASK_IMAGES_DIR+"segmentation_mask_selected_node.png")

# =====================================================================

# INPAINTING BEGINS HERE

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pipeline = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
# )

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
)

pipeline = pipeline.to(device)

init_image = load_image(args.image_path)
mask_image = load_image(MASK_IMAGES_DIR+"segmentation_mask_selected_node.png")


prompt = "add background"

if args.prompt != "":
    prompt = args.prompt + ", high resolution, very detailed, " + args.prompt

image_pipeline_output = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=200, num_images_per_prompt = 3, strength = 1)

img_filename = args.image_path.split("/")[-1].split(".")[0]

inpainted_images = image_pipeline_output.images

for i in range(0,len(inpainted_images)):
    inpainted_image = inpainted_images[i]
    inpainted_image.save(f"./data/inpaint_ops/{img_filename}_inpainted_{i}.png")

