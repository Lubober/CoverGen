import argparse
import os
import sys
from .lora_mapper import LoRAAudioToImageMapper

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

def generate_image(adapter_checkpoint,audio_embedding,output_image="generated.png"):

    # ---------------- Load Stable Diffusion Pipeline ----------------
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    pipe.unet.eval()
    pipe.vae.eval()
    for param in pipe.unet.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False

    # ---------------- Load Adapter ----------------
    print("Loading adapter checkpoint...")
    adapter = LoRAAudioToImageMapper(input_dim=512, output_dim=768).to("cuda")
    adapter.load_state_dict(torch.load(adapter_checkpoint))
    adapter.eval()

    # ---------------- Load Audio Embedding ----------------
    print("Loading audio embedding...")
    audio_emb = np.load(audio_embedding)  # shape could be [time, 512] or [512]
    audio_emb = torch.tensor(audio_emb, dtype=torch.float).to("cuda")

    # If the embedding has a time dimension, reduce it (here, simple mean pooling)
    if audio_emb.dim() == 2:  # [time, 512]
        audio_emb = audio_emb.mean(dim=0, keepdim=True)  # shape: [1, 512]
    else:
        audio_emb = audio_emb.unsqueeze(0)  # shape: [1, 512] if it was [512]

    # ---------------- Map Embedding to 768-Dim ----------------
    with torch.no_grad():
        mapped_emb = adapter(audio_emb)  # shape: [1, 768]

    # ---------------- Prepare Conditioning & Generate Image ----------------
    # Expand to [B, 77, 768] (half precision)
    audio_condition = mapped_emb.unsqueeze(1).repeat(1, 77, 1).half()  # shape: [1, 77, 768]

    print("Generating image from Stable Diffusion...")
    with torch.no_grad():
        result = pipe(prompt_embeds=audio_condition)
        image = result.images[0]

    # ---------------- Save the Output Image ----------------
    os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
    image.save(output_image)
    print(f"Saved generated image to {output_image}")

    # ---------------- Rename Output Image Based on Audio Embedding File ----------------
    # Get base name of the audio embedding file (e.g., "nvu938340.npy" -> "nvu938340")
    audio_base = os.path.splitext(os.path.basename(audio_embedding))[0]
    new_image_name = f"{audio_base}.png"
    new_image_path = os.path.join(os.path.dirname(output_image), new_image_name)
    
    os.rename(output_image, new_image_path)
    print(f"Renamed output image to {new_image_path}")

    # # ---------------- Move Evaluated Audio Embedding ----------------
    # # Define folder to store evaluated embeddings; create it if it doesn't exist
    # evaluated_folder = os.path.join(os.path.dirname(audio_embedding), "../evaluated_embeddings")
    # os.makedirs(evaluated_folder, exist_ok=True)
    # new_embedding_path = os.path.join(evaluated_folder, os.path.basename(audio_embedding))
    
    # os.rename(audio_embedding, new_embedding_path)
    # print(f"Moved audio embedding to {new_embedding_path}")