import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image




def denoise_image(sd_image_path,denoised_image_path):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()


    # prepare image
    init_image = Image.open(sd_image_path)

    prompt = "album cover high resolution"

    # pass prompt and image to pipeline
    image = pipeline(prompt, image=init_image, strength=0.5).images[0]
    # make_image_grid([init_image, image], rows=1, cols=2)

    image.show()
    image.save(denoised_image_path)