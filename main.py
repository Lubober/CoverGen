from CoverGen import extract_chorus,extract_embedding,generate_image, denoise_image

song_name = "never_alone"

input_mp3 = f"./data/audios/{song_name}.mp3"
output_mp3 = f"./data/ext_chorus/{song_name}.mp3"

checkpoint = "./checkpoints/covergen_best.pth"

embeding_path = f"./data/embeddings/{song_name}.npy"

gen_image_dir = f"./data/sd_img/{song_name}.png"

img2img_path = f"./data/img2img/{song_name}.png"

extract_chorus(input_mp3=input_mp3,output_mp3=output_mp3)

extract_embedding(output_mp3,embeding_path)


generate_image(checkpoint,embeding_path, gen_image_dir)

denoise_image(gen_image_dir,img2img_path)


print("done")