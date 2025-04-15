import os
import librosa
import openl3
import numpy as np
from tqdm import tqdm
import sys

def extract_embedding(audio_path,embedding_path):
    print("Extractig Embedding...")
    audio, sr = librosa.load(audio_path, sr=None)
    embeddings, _ = openl3.get_audio_embedding(
        audio, sr, input_repr="mel256", content_type="music", embedding_size=512
    )
    np.save(embedding_path, embeddings)

    print("Audio embedding extraction complete!")
