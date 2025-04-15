import argparse
import os
import librosa
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

def compute_window_score(y, sr):
    """
    Compute a score for a window of audio using its RMS energy and spectral contrast.
    The idea is that a chorus (or eventful segment) will be loud and have a rich spectrum.
    """
    # Compute RMS energy (returns shape (1, frames))
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    
    # Compute spectral contrast (returns shape (n_bands+1, frames))
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mean_contrast = np.mean(spec_contrast)
    
    # Combine the metrics (multiplication emphasizes windows high on both metrics)
    score = mean_rms * mean_contrast
    return score

def find_best_segment(y, sr, window_duration=30.0, step_duration=1.0):
    """
    Slide a window of duration window_duration (seconds) over the audio with steps of step_duration (seconds),
    and return the start time (in seconds) of the window with the highest score.
    """
    total_samples = len(y)
    window_length = int(window_duration * sr)
    step_length = int(step_duration * sr)
    
    best_score = -np.inf
    best_start_sample = 0

    # Slide the window across the audio
    for start in range(0, total_samples - window_length + 1, step_length):
        window = y[start:start + window_length]
        score = compute_window_score(window, sr)
        if score > best_score:
            best_score = score
            best_start_sample = start

    best_start_time = best_start_sample / sr
    return best_start_time, best_score

def extract_chorus(input_mp3,output_mp3,window_duration=30.0,step_duration=1.0):

    # Load the audio file using librosa
    print(f"Loading audio file: {input_mp3}")
    y, sr = librosa.load(input_mp3, sr=None)
    total_duration = len(y) / sr
    print(f"Total duration: {total_duration:.2f}s")
    
    if total_duration < window_duration:
        print("Error: The input song is shorter than the desired window duration.")
        return

    # Analyze the song with a sliding window to find the best segment
    print("Analyzing audio to find the most eventful 30s segment...")
    best_start_time, best_score = find_best_segment(y, sr, window_duration=window_duration, step_duration=step_duration)
    print(f"Best segment starts at {best_start_time:.2f}s (score: {best_score:.4f})")
    
    # Use PyDub to load and extract the segment
    song = AudioSegment.from_mp3(input_mp3)
    start_ms = int(best_start_time * 1000)
    end_ms = start_ms + int(window_duration * 1000)
    extracted_segment = song[start_ms:end_ms]
    
    # Save the extracted segment
    os.makedirs(os.path.dirname(output_mp3), exist_ok=True)
    extracted_segment.export(output_mp3, format="mp3")
    print(f"Extracted 30s segment saved to {output_mp3}")
