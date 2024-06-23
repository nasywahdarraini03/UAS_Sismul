import streamlit as st
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from pydub import AudioSegment
import time
import cv2
import os

# Function to compress image using DCT
def compress_image_dct(image, quality):
    image = np.array(image)
    dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')
    dct_image = np.round(dct_image / quality) * quality
    idct_image = idct(idct(dct_image.T, norm='ortho').T, norm='ortho')
    return Image.fromarray(np.clip(idct_image, 0, 255).astype(np.uint8))

# Placeholder function for fractal compression (not a real implementation)
def compress_image_fractal(image):
    return image

# Function to compress audio using DCT
def compress_audio_dct(audio, quality):
    dct_audio = dct(audio, norm='ortho')
    dct_audio = np.round(dct_audio / quality) * quality
    idct_audio = idct(dct_audio, norm='ortho')
    return idct_audio

# Placeholder function for fractal compression (not a real implementation)
def compress_audio_fractal(audio):
    return audio

# Function to compress video using DCT
def compress_video_dct(video_path, output_path, quality):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dct_frame = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            dct_frame = np.round(dct_frame / quality) * quality
            idct_frame = idct(idct(dct_frame.T, norm='ortho').T, norm='ortho')
            out.write(cv2.cvtColor(np.clip(idct_frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        else:
            break
    cap.release()
    out.release()

# Placeholder function for fractal compression (not a real implementation)
def compress_video_fractal(video_path, output_path):
    os.system(f"cp {video_path} {output_path}")

def file_size(file_path):
    return os.path.getsize(file_path)

st.title("Image, Audio, and Video Compression")

uploaded_files = st.file_uploader("Upload Images, Audio, or Video", accept_multiple_files=True)
compression_algorithm = st.selectbox("Select Compression Algorithm", ["DCT", "Fractal"])
quality = st.slider("Compression Quality", 1, 100, 50)
audio_bitrate = st.slider("Audio Bitrate (kbps)", 32, 320, 128)

for uploaded_file in uploaded_files:
    if uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)
        
        # Save original image
        original_path = "original_image.png"
        image.save(original_path)
        original_size = file_size(original_path)
        
        start_time = time.time()
        if compression_algorithm == "DCT":
            compressed_image = compress_image_dct(image, quality)
        else:
            compressed_image = compress_image_fractal(image)
        compression_time = time.time() - start_time
        
        # Save compressed image
        compressed_path = "compressed_image.png"
        compressed_image.save(compressed_path)
        compressed_size = file_size(compressed_path)

        st.image(compressed_image, caption='Compressed Image', use_column_width=True)
        st.write(f"Compression Time: {compression_time:.2f} seconds")
        st.write(f"Original Size: {original_size / 1024:.2f} KB")
        st.write(f"Compressed Size: {compressed_size / 1024:.2f} KB")
    
    elif uploaded_file.type.startswith("audio/"):
        audio_format = uploaded_file.type.split('/')[1]
        audio_path = f"original_audio.{audio_format}"
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.read())
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        st.audio(uploaded_file)
        
        # Save original audio
        original_path = "original_audio.wav"
        audio.export(original_path, format="wav")
        original_size = file_size(original_path)
        
        start_time = time.time()
        if compression_algorithm == "DCT":
            compressed_audio = compress_audio_dct(samples, quality)
        else:
            compressed_audio = compress_audio_fractal(samples)
        compression_time = time.time() - start_time
        
        # Convert back to AudioSegment
        compressed_audio_segment = AudioSegment(
            compressed_audio.tobytes(), 
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width, 
            channels=audio.channels
        )
        
        # Save compressed audio
        compressed_path = "compressed_audio.mp3"
        compressed_audio_segment.export(compressed_path, format="mp3", bitrate=f"{audio_bitrate}k")
        compressed_size = file_size(compressed_path)

        st.audio(compressed_path)
        st.write(f"Compression Time: {compression_time:.2f} seconds")
        st.write(f"Original Size: {original_size / 1024:.2f} KB")
        st.write(f"Compressed Size: {compressed_size / 1024:.2f} KB")
    
    elif uploaded_file.type.startswith("video/"):
        video_path = uploaded_file.name
        with open(video_path, mode='wb') as f:
            f.write(uploaded_file.read())
        
        st.video(video_path)
        
        # Save original video
        original_path = video_path
        original_size = file_size(original_path)
        
        start_time = time.time()
        output_path = "compressed_video.avi"
        if compression_algorithm == "DCT":
            compress_video_dct(video_path, output_path, quality)
        else:
            compress_video_fractal(video_path, output_path)
        compression_time = time.time() - start_time
        
        compressed_size = file_size(output_path)

        st.video(output_path)
        st.write(f"Compression Time: {compression_time:.2f} seconds")
        st.write(f"Original Size: {original_size / (1024 * 1024):.2f} MB")
        st.write(f"Compressed Size: {compressed_size / (1024 * 1024):.2f} MB")




