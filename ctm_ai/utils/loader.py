import base64
import os
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def load_audio(audio_path: str) -> Tuple[NDArray[np.float32], int]:
    import librosa

    audio, sr = librosa.load(audio_path, sr=None)
    return (audio.astype(np.float32), int(sr))


def load_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image


def load_images(image_paths: List[str]) -> List[Image.Image]:
    if not image_paths:
        raise ValueError('No images provided')

    image_paths_sorted = sorted(image_paths)

    images = []
    for img_path in image_paths_sorted:
        try:
            image = Image.open(img_path)
            images.append(image)
        except Exception as e:
            raise RuntimeError(f'Failed to load image {img_path}: {e}')
    return images


def load_video(video_path: str, frame_num: int = 5) -> List[NDArray[np.uint8]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: List[NDArray[np.uint8]] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.astype(np.uint8))
    finally:
        cap.release()

    if len(frames) >= frame_num:
        step = len(frames) // frame_num
        frames = [frames[i] for i in range(0, len(frames), step)]
    return frames


def extract_video_frames(
    video_path: str,
    output_dir: str,
    max_frames: Optional[int] = None,
) -> List[str]:
    import os

    import cv2

    cap = cv2.VideoCapture(video_path)
    frame_list = []
    os.makedirs(output_dir, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[attr-defined]

    if max_frames is not None:
        sample_indices = [int(total_frames * i / max_frames) for i in range(max_frames)]
    else:
        sample_indices = list(range(total_frames))

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_index >= total_frames:
                break

            if frame_index in sample_indices:
                frame_filename = os.path.join(
                    output_dir, f'frame_{frame_index:05d}.jpg'
                )
                cv2.imwrite(frame_filename, frame)  # type: ignore[attr-defined]
                frame_list.append(frame_filename)

            frame_index += 1

    finally:
        cap.release()
    return frame_list


def extract_audio_from_video(
    video_path: str, output_dir: str, audio_format: str = 'mp3'
) -> str:
    from moviepy import VideoFileClip

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video not found: {video_path}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = VideoFileClip(video_path)
    audio = video.audio

    if audio is None:
        raise ValueError('This video has no audio')

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    audio_filename = f'{base_filename}_audio.{audio_format}'
    audio_path = os.path.join(output_dir, audio_filename)

    audio.write_audiofile(audio_path, logger=None)

    audio.close()
    video.close()

    return audio_filename
