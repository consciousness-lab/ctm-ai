import base64
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray


def load_audio(audio_path: str) -> Tuple[NDArray[np.float32], int]:
    import librosa

    audio, sr = librosa.load(audio_path, sr=None)
    import pdb

    pdb.set_trace()  # Debugging breakpoint
    return (audio.astype(np.float32), int(sr))


def load_image(image_path: str) -> str:
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_image


def load_video(video_path: str, frame_num: int = 5) -> List[NDArray[np.uint8]]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray[np.uint8, Any]] = []
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
