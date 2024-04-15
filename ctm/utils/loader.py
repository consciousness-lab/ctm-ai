def load_image(image_path):  # type: ignore[no-untyped-def] # FIX ME
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_audio(audio_path):
    import librosa

    # load audio from audio
    audio, sr = librosa.load(audio_path)
    return audio


def load_video(video_path, frame_num=5):
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Here, you might save or process the frame
            frames.append(frame)
    finally:
        cap.release()
    frames = [
        frames[i] for i in range(0, len(frames), len(frames) // frame_num)
    ]
    return frames
