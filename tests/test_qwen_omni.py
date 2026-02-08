import os
import base64
import subprocess
import tempfile
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# URFunny sample 408
# Context: "so yeah i'm a newspaper cartoonist political cartoonist"
# Punchline: "i don't know if you've heard about it newspapers it's a sort of paper based reader"
# Label: 1 (humorous)
SAMPLE_KEY = "408"
# 直接使用已有的文件
AUDIO_PATH = f"../exp_urfunny/urfunny_audios/{SAMPLE_KEY}_audio.mp4"
VIDEO_NO_AUDIO_PATH = f"../exp_urfunny/urfunny_muted_videos/{SAMPLE_KEY}.mp4"
VIDEO_WITH_AUDIO_PATH = f"../exp_urfunny/urfunny_videos/{SAMPLE_KEY}.mp4"
TEXT_CONTEXT = (
    "Context: so yeah i'm a newspaper cartoonist political cartoonist\n"
    "Punchline: i don't know if you've heard about it newspapers it's a sort of paper based reader"
)

QUESTION = (
    "Please analyze the inputs provided to determine the punchline provided humor or not. "
    "Answer 'Yes' or 'No' and provide your reason."
)


def stream_response(completion):
    """Collect streamed response into a single string."""
    full_text = ""
    usage_info = None
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            full_text += chunk.choices[0].delta.content
        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage_info = chunk.usage
    return full_text, usage_info


def make_black_video_with_audio(audio_path, output_path):
    """Convert audio-only mp4 to a video mp4 with a black video track using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=1",
        "-i",
        audio_path,
        "-shortest",
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False
    return True


def encode_video_base64(video_path):
    """Read a video file and return base64 encoded string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_qwen_with_video(video_b64, prompt):
    """Call Qwen-Omni with a video (base64) and text prompt."""
    completion = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )
    return stream_response(completion)


# =============================================================================
# TEST 1: Text Only
# =============================================================================
def test_text_only():
    print("=" * 60)
    print("TEST 1: TEXT ONLY")
    print("=" * 60)
    completion = client.chat.completions.create(
        model="qwen3-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{QUESTION}\n\n{TEXT_CONTEXT}"},
                ],
            }
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )
    text, usage = stream_response(completion)
    print(f"Response:\n{text}")
    if usage:
        print(f"\nUsage: {usage}")
    print()


# =============================================================================
# TEST 2: Video Only (no audio) — 使用已有的静音视频文件
# =============================================================================
def test_video_only():
    print("=" * 60)
    print("TEST 2: VIDEO ONLY (no audio) + TEXT")
    print("=" * 60)

    if not os.path.exists(VIDEO_NO_AUDIO_PATH):
        print(f"Video file not found: {VIDEO_NO_AUDIO_PATH}")
        return

    print(f"Using existing video: {VIDEO_NO_AUDIO_PATH}")
    print(f"Video file size: {os.path.getsize(VIDEO_NO_AUDIO_PATH)} bytes")
    video_b64 = encode_video_base64(VIDEO_NO_AUDIO_PATH)

    prompt = f"Focus ONLY on the visual content. Ignore any audio.\n{QUESTION}"
    text, usage = call_qwen_with_video(video_b64, prompt)
    print(f"Response:\n{text}")
    if usage:
        print(f"\nUsage: {usage}")
    print()


# =============================================================================
# TEST 3: Audio Only — 使用已有的音频文件 + 黑屏视频
# =============================================================================
def test_audio_only():
    print("=" * 60)
    print("TEST 3: AUDIO ONLY (black screen video) + TEXT")
    print("=" * 60)

    if not os.path.exists(AUDIO_PATH):
        print(f"Audio file not found: {AUDIO_PATH}")
        return

    tmp_video = tempfile.mktemp(suffix=".mp4")
    print(f"Converting {AUDIO_PATH} -> black screen video...")
    if not make_black_video_with_audio(AUDIO_PATH, tmp_video):
        print("Failed to convert audio to video")
        return

    print(f"Video file size: {os.path.getsize(tmp_video)} bytes")
    video_b64 = encode_video_base64(tmp_video)

    try:
        prompt = f"Focus ONLY on what the person is SAYING in the audio. The video is just a black screen.\n{QUESTION}"
        text, usage = call_qwen_with_video(video_b64, prompt)
        print(f"Response:\n{text}")
        if usage:
            print(f"\nUsage: {usage}")
    finally:
        os.unlink(tmp_video)
    print()


def test_all_modalities():
    print("=" * 60)
    print("TEST 4: ALL MODALITIES (Video + Audio + Text)")
    print("=" * 60)

    if not os.path.exists(VIDEO_WITH_AUDIO_PATH):
        print(f"Video file not found: {VIDEO_WITH_AUDIO_PATH}")
        return

    print(f"Using existing video with audio: {VIDEO_WITH_AUDIO_PATH}")
    print(f"Video file size: {os.path.getsize(VIDEO_WITH_AUDIO_PATH)} bytes")
    video_b64 = encode_video_base64(VIDEO_WITH_AUDIO_PATH)

    prompt = (
        f"Analyze BOTH the visual content and the audio together.\n"
        f"{QUESTION}\n\n{TEXT_CONTEXT}"
    )
    text, usage = call_qwen_with_video(video_b64, prompt)
    print(f"Response:\n{text}")
    if usage:
        print(f"\nUsage: {usage}")
    print()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Sample: {SAMPLE_KEY}")
    print(f"Text: {TEXT_CONTEXT}")
    print(f"Audio: {AUDIO_PATH} (exists: {os.path.exists(AUDIO_PATH)})")
    print(
        f"Video (no audio): {VIDEO_NO_AUDIO_PATH} (exists: {os.path.exists(VIDEO_NO_AUDIO_PATH)})"
    )
    print(
        f"Video (with audio): {VIDEO_WITH_AUDIO_PATH} (exists: {os.path.exists(VIDEO_WITH_AUDIO_PATH)})"
    )
    print()

    test_text_only()
    test_video_only()
    test_audio_only()
    test_all_modalities()
