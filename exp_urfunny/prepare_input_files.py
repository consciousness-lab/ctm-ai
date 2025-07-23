import json
import os
import subprocess

from ctm_ai.utils import extract_video_frames


def prepare_frames(
    file_list_path: str, source_folder: str, target_folder: str, max_frames: int
) -> None:
    with open(file_list_path, "r", encoding="utf-8") as json_file:
        sample_list = json.load(json_file)
    file_list = []
    for key in sample_list:
        file_list.append(key)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for test_file in file_list:
        video_path = f"{source_folder}/{test_file}.mp4"
        video_frames_path = f"{target_folder}/{test_file}_frames"
        extract_video_frames(video_path, video_frames_path, max_frames)


def prepare_audios(file_list_path: str, input_folder: str, output_folder: str) -> None:
    with open(file_list_path, "r", encoding="utf-8") as json_file:
        sample_list = json.load(json_file)
    file_list = []
    for key in sample_list:
        file_list.append(key)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        base_name = os.path.splitext(file_name)[0]
        if base_name in file_list:
            input_path = os.path.join(input_folder, file_name)
            output_file = f"{base_name}_audio.mp4"
            output_path = os.path.join(output_folder, output_file)

            command = [
                "ffmpeg",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "copy",
                output_path,
            ]

            print(f"processing file: {file_name}")
            try:
                subprocess.run(command, check=True)
                print(f"audio saved: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"fail in extracting audio: {file_name}, error: {e}")


if __name__ == "__main__":
    video_folder = "urfunny2_videos"
    audio_folder = "test_inputs/urfunny_audios"
    frames_folder = "test_inputs/urfunny_frames"
    sample_data = "data_raw/urfunny_dataset_test.json"
    prepare_frames(sample_data, video_folder, frames_folder, 5)
    # prepare_audios(sample_data, video_folder, audio_folder)
