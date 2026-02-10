"""
Prepare input files (video frames, audio extraction, etc.)

Supports preprocessing for multiple datasets
"""

import argparse
import json
import os
import subprocess

from ctm_ai.utils import extract_video_frames
from dataset_configs import get_dataset_config


def prepare_frames(
    file_list_path: str, source_folder: str, target_folder: str, max_frames: int
) -> None:
    """Extract frames from videos"""
    with open(file_list_path, 'r', encoding='utf-8') as json_file:
        sample_list = json.load(json_file)
    file_list = []
    for key in sample_list:
        file_list.append(key)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for test_file in file_list:
        video_path = f'{source_folder}/{test_file}.mp4'
        video_frames_path = f'{target_folder}/{test_file}_frames'
        extract_video_frames(video_path, video_frames_path, max_frames)


def prepare_audios(file_list_path: str, input_folder: str, output_folder: str) -> None:
    """Extract audio from videos"""
    with open(file_list_path, 'r', encoding='utf-8') as json_file:
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
            output_file = f'{base_name}_audio.mp4'
            output_path = os.path.join(output_folder, output_file)

            command = [
                'ffmpeg',
                '-i',
                input_path,
                '-vn',
                '-acodec',
                'copy',
                output_path,
            ]

            print(f'processing file: {file_name}')
            try:
                subprocess.run(command, check=True)
                print(f'audio saved: {output_file}')
            except subprocess.CalledProcessError as e:
                print(f'fail in extracting audio: {file_name}, error: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare input files (frame extraction, audio extraction)'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['urfunny', 'mustard'],
        help='Dataset name',
    )
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['frames', 'audios'],
        help='Task type: frames (extract frames) or audios (extract audio)',
    )
    parser.add_argument(
        '--video_folder',
        type=str,
        required=True,
        help='Video source folder',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Output folder',
    )
    parser.add_argument(
        '--dataset_file',
        type=str,
        default=None,
        help='Path to dataset JSON file (default: auto)',
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=10,
        help='Max number of frames to extract per video (for frame extraction only, default: 10)',
    )
    args = parser.parse_args()

    # Get dataset configuration
    config = get_dataset_config(args.dataset_name)

    # Set default dataset file if not specified
    if args.dataset_file is None:
        args.dataset_file = config.get_default_dataset_path()

    print(f'Dataset: {args.dataset_name}')
    print(f'Task: {args.task}')
    print(f'Video folder: {args.video_folder}')
    print(f'Output folder: {args.output_folder}')
    print(f'Dataset file: {args.dataset_file}')

    if args.task == 'frames':
        print(f'Max frames: {args.max_frames}')
        prepare_frames(
            args.dataset_file, args.video_folder, args.output_folder, args.max_frames
        )
    elif args.task == 'audios':
        prepare_audios(args.dataset_file, args.video_folder, args.output_folder)

    print('Done!')
