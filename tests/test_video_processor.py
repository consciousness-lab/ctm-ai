"""Test VideoProcessor with decomposed scoring."""

import os
from ctm_ai.processors import BaseProcessor


def test_video_processor():
    """Test VideoProcessor with a video file."""
    processor = BaseProcessor("video_processor")
    video_path = (
        "/Users/zhaoyining/Desktop/ctm-ai/exp_mustard/mustard_muted_videos/1_60.mp4"
    )

    if not os.path.exists(video_path):
        print(f"Test skipped: Video file not found at {video_path}")
        return

    # Check file size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"Testing video file: {video_path}")
    print(f"File size: {file_size_mb:.2f}MB")

    if file_size_mb > 20:
        print(f"Error: Video file exceeds 20MB limit (size: {file_size_mb:.2f}MB)")
        return

    query = "Analyze this video. Describe the scene and the characters in detail."

    chunk = processor.ask(
        query=query,
        video_path=video_path,
    )

    print("\nResponse:", chunk.gist)
    print("\nScores:")
    print(f"  Relevance:  {chunk.relevance:.2f}")
    print(f"  Confidence: {chunk.confidence:.2f}")
    print(f"  Surprise:   {chunk.surprise:.2f}")
    print(f"  Weight:     {chunk.weight:.2f}")

    if chunk.additional_question:
        print(f"\nAdditional Question: {chunk.additional_question}")

    # Assertions
    assert chunk is not None
    assert chunk.gist
    assert 0 <= chunk.relevance <= 1
    assert 0 <= chunk.confidence <= 1
    assert 0 <= chunk.surprise <= 1
    assert chunk.weight > 0

    print("\nTest passed!")
    return chunk


if __name__ == "__main__":
    test_video_processor()
