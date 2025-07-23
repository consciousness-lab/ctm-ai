from ctm_ai.processors import AudioProcessor

QUERY = query = """
You are an AI designed to detect sarcasm in audio by analyzing tone, pitch, intonation, and rhythm. Sarcasm often contrasts what is said with how it is said—such as a mocking tone, exaggerated emphasis, or unusual pitch shifts. Look for speech patterns where the voice rises or falls unexpectedly, or when intonation is exaggerated compared to the literal meaning.

However, be aware that some situations may involve humor or playful exaggeration, which might **not** be sarcasm.
For instance, if someone is joking or making fun in a playful manner with a lighthearted tone—such as a humorous story or exaggeration without irony—it may **not** be sarcasm.
Similarly, when the tone is consistent with the content, such as when someone expresses excitement, despair, unfortable, even if the words seem overly dramatic, it may be humor or exaggeration, not sarcasm.

Your task is to determine whether the speaker's tone suggests sarcasm or if the statement is just playful humor or exaggeration. Only vocal patterns that imply irony, exaggeration, or a clear contrast between spoken content and tone should be flagged as sarcasm. If the tone is lighthearted or playful, it may not be sarcasm. Based on the audio, does the speaker appear to be using sarcasm or not?
"""


def test_audio_processor_with_audio_file():
    processor = AudioProcessor(name='audio_processor')
    audio_path = '../exp_mustard/mustard_audios/2_464_audio.mp4'
    chunk = processor.ask(
        query=QUERY,
        audio_path=audio_path,
    )
    print('Results:', chunk.gist)


if __name__ == '__main__':
    test_audio_processor_with_audio_file()
