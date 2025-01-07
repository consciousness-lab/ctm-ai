from ctm_ai.processors import BaseProcessor


def test_gpt4_processor() -> None:
    processor = BaseProcessor(name='language_processor')
    chunk = processor.ask(
        query='what is 1+1?',
        text='1+1=2',
    )

    assert chunk.processor_name == 'language_processor'
    assert chunk.time_step == 0
    assert chunk.gist is not None
    assert chunk.confidence >= 0 and chunk.confidence <= 1
    assert chunk.surprise >= 0 and chunk.surprise <= 1
    assert chunk.weight >= 0 and chunk.weight <= 1
    assert chunk.intensity >= 0 and chunk.intensity <= 1
    assert chunk.mood >= 0 and chunk.mood <= 1
