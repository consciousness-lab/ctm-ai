from ctm.processors.processor_base import BaseProcessor


def test_gpt4_processor() -> None:
    processor = BaseProcessor(name='gpt4_processor')
    chunk = processor.ask(
        query='what is 1+1?',
        text='1+1=2',
    )

    assert chunk.processor_name == 'gpt4_processor'
    assert chunk.time_step == 0
    assert chunk.gist is not None
    assert chunk.relavance >= 0 and chunk.relavance <= 1
    assert chunk.confidence >= 0 and chunk.confidence <= 1
    assert chunk.surprise >= 0 and chunk.surprise <= 1
    assert chunk.weight >= 0 and chunk.weight <= 1
    assert chunk.intensity >= 0 and chunk.intensity <= 1
    assert chunk.mood >= 0 and chunk.mood <= 1
