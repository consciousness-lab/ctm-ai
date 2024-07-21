from typing import List, Optional

from .message import Message
from .messenger_base import BaseMessenger


@BaseMessenger.register_messenger('gpt4_messenger')
class GPT4Messenger(BaseMessenger):
    def collect_executor_messages(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[List[str]] = None,
    ) -> List[Message]:
        content = """
        You need to infer and do reasoning on the provided text to provide supporting evidence that can be related to answering the query.
        Typically you need to provide a sentence of evidence and a question that you want to ask that is very important to answer the query based on the current insufficient information.
        You need to answer with the format of "\{'evidence': <evidence that is related to answering the query>, 'question': <question that is the most important to answer>\}"
        For example, if the query is "what is the job of the man" and the provided text is "the man lives in new york and works from day to night".
        Then you need to answer that "\{'evidence': Long working hour and living in New York can be related to financial job or cleaning job or jobs that requires a lot of time". 'question': "how is his salary?"\}
        You should never say that you are not sure about something. You should always try to infer and provide a logical answer.
        Additionally, you should make the answer as short as possible.
        Provide potentially helpful information that you infer from the given information related to the query "{}"\n'.format(query)
        """
        if text is not None:
            content += 'Text: {}\n'.format(text)
        message = Message(
            role='user',
            content=content,
        )
        self.executor_messages.append(message)
        return self.executor_messages

    def collect_scorer_messages(
        self,
        executor_output: Message,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video_frames: Optional[List[str]] = None,
    ) -> List[Message]:
        message = Message(
            role='user',
            query=query,
            gist=executor_output.gist,
            gists=executor_output.gists,
        )
        self.scorer_messages.append(message)
        return self.scorer_messages
