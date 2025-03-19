import os
from typing import Any, List

import requests
from newspaper import Article

from ..messengers import Message
from ..utils import logger, message_exponential_backoff
from .executor_base import BaseExecutor
from openai import OpenAI


@BaseExecutor.register_executor('search_executor')
class SearchExecutor(BaseExecutor):
    def init_model(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.environ['GOOGLE_API_KEY']
        self.cse_id = os.environ['GOOGLE_CSE_ID']
        self.model = OpenAI()
        self.url = 'https://www.googleapis.com/customsearch/v1'

    @message_exponential_backoff()
    def ask(self, messages: List[Message], *args: Any, **kwargs: Any) -> Message:
        query = messages[-1].content
        keywords = self.model.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    'role': 'user',
                    'content': f"Can you convert '{query}' into a search engine query that I can use to google? It should be keywords-type. You should only output the query for search engine, nothing else should be outputed.",
                }
            ],
            max_tokens=100,
        ).choices[0].message.content
        params = {'key': self.api_key, 'cx': self.cse_id, 'q': keywords}
        try:
            response = requests.get(self.url, params=params, timeout=1)
            response.raise_for_status()
            search_results = response.json()

            content = ''
            gist = ''
            for item in search_results.get('items', []):
                title = item.get('title', '')
                link = item.get('link', '')

                try:
                    page_response = requests.get(link, params=params, timeout=1)
                    page_response.raise_for_status()

                    article = Article(link)
                    article.download()
                    article.parse()
                    article.nlp()
                    page_summary = article.summary

                except Exception as e:
                    logger.error(f'Failed to fetch the page {link}: {e}')
                    page_summary = ''

                info_str = f'Title: {title}\n Summary:\n{page_summary}\n\n'
                content += info_str

                gist = self.model.chat.completions.create(
                    model='gpt-4o-mini',
                    messages=[
                        {
                            'role': 'user',
                            'content': f"Can you summarize the following information into a single sentence: {content}. The output should be a one-sentence answer to '{keywords}'",
                        }
                    ],
                    max_tokens=100,
                ).choices[0].message.content


            return Message(role='assistant', content=content, gist=gist)
        except requests.exceptions.HTTPError as err:
            logger.error(f'HTTP error occurred: {err}')
            return Message(role='assistant', content='')
        except Exception as err:
            logger.error(f'Other error occurred: {err}')
            return Message(role='assistant', content='')
