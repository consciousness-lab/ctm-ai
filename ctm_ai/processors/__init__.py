from .processor_api import APIProcessor
from .processor_audio import AudioProcessor
from .processor_axtree import AXTreeProcessor
from .processor_base import BaseProcessor
from .processor_code import CodeProcessor
from .processor_html import HTMLProcessor
from .processor_language import LanguageProcessor
from .processor_math import MathProcessor
from .processor_screenshot import ScreenshotProcessor
from .processor_search import SearchProcessor
from .processor_tool import ToolProcessor, register_tool_processors
from .processor_video import VideoProcessor
from .processor_vision import VisionProcessor
from .rapidapi_processors.processor_exercise import ExerciseProcessor
from .rapidapi_processors.processor_finance import FinanceProcessor
from .rapidapi_processors.processor_geodb import GeoDBProcessor
from .rapidapi_processors.processor_music import MusicProcessor
from .rapidapi_processors.processor_news import NewsProcessor
from .rapidapi_processors.processor_social import SocialProcessor
from .rapidapi_processors.processor_twitter import TwitterProcessor
from .rapidapi_processors.processor_weather import WeatherProcessor
from .rapidapi_processors.processor_youtube import YouTubeProcessor

__all__ = [
    'APIProcessor',
    'AudioProcessor',
    'AXTreeProcessor',
    'BaseProcessor',
    'CodeProcessor',
    'FinanceProcessor',
    'GeoDBProcessor',
    'HTMLProcessor',
    'LanguageProcessor',
    'MathProcessor',
    'NewsProcessor',
    'ScreenshotProcessor',
    'SearchProcessor',
    'SocialProcessor',
    'ToolProcessor',
    'TwitterProcessor',
    'VideoProcessor',
    'VisionProcessor',
    'WeatherProcessor',
    'YouTubeProcessor',
    'ExerciseProcessor',
    'MusicProcessor',
    'register_tool_processors',
]
