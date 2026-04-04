"""
CTM-AI Prompts Module

This module centralizes all prompts used across different processors:
- base_prompts:      Common/base prompts used across all processor types
- tool_prompts:      For ToolBench/Tool-Use tasks (API calling, tool execution)
- webagent_prompts:  For web-agent processors (axtree, html, screenshot)
"""

from .base_prompts import (
    BASE_JSON_FORMAT_FUSE,
    BASE_JSON_FORMAT_LINK_FORM,
    BASE_JSON_FORMAT_SCORE,
    build_base_score_format,
)
from .tool_prompts import (
    DEFAULT_NUM_ADDITIONAL_QUESTIONS,
    TOOLBENCH_SYSTEM_PROMPT,
    TOOLBENCH_TOOL_DECISION_PROMPT,
    build_tool_stage2_prompt,
)
from .webagent_prompts import (
    AXTREE_SYSTEM_PROMPT,
    HTML_SYSTEM_PROMPT,
    SCREENSHOT_SYSTEM_PROMPT,
    build_axtree_user_prompt,
    build_html_user_prompt,
    build_screenshot_user_prompt,
    parse_webagent_response,
)

__all__ = [
    # Base prompts
    'BASE_JSON_FORMAT_SCORE',
    'BASE_JSON_FORMAT_LINK_FORM',
    'BASE_JSON_FORMAT_FUSE',
    'build_base_score_format',
    # Tool prompts
    'DEFAULT_NUM_ADDITIONAL_QUESTIONS',
    'TOOLBENCH_SYSTEM_PROMPT',
    'TOOLBENCH_TOOL_DECISION_PROMPT',
    'build_tool_stage2_prompt',
    # Web-agent prompts
    'AXTREE_SYSTEM_PROMPT',
    'HTML_SYSTEM_PROMPT',
    'SCREENSHOT_SYSTEM_PROMPT',
    'build_axtree_user_prompt',
    'build_html_user_prompt',
    'build_screenshot_user_prompt',
    'parse_webagent_response',
]
