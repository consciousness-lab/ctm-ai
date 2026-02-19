"""
CTM-AI Prompts Module

This module centralizes all prompts used across different processors:
- base_prompts: Common/base prompts used across all processor types
- tool_prompts: For ToolBench/Tool-Use tasks (API calling, tool execution)
"""

from .base_prompts import (
    BASE_JSON_FORMAT_SCORE,
    BASE_JSON_FORMAT_LINK_FORM,
    BASE_JSON_FORMAT_FUSE,
    build_base_score_format,
)

from .tool_prompts import (
    DEFAULT_NUM_ADDITIONAL_QUESTIONS,
    TOOLBENCH_SYSTEM_PROMPT,
    TOOLBENCH_TOOL_DECISION_PROMPT,
    build_tool_stage2_prompt,
)

__all__ = [
    # Base prompts
    "BASE_JSON_FORMAT_SCORE",
    "BASE_JSON_FORMAT_LINK_FORM",
    "BASE_JSON_FORMAT_FUSE",
    "build_base_score_format",
    # Tool prompts
    "DEFAULT_NUM_ADDITIONAL_QUESTIONS",
    "TOOLBENCH_SYSTEM_PROMPT",
    "TOOLBENCH_TOOL_DECISION_PROMPT",
    "build_tool_stage2_prompt",
]
