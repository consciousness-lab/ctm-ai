"""
CTM AI Tools Module

This module provides built-in tool functions that can be called by ToolExecutor,
including search and math capabilities as direct functions.
"""

from .builtin_tools import (
    get_builtin_tool_definitions,
    call_builtin_tool,
    generate_tool_question,
    search_web,
    calculate_math,
)

__all__ = [
    'get_builtin_tool_definitions',
    'call_builtin_tool',
    'generate_tool_question',
    'search_web',
    'calculate_math',
] 