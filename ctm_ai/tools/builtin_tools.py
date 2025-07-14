"""
Built-in Tools for CTM AI
=========================

This module provides built-in tool functions as direct implementations for search 
and math capabilities. These functions are used by ToolExecutor without requiring 
separate executor classes.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from litellm import completion
from newspaper import Article

from ..utils import logger


def search_web(query: str) -> str:
    """
    Web search tool function
    
    Args:
        query: Search query string
        
    Returns:
        Search results content
    """
    try:
        # Check required environment variables
        api_key = os.environ.get('GOOGLE_API_KEY')
        cse_id = os.environ.get('GOOGLE_CSE_ID')
        
        if not api_key or not cse_id:
            return "Error: Missing required environment variables GOOGLE_API_KEY or GOOGLE_CSE_ID"
        
        # Use LiteLLM to generate search keywords
        try:
            keywords_response = completion(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'user',
                        'content': f"Can you convert '{query}' into a search engine query that I can use to google? It should be keywords-type. You should only output the query for search engine, nothing else should be outputed.",
                    }
                ],
                max_tokens=100,
                temperature=0.0,
            )
            keywords = keywords_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate search keywords: {e}")
            keywords = query  # Use original query as fallback
        
        # Call Google Search API
        search_url = 'https://www.googleapis.com/customsearch/v1'
        params = {'key': api_key, 'cx': cse_id, 'q': keywords}
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        
        content = ''
        for item in search_results.get('items', []):
            title = item.get('title', '')
            link = item.get('link', '')
            
            try:
                # Get webpage content
                page_response = requests.get(link, timeout=5)
                page_response.raise_for_status()
                
                # Use newspaper to extract content
                article = Article(link)
                article.download()
                article.parse()
                article.nlp()
                page_summary = article.summary
                
            except Exception as e:
                logger.error(f'Failed to get page content {link}: {e}')
                page_summary = item.get('snippet', '')  # Use search result snippet as fallback
            
            info_str = f'Title: {title}\nSummary:\n{page_summary}\n\n'
            content += info_str
        
        # If no content, return basic error message
        if not content.strip():
            return f"No search results found for '{query}'"
        
        # Use LiteLLM to generate summary
        try:
            gist_response = completion(
                model='gpt-4o-mini',
                messages=[
                    {
                        'role': 'user',
                        'content': f"Can you summarize the following information into a single sentence: {content}. The output should be a one-sentence answer to '{keywords}'",
                    }
                ],
                max_tokens=100,
                temperature=0.0,
            )
            gist = gist_response.choices[0].message.content
            
            # Return detailed content and summary
            return f"Search Summary: {gist}\n\nDetailed Information:\n{content}"
            
        except Exception as e:
            logger.error(f"Failed to generate search summary: {e}")
            return content  # Return original content as fallback
            
    except requests.exceptions.HTTPError as err:
        logger.error(f'HTTP error: {err}')
        return f"Search request failed: {err}"
    except Exception as err:
        logger.error(f'Search error occurred: {err}')
        return f"Error occurred during search: {err}"


def calculate_math(expression: str) -> str:
    """
    Math calculation tool function
    
    Args:
        expression: Math expression or problem
        
    Returns:
        Calculation result
    """
    try:
        # Check required environment variables
        api_key = os.environ.get('WOLFRAM_API_KEY')
        
        if not api_key:
            return "Error: Missing required environment variable WOLFRAM_API_KEY"
        
        # Call Wolfram Alpha API
        url = 'http://api.wolframalpha.com/v2/query'
        params = {'input': expression, 'appid': api_key, 'output': 'json'}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        
        content = ''
        query_result = search_results.get('queryresult', {})
        
        if not query_result.get('success', False):
            return f"Unable to parse math expression: {expression}"
        
        # Extract text content from all pods
        for pod in query_result.get('pods', []):
            pod_title = pod.get('title', '')
            if pod_title:
                content += f"{pod_title}:\n"
            
            for subpod in pod.get('subpods', []):
                plaintext = subpod.get('plaintext', '')
                if plaintext:
                    content += f"{plaintext}\n"
            content += "\n"
        
        if not content.strip():
            return f"Wolfram Alpha returned no results for '{expression}'"
            
        return content.strip()
        
    except requests.exceptions.HTTPError as err:
        logger.error(f'HTTP error: {err}')
        return f"Math calculation request failed: {err}"
    except Exception as err:
        logger.error(f'Math calculation error occurred: {err}')
        return f"Error occurred during calculation: {err}"


def get_builtin_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get builtin tool function definitions (OpenAI format)
    
    Returns:
        List of builtin tool function definitions
    """
    return [
        {
            "name": "search_web",
            "description": "Search the internet for real-time information. Suitable for finding latest news, facts, definitions, or any query requiring up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string to search for, should be descriptive keywords"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculate_math",
            "description": "Execute mathematical calculations and solve math problems. Supports basic arithmetic, algebra, calculus, statistics, and various mathematical operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression or problem to calculate, can be equations, functions, numerical calculations, etc."
                    }
                },
                "required": ["expression"]
            }
        }
    ]


def call_builtin_tool(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Call builtin tool function
    
    Args:
        function_name: Function name
        arguments: Function parameters
        
    Returns:
        Function execution result
    """
    if function_name == "search_web":
        query = arguments.get("query", "")
        if not query:
            return "Error: Search query cannot be empty"
        return search_web(query)
    
    elif function_name == "calculate_math":
        expression = arguments.get("expression", "")
        if not expression:
            return "Error: Math expression cannot be empty"
        return calculate_math(expression)
    
    else:
        return f"Error: Unknown builtin tool function '{function_name}'"


def generate_tool_question(tool_name: str, original_query: str, tool_result: str) -> str:
    """
    Generate follow-up questions for tool calls
    
    Args:
        tool_name: Tool name
        original_query: Original query
        tool_result: Tool execution result
        
    Returns:
        Generated follow-up question
    """
    if tool_name == "search_web":
        questions = [
            f"Would you like me to search for more specific information about '{original_query}'?",
            f"Which specific aspect of the search results would you like to know more about?",
            f"Would you like me to search within a specific time range or geographic location?",
        ]
    elif tool_name == "calculate_math":
        questions = [
            f"Would you like me to explain the steps of this calculation process?",
            f"Would you like to calculate other related math problems?",
            f"Would you like me to provide more detailed mathematical analysis?",
        ]
    else:
        questions = [
            f"Regarding '{original_query}', what else can I help you with?",
            f"Would you like to use other tools to get more information?",
        ]
    
    # Simply choose the first question, could be more intelligent based on context
    return questions[0] 