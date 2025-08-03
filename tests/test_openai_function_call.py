from openai import OpenAI

inference_data = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_engine_query",
                "description": "Conducts a search based on the provided prompt and returns relevant real-time information, specific details, or general facts from various sources, including internet browsing and data from Google. Note that the provided function is in Python 3 syntax.",
                "parameters": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The search query to be executed by the search engine. Example: 'latest scientific discoveries'.",
                        },
                        "since_year": {
                            "type": "integer",
                            "description": "Filter results to include only information published after the specified year. Default to the current year if not provided.",
                            "default": 2023,
                        },
                        "source": {
                            "type": "string",
                            "description": "Preferred source for the search results, such as 'all', 'internet', or 'google'.",
                            "enum": ["all", "internet", "google"],
                            "default": "all",
                        },
                        "include_facts": {
                            "type": "boolean",
                            "description": "Indicates whether to include factual information in the search results. Default is true.",
                            "default": True,
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a given text prompt. The function is versatile and can be used for various general purposes without specializing in any specific category. Note that the provided function is in Python 3 syntax.",
                "parameters": {
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt that describes the desired content of the image.",
                        },
                        "image_format": {
                            "type": "string",
                            "description": "The file format for the output image.",
                            "enum": ["JPEG", "PNG", "BMP"],
                            "default": "PNG",
                        },
                        "width": {
                            "type": "integer",
                            "description": "The width of the generated image in pixels.",
                            "default": 1024,
                        },
                        "height": {
                            "type": "integer",
                            "description": "The height of the generated image in pixels.",
                            "default": 768,
                        },
                        "color_mode": {
                            "type": "string",
                            "description": "The color mode of the image.",
                            "enum": ["RGB", "Grayscale"],
                            "default": "RGB",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_human_image",
                "description": "Generates a realistic image of a human based on the provided prompt, with options for creating images representing different age groups and genders. Note that the provided function is in Python 3 syntax.",
                "parameters": {
                    "type": "object",
                    "required": ["prompt", "image_quality"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A descriptive text guiding the generation of the human image, such as 'a smiling girl with blue eyes' or 'an elderly man playing chess'.",
                        },
                        "image_quality": {
                            "type": "string",
                            "description": "The desired quality of the generated image, affecting the level of detail and realism. Options include 'low', 'medium', and 'high'.",
                            "enum": ["low", "medium", "high"],
                        },
                        "include_background": {
                            "type": "boolean",
                            "description": "Determines whether a background should be included in the generated image.",
                            "default": False,
                        },
                        "output_format": {
                            "type": "string",
                            "description": "Specifies the file format for the output image.",
                            "enum": ["JPEG", "PNG", "BMP"],
                            "default": "JPEG",
                        },
                    },
                },
            },
        },
    ],
}
client = OpenAI()
kwargs = {
    "messages": [
        {
            "role": "user",
            "content": "I'd like to have a digital painting created. The image should be a detailed portrait of a masked woman, including bright peacock feathers, with an elegant and highly detailed style. It should have a fluid illustration quality, with green highlighted lines and complex patterns, reminiscent of cyberpunk and Alphonse Mucha's art.",
        }
    ],
    "model": "o3-mini",
    "tools": inference_data["tools"],
}
api_response = client.chat.completions.create(**kwargs)
print(api_response)
breakpoint()
