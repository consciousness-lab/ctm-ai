import glob
import os
from typing import Dict, List, Optional, Union, cast

import google.generativeai as genai
from PIL import Image


class GeminiMultimodalLLM:
    def __init__(
        self,
        file_name: str,
        image_frames_folder: str,
        audio_file_path: str,
        context: str,
        query: str,
        model_name: str = 'gemini-1.5-pro',
    ) -> None:
        self.file_name = file_name
        self.image_frames_folder = image_frames_folder
        self.audio_file_path = audio_file_path
        self.context = context
        self.query = query
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = model_name

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(self.model_name)

        self.images = self._load_images()
        self.audio_file = self._upload_audio_file()

    def _load_images(self) -> List[Image.Image]:
        image_pattern = os.path.join(self.image_frames_folder, '*.jpg')
        image_paths = glob.glob(image_pattern)
        if not image_paths:
            raise ValueError(f'No images found in folder: {self.image_frames_folder}')

        image_paths_sorted = sorted(image_paths)

        images = []
        for img_path in image_paths_sorted:
            try:
                image = Image.open(img_path)
                images.append(image)
            except Exception as e:
                raise RuntimeError(f'Failed to load image {img_path}: {e}')
        return images

    def _upload_audio_file(self) -> Dict[str, Union[str, bytes]]:
        if not os.path.isfile(self.audio_file_path):
            raise ValueError(f'Audio file does not exist: {self.audio_file_path}')

        if not self.audio_file_path.lower().endswith('.mp4'):
            raise ValueError(
                f'Unsupported audio format for file: {self.audio_file_path}. Expected a .mp4 file.'
            )

        try:
            with open(self.audio_file_path, 'rb') as f:
                audio_data = f.read()
            return {'mime_type': 'audio/mp4', 'data': audio_data}
        except Exception as e:
            raise RuntimeError(f'Failed to upload audio {self.audio_file_path}: {e}')

    @staticmethod
    def _get_sarcasm_description() -> str:
        return (
            "Sarcasm is a form of communication where the literal meaning of a statement differs from the speaker's intent. "
            'It often conveys humor, mockery, or criticism indirectly. Key characteristics include:\n'
            '- **Humorous Effect:** Uses exaggeration or understatement to create humor.\n'
            '- **Criticism or Mockery:** Targets individuals or situations without direct confrontation.\n'
            '- **Social Commentary:** Highlights societal issues or absurdities.\n'
            '- **Verbal Forms:** Positive or neutral statements with negative or ironic meanings.\n'
            '- **Situational Forms:** Irony arising from unexpected outcomes.\n'
            '- **Recognition Cues:** Relies on tone of voice, facial expressions, and contextual knowledge.'
        )

    def generate_response(self) -> Optional[str]:
        prompt = (
            f'### Description of Sarcasm:\n{self._get_sarcasm_description()}\n\n'
            f'### Context:\n{self.context}\n\n'
            f'### Query:\n{self.query}\n\n'
            'Analyze the query based on the provided description of sarcasm, context, and multimodal inputs (video frames and audio). '
            'Provide a detailed and insightful response addressing the query comprehensively.'
        )

        inputs: List[Union[str, Image.Image, Dict[str, Union[str, bytes]]]] = [prompt]

        inputs.extend(self.images)

        inputs.append(self.audio_file)

        try:
            response = self.model.generate_content(inputs)
            return cast(Optional[str], getattr(response, 'text', None))
        except Exception as e:
            raise RuntimeError(f'Failed to generate response: {e}')
