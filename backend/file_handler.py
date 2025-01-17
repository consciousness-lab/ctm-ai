import os
import uuid
from typing import Optional

from config import Config
from flask import Flask
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


class FileHandler:
    @staticmethod
    def allowed_file(filename: str, file_type: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[
            1
        ].lower() in Config.ALLOWED_EXTENSIONS.get(file_type, set())

    @staticmethod
    def generate_unique_filename(filename: str) -> str:
        ext = filename.rsplit('.', 1)[1].lower()
        return f'{uuid.uuid4().hex}.{ext}'

    @staticmethod
    def save_file(file: FileStorage, file_type: str, app: Flask) -> Optional[str]:
        if file and FileHandler.allowed_file(file.filename or '', file_type):
            filename = secure_filename(file.filename or '')
            unique_filename = FileHandler.generate_unique_filename(filename)
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file_type, unique_filename
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            return unique_filename
        return None
