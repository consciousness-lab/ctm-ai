export GOOGLE_API_KEY=xxx # for language and vision
export OPENAI_API_KEY=xxx # for search
export GOOGLE_CSE_ID=xxx # for search
export GEMINI_API_KEY=xxx # for audio

poetry install
poetry run gunicorn app:app --bind 0.0.0.0:5000
